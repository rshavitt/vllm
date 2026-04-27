# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManagerPython: Pure-Python disk-backed secondary tier for KV cache offloading.

Mirrors the structure of FileSystemTierManager (C++ tier) exactly — same file layout,
LRU tracking, in-flight state, and job lifecycle — but performs I/O via a Python
DualQueueThreadPool instead of the C++ extension.

Thread pool:
    Two queues (read, write) and two sets of threads:
      - Read-priority threads: drain the read queue first, then the write queue.
      - Write-priority threads: drain the write queue first, then the read queue.
    Load jobs are enqueued to the read queue; store jobs to the write queue.

Store path:
    Each block is written atomically: data is written to a temp file
    (<dest>.{job_id}.tmp) via os.write, then os.replace'd to the final path.

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>/<hhh>/<hh>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import collections
import functools
import mmap
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# DualQueueThreadPool — two queues, two priority classes of threads
# ---------------------------------------------------------------------------

class _DualQueueThreadPool:
    """
    Thread pool with two task queues (read and write) and two thread groups.

    Read-priority threads drain the read queue first, then fall back to the
    write queue.  Write-priority threads do the reverse.  Both queues share
    a single condition variable (same as the C++ DualQueueThreadPool).
    """

    def __init__(
        self,
        n_read_prio: int,
        n_write_prio: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._read_q: collections.deque = collections.deque()
        self._write_q: collections.deque = collections.deque()
        self._cv = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []

        for i in range(n_read_prio):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_r{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_prio):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_w{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_read(self, fn) -> None:
        """Enqueue a read task (high-priority for read-priority threads)."""
        with self._cv:
            self._read_q.append(fn)
            self._cv.notify()

    def enqueue_write(self, fn) -> None:
        """Enqueue a write task (high-priority for write-priority threads)."""
        with self._cv:
            self._write_q.append(fn)
            self._cv.notify()

    def shutdown(self, wait: bool = True) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, read_priority: bool) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: self._stop or self._read_q or self._write_q
                )
                if self._stop and not self._read_q and not self._write_q:
                    return
                primary   = self._read_q  if read_priority else self._write_q
                secondary = self._write_q if read_priority else self._read_q
                task = primary.popleft() if primary else secondary.popleft()
            task()


# ---------------------------------------------------------------------------
# _JobState — tracks per-block task completion for one job
# ---------------------------------------------------------------------------

class _JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(ok) when it finishes.  The scheduler thread
    polls is_done to detect completion.
    """
    __slots__ = ("_total", "_completed", "_success", "_errors", "_lock")

    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0
        self._success = True
        self._errors: list[Exception] = []
        self._lock = threading.Lock()

    def task_done(self, ok: bool, exc: Exception | None = None) -> None:
        with self._lock:
            if not ok:
                self._success = False
                if exc is not None:
                    self._errors.append(exc)
            self._completed += 1

    @property
    def is_done(self) -> bool:
        # _completed is monotonically increasing; reading without a lock is
        # safe for a poll: we may see it one cycle late at worst.
        return self._completed >= self._total

    @property
    def success(self) -> bool:
        return self._success

    @property
    def errors(self) -> list[Exception]:
        return list(self._errors)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _write_block(
    tmp_path: str,
    dest_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Write one KV block from *view[offset:offset+block_size]* to *dest_path*
    atomically (write to tmp_path, os.replace).

    O_DIRECT requires a page-aligned buffer; data is bounced through an
    anonymous mmap so the kernel bypass is preserved without posix_memalign.
    """
    _ensure_dirs(dest_path)
    aligned_size = -(-block_size // mmap.PAGESIZE) * mmap.PAGESIZE
    bounce = mmap.mmap(-1, aligned_size)
    mv = memoryview(bounce)
    try:
        mv[:block_size] = view[offset: offset + block_size]
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )
        try:
            os.write(fd, mv[:aligned_size])
        finally:
            os.close(fd)
    finally:
        mv.release()
        bounce.close()
    os.replace(tmp_path, dest_path)


def _read_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Read one KV block from *source_path* into *view[offset:offset+block_size]*.
    """
    aligned_size = -(-block_size // mmap.PAGESIZE) * mmap.PAGESIZE
    bounce = mmap.mmap(-1, aligned_size)
    mv = memoryview(bounce)
    try:
        fd = os.open(source_path, os.O_RDONLY | os.O_DIRECT)
        try:
            os.readv(fd, [mv[:aligned_size]])
        finally:
            os.close(fd)
        view[offset: offset + block_size] = mv[:block_size]
    finally:
        mv.release()
        bounce.close()


# ---------------------------------------------------------------------------
# Per-block I/O callbacks — module-level so they are not re-created each loop
# ---------------------------------------------------------------------------

def _run_store(
    tmp_path: str,
    dest_path: str,
    buf: memoryview,
    offset: int,
    block_size: int,
    state: "_JobState",
) -> None:
    try:
        _write_block(tmp_path, dest_path, buf, offset, block_size)
        state.task_done(True)
    except Exception as exc:
        state.task_done(False, exc)


def _run_load(
    source_path: str,
    buf: memoryview,
    offset: int,
    block_size: int,
    state: "_JobState",
) -> None:
    try:
        _read_block(source_path, buf, offset, block_size)
        state.task_done(True)
    except Exception as exc:
        state.task_done(False, exc)


# ---------------------------------------------------------------------------
# _ActiveJob
# ---------------------------------------------------------------------------

@dataclass
class _ActiveJob:
    """A job whose per-block tasks have been enqueued in the thread pool."""
    is_store: bool
    keys: list[OffloadKey]
    buffer: memoryview  # kept alive until state.is_done
    state: _JobState


# ---------------------------------------------------------------------------
# FileSystemTierManagerPython
# ---------------------------------------------------------------------------

class FileSystemTierManagerPython(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Mirrors FileSystemTierManager (C++ tier) in structure: same file naming,
    LRU eviction, in-flight tracking, and job lifecycle.  I/O is performed by
    a DualQueueThreadPool (one task per block file).

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished() polls job completion and returns completed JobResults.

    Eviction:
        LRU via OrderedDict. Blocks in-flight are never evicted.
    """

    def __init__(
        self,
        base_path: str,
        max_blocks: int,
        tier_name: str = "StoragePython",
        n_read_threads: int = 32,
        n_write_threads: int = 16,
        tmp_dir: str | None = None,
    ):
        """
        Args:
            base_path: Root directory for block files.
            max_blocks: Maximum number of blocks to keep indexed (LRU eviction).
            tier_name: Identifier string returned by get_tier_name().
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            tmp_dir: Directory for atomic-write temp files. Defaults to
                base_path. Must be on the same filesystem as base_path for
                os.replace() to be atomic.
        """
        self._base_path = base_path
        self._max_blocks = max_blocks
        self._tier_name = tier_name
        self._tmp_dir = tmp_dir or base_path

        self._pool = _DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        # LRU ordered set: block_hash -> True
        self._blocks: OrderedDict[OffloadKey, bool] = OrderedDict()

        # block_hash -> job_id (prevents eviction mid-transfer)
        self._in_flight: dict[OffloadKey, JobId] = {}

        # job_id -> _ActiveJob for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, _ActiveJob] = {}

        # Number of blocks in _blocks that are NOT currently in _in_flight.
        self._evictable_count: int = 0

        # Long-lived memoryview of the primary CPU tensor (set once by TieringOffloadingManager).
        self._primary_view: memoryview | None = None

    # ------------------------------------------------------------------
    # File-naming (identical to C++ tier)
    # ------------------------------------------------------------------

    def get_file_name(self, block_hash: int | bytes) -> str:
        """
        Return the file path for a KV block.
        <base>/<hhh>/<hh>/<hash>.bin — hash-based subdirectories to limit
        directory fan-out.
        """
        if isinstance(block_hash, bytes):
            block_hash = int.from_bytes(block_hash, "big")
        if not isinstance(block_hash, int):
            raise TypeError(
                f"block_hash must be int or bytes, got {type(block_hash).__name__}"
            )
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self._base_path}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

    def _tmp_path(self, dest_path: str, job_id: JobId) -> str:
        """Return a temp path for atomic writes."""
        return f"{dest_path}.{job_id}.tmp"

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    def set_primary_view(self, view: memoryview) -> None:
        self._primary_view = view
        self._block_size = view.strides[0]

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if key in self._in_flight:
            return None
        return key in self._blocks

    def submit_store(self, job_metadata: JobMetadata) -> bool:
        """
        Submit a store job: enqueue one write task per block to the write queue.

        Returns True if an async job was submitted, False if dropped.
        """
        assert isinstance(job_metadata.spec, CPULoadStoreSpec), (
            f"Expected CPULoadStoreSpec, got {type(job_metadata.spec)}"
        )
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_store()"
        )
        spec: CPULoadStoreSpec = job_metadata.spec
        job_id = job_metadata.job_id
        all_keys = job_metadata.keys

        # Keep only blocks not yet on disk and not currently being stored.
        pairs = [
            (key, int(bid))
            for key, bid in zip(all_keys, spec.block_ids)
            if key not in self._blocks and key not in self._in_flight
        ]
        if not pairs:
            return False

        keys_to_store    = [key for key, _   in pairs]
        block_ids_to_store = [bid for _,  bid in pairs]

        # LRU eviction: free space for the incoming blocks.
        num_to_evict = len(keys_to_store) - (
            self._max_blocks - len(self._blocks)
        )
        if num_to_evict > 0:
            if self._evictable_count < num_to_evict:
                logger.warning(
                    "FileSystemTierManagerPython(%s): insufficient evictable "
                    "blocks (%d evictable, %d needed); dropping store job %s.",
                    self._tier_name,
                    self._evictable_count,
                    num_to_evict,
                    job_id,
                )
                return False

            protected = set(all_keys)
            evicted = [
                key for key in self._blocks
                if key not in protected and key not in self._in_flight
            ][:num_to_evict]
            if len(evicted) < num_to_evict:
                logger.warning(
                    "FileSystemTierManagerPython(%s): could not find %d "
                    "evictable blocks (protected overlap reduced candidates); "
                    "dropping store job %s.",
                    self._tier_name,
                    num_to_evict,
                    job_id,
                )
                return False
            for key in evicted:
                file_path = self.get_file_name(get_offload_block_hash(key))
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                del self._blocks[key]
                self._evictable_count -= 1

        # Cast to 1D byte view so os.write receives a flat buffer.
        # The cast shares the same memory as self._primary_view (no copy).
        buffer = self._primary_view.cast("B")
        state = _JobState(len(keys_to_store))

        for key, bid in zip(keys_to_store, block_ids_to_store):
            dest_path = self.get_file_name(get_offload_block_hash(key))
            tmp_path = self._tmp_path(dest_path, job_id)
            offset = bid * self._block_size
            self._pool.enqueue_write(
                functools.partial(_run_store, tmp_path, dest_path, buffer, offset, self._block_size, state)
            )

        for key in keys_to_store:
            self._in_flight[key] = job_id

        self._active_jobs[job_id] = _ActiveJob(
            is_store=True,
            keys=keys_to_store,
            buffer=buffer,
            state=state,
        )
        return True

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job: enqueue one read task per block to the read queue.
        """
        assert isinstance(job_metadata.spec, CPULoadStoreSpec), (
            f"Expected CPULoadStoreSpec, got {type(job_metadata.spec)}"
        )
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_load()"
        )
        spec: CPULoadStoreSpec = job_metadata.spec
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = [int(bid) for bid in spec.block_ids]

        for key in keys:
            if key not in self._blocks:
                logger.warning(
                    "FileSystemTierManagerPython(%s): block %s not found on "
                    "disk; dropping load job %s.",
                    self._tier_name,
                    key,
                    job_id,
                )
                return

        # Cast to 1D byte view so os.readv receives a flat writable buffer.
        buffer = self._primary_view.cast("B")
        state = _JobState(len(keys))

        for key, bid in zip(keys, block_ids):
            source_path = self.get_file_name(get_offload_block_hash(key))
            offset = bid * self._block_size
            self._pool.enqueue_read(
                functools.partial(_run_load, source_path, buffer, offset, self._block_size, state)
            )

        for key in keys:
            self._in_flight[key] = job_id
            self._evictable_count -= 1

        self._active_jobs[job_id] = _ActiveJob(
            is_store=False,
            keys=keys,
            buffer=buffer,
            state=state,
        )

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs by polling _JobState.is_done.
        """
        results: list[JobResult] = []

        for job_id, job in list(self._active_jobs.items()):
            if not job.state.is_done:
                continue

            success = job.state.success
            for err in job.state.errors:
                logger.error(
                    "FileSystemTierManagerPython(%s): job %s block I/O "
                    "failed: %s",
                    self._tier_name,
                    job_id,
                    err,
                )

            job.buffer.release()
            del self._active_jobs[job_id]

            for key in job.keys:
                del self._in_flight[key]
                if job.is_store and success:
                    self._blocks[key] = True
                    self._evictable_count += 1
                elif not job.is_store:
                    # Load complete (success or failure): block stays in
                    # _blocks and is no longer in-flight → evictable again.
                    self._evictable_count += 1

            results.append(JobResult(job_id=job_id, success=success))

        return results

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self._blocks:
                self._blocks.move_to_end(key)

    def get_tier_name(self) -> str:
        return self._tier_name

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_num_blocks(self) -> int:
        return len(self._blocks)

    def get_num_in_flight(self) -> int:
        return len(self._in_flight)

