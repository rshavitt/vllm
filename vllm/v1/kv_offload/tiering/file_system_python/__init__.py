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
    (<dest_path_with_.tmp_instead_of_.bin>) via os.write, then os.replace'd to the final path.

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>/<hhh>/<hh>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import collections
import functools
import os
import threading
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)

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


# ---------------------------------------------------------------------------
# Per-block I/O callbacks — module-level so they are not re-created each loop
# ---------------------------------------------------------------------------

def _store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    state: "_JobState",
) -> None:
    """
    Store callback: write one KV block atomically, optionally checking if it exists first.
    
    Writes to a temp file then atomically replaces the destination.
    Uses O_DIRECT with page-aligned bounce buffer for kernel bypass.
    """
    tmp_path = dest_path.replace('.bin', '.tmp')
    try:
        # Check if block already exists to avoid redundant writes
        if os.path.exists(dest_path):
            state.task_done(True)
            return
        
        # Write block atomically
        _ensure_dirs(dest_path)
        slice = buffer[offset: offset + block_size]
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )
        try:
            os.write(fd, slice)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
        state.task_done(True)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        state.task_done(False, exc)


def _load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    state: "_JobState",
) -> None:
    """
    Load callback: read one KV block from disk.
    """
    try:
        fd = os.open(source_path, os.O_RDONLY | os.O_DIRECT)
        slice = view[offset: offset + block_size]
        try:
            os.readv(fd, [slice])
        finally:
            os.close(fd)
        state.task_done(True)
    except Exception as exc:
        state.task_done(False, exc)

# ---------------------------------------------------------------------------
# FileSystemTierManagerPython
# ---------------------------------------------------------------------------

class FileSystemTierManagerPython(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished() polls job completion and returns completed JobResults.

    """

    def __init__(
        self,
        base_path: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            base_path: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
        """
        self._base_path = base_path

        self._pool = _DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        # job_id -> _JobState for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, _JobState] = {}

        # Long-lived memoryview of the primary CPU tensor (set once by TieringOffloadingManager).
        self._primary_view: memoryview | None = None
        self._block_size: int = 0

    def get_file_name(self, key: OffloadKey) -> str:
        """
        Return the file path for a KV block.
        <base>/<hhh>/<hh>/<hash>.bin — hash-based subdirectories to limit
        directory fan-out.
        """
        block_hash = get_offload_block_hash(key)
        block_hash_hex = block_hash[:8].hex()
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self._base_path}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

    def set_primary_view(self, view: memoryview) -> None:
        self._block_size = view.strides[0]  # type: ignore
        self._primary_view = view.cast("B")

    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None) -> bool | None:
        file_path = self.get_file_name(key)
        return os.path.exists(file_path)

    def submit_store(self, job_metadata: JobMetadata) -> bool:
        """
        Submit a store job: enqueue one write task per block to the write queue.

        Returns True if an async job was submitted, False if dropped.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_store()"
        )

        # TODO: remove or change tests if decideing to move this check to workers
        if all(self.lookup(key) for key in job_metadata.keys):
            return False

        state = _JobState(len(job_metadata.keys))

        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            dest_path = self.get_file_name(key)
            offset = bid * self._block_size
            self._pool.enqueue_write(
                functools.partial(_store_block, dest_path, self._primary_view, offset, self._block_size, state)
            )

        self._active_jobs[job_metadata.job_id] = state
        # TODO: remove after changing the method to return None after changing the API.
        return True

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job: enqueue one read task per block to the read queue.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_load()"
        )
        block_ids = [int(bid) for bid in job_metadata.block_ids]
        state = _JobState(len(job_metadata.keys))

        for key, bid in zip(job_metadata.keys, block_ids):
            source_path = self.get_file_name(key)
            offset = bid * self._block_size
            self._pool.enqueue_read(
                functools.partial(_load_block, source_path, self._primary_view, offset, self._block_size, state)
            )

        self._active_jobs[job_metadata.job_id] = state

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs by polling _JobState.is_done.
        """
        results: list[JobResult] = []

        for job_id, state in list(self._active_jobs.items()):
            if not state.is_done:
                continue

            success = state.success
            for err in state.errors:
                logger.error(
                    "FileSystemTierManagerPython: job %s block I/O "
                    "failed: %s",
                    job_id,
                    err,
                )

            del self._active_jobs[job_id]

            results.append(JobResult(job_id=job_id, success=success))

        return results

    @staticmethod
    def get_tier_type() -> str:
        return "file_system_python"
