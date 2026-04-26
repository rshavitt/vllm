# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManagerNixl: NIXL-backed disk secondary tier for KV cache offloading.

Mirrors the structure of FileSystemTierManager (C++ tier) exactly — same file layout,
LRU tracking, in-flight state, and job lifecycle — but performs I/O via NIXL agents
using the POSIX backend instead of a C++ extension.

Thread pool:
    Two queues (read, write) and two sets of threads:
      - Read-priority threads: drain the read queue first, then the write queue.
      - Write-priority threads: drain the write queue first, then the read queue.
    Each thread holds a dedicated NIXL agent (NIXL agents are not thread-safe).
    Load jobs are enqueued to the read queue; store jobs to the write queue.

Store path:
    Each block is written atomically via a NIXL WRITE transfer:
    data is transferred to a temp file, then os.replace'd to the final path.

Load path:
    Each block is read from disk into the CPU buffer via a NIXL READ transfer.

File naming:  <base_path>/<hhh>/<hh>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)

NIXL agents and the thread pool are created lazily on first use.
"""

import collections
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import ctypes

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec
try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    raise ImportError(
        "FileSystemTierManagerNixl requires the 'nixl' package. "
        "Install it with: pip install nixl"
    ) from e
    
logger = init_logger(__name__)



# ---------------------------------------------------------------------------
# NixlDualQueuePool — two queues, two priority classes of threads
# Each thread carries a dedicated NIXL agent; tasks receive it as argument.
# ---------------------------------------------------------------------------

class _NixlDualQueuePool:
    """
    Thread pool with two task queues (read and write) and two thread groups.

    Read-priority threads drain the read queue first, then fall back to the
    write queue.  Write-priority threads do the reverse.  Both queues share
    a single condition variable (same as the C++ DualQueueThreadPool).

    Each thread is assigned a dedicated NIXL agent at creation time and passes
    it to every task it runs (NIXL agents are not thread-safe).

    Tasks submitted to this pool must accept a single positional argument:
        task(agent) -> None
    """

    def __init__(
        self,
        n_read_prio: int,
        n_write_prio: int,
        make_agent,  # callable(thread_index: int) -> nixl_agent
        thread_name_prefix: str = "vllm_nixl_dq",
    ) -> None:
        self._read_q: collections.deque = collections.deque()
        self._write_q: collections.deque = collections.deque()
        self._cv = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []

        # Create all agents upfront so each thread gets its own.
        agents = [make_agent(i) for i in range(n_read_prio + n_write_prio)]

        for i in range(n_read_prio):
            t = threading.Thread(
                target=self._worker,
                args=(True, agents[i]),
                name=f"{thread_name_prefix}_r{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_prio):
            t = threading.Thread(
                target=self._worker,
                args=(False, agents[n_read_prio + i]),
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

    def _worker(self, read_priority: bool, agent) -> None:
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
            task(agent)


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
# Memory address helper
# ---------------------------------------------------------------------------

def _memview_addr(view: memoryview) -> int:
    """Return the memory address of a writable memoryview."""
    view_1d = view.cast("B") if view.ndim > 1 else view
    return ctypes.addressof(ctypes.c_char.from_buffer(view_1d))


# ---------------------------------------------------------------------------
# NIXL I/O helpers
# ---------------------------------------------------------------------------

def _nixl_write_block(
    agent,
    block_size: int,
    buffer_addr: int,
    block_idx: int,
    dest_fname: str,
    tmp_fname: str,
) -> None:
    """
    Write one KV block at *buffer_addr + block_idx * block_size* to *dest_fname*
    via a NIXL WRITE transfer, using *tmp_fname* for atomic rename.
    """
    fd = -1
    xfer_handle = None
    file_handle = None
    dram_handle = None
    try:
        os.makedirs(os.path.dirname(dest_fname), exist_ok=True)
        fd = os.open(
            tmp_fname,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )

        block_addr = buffer_addr + block_idx * block_size
        local_descs = [(block_addr, block_size, 0)]
        remote_descs = [(0, block_size, fd, "")]

        dram_handle = agent.register_memory(
            [(block_addr, block_size, 0, "")], mem_type="DRAM", backends=["POSIX"]
        )
        local_dl = agent.get_xfer_descs(local_descs, mem_type="DRAM")
        file_handle = agent.register_memory(
            remote_descs, mem_type="FILE", backends=["POSIX"]
        )
        xfer_handle = agent.initialize_xfer(
            operation="WRITE",
            local_descs=local_dl,
            remote_descs=file_handle.trim(),
            remote_agent=agent.name,
            backends=["POSIX"],
        )
        agent.transfer(xfer_handle)

        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            if state == "ERR":
                raise RuntimeError(
                    f"NIXL WRITE transfer failed for block {block_idx} "
                    f"-> {dest_fname}"
                )

        os.close(fd)
        fd = -1
        os.replace(tmp_fname, dest_fname)

    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    finally:
        if xfer_handle is not None:
            agent.release_xfer_handle(xfer_handle)
        if file_handle is not None:
            agent.deregister_memory(file_handle)
        if dram_handle is not None:
            agent.deregister_memory(dram_handle)


def _nixl_read_block(
    agent,
    block_size: int,
    buffer_addr: int,
    block_idx: int,
    source_fname: str,
) -> None:
    """
    Read one KV block from *source_fname* into the buffer at
    *buffer_addr + block_idx * block_size* via a NIXL READ transfer.
    """
    fd = -1
    xfer_handle = None
    file_handle = None
    dram_handle = None
    try:
        fd = os.open(source_fname, os.O_RDONLY | os.O_DIRECT)

        block_addr = buffer_addr + block_idx * block_size
        local_descs = [(block_addr, block_size, 0)]
        remote_descs = [(0, block_size, fd, "")]

        dram_handle = agent.register_memory(
            [(block_addr, block_size, 0, "")], mem_type="DRAM", backends=["POSIX"]
        )
        local_dl = agent.get_xfer_descs(local_descs, mem_type="DRAM")
        file_handle = agent.register_memory(
            remote_descs, mem_type="FILE", backends=["POSIX"]
        )
        xfer_handle = agent.initialize_xfer(
            operation="READ",
            local_descs=local_dl,
            remote_descs=file_handle.trim(),
            remote_agent=agent.name,
            backends=["POSIX"],
        )
        agent.transfer(xfer_handle)

        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            if state == "ERR":
                raise RuntimeError(
                    f"NIXL READ transfer failed for block {block_idx} "
                    f"<- {source_fname}"
                )

        os.close(fd)
        fd = -1

    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    finally:
        if xfer_handle is not None:
            agent.release_xfer_handle(xfer_handle)
        if file_handle is not None:
            agent.deregister_memory(file_handle)
        if dram_handle is not None:
            agent.deregister_memory(dram_handle)


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
# FileSystemTierManagerNixl
# ---------------------------------------------------------------------------

class FileSystemTierManagerNixl(SecondaryTierManager):
    """
    NIXL-backed disk secondary tier.

    Mirrors FileSystemTierManager (C++ tier) in structure: same file naming,
    LRU eviction, in-flight tracking, and job lifecycle.  I/O is performed via
    NIXL agents using the POSIX backend, one task per block file.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.  Each thread holds a dedicated NIXL agent.

    The thread pool and agents are initialised lazily on first store/load call.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished() polls job completion and returns completed JobResults.

    Eviction:
        LRU via OrderedDict. Blocks in-flight are never evicted.
    """

    def __init__(
        self,
        base_path: str,
        max_blocks: int,
        tier_name: str = "StorageNixl",
        n_read_agents: int = 4,
        n_write_agents: int = 4,
        tmp_dir: str | None = None,
    ):
        """
        Args:
            base_path: Root directory for block files.
            max_blocks: Maximum number of blocks to keep indexed (LRU eviction).
            tier_name: Identifier string returned by get_tier_name().
            n_read_agents: Number of read-priority threads (each with its own
                NIXL agent).
            n_write_agents: Number of write-priority threads (each with its own
                NIXL agent).
            tmp_dir: Directory for atomic-write temp files. Defaults to
                base_path. Must be on the same filesystem as base_path for
                os.replace() to be atomic.
        """
        self._base_path = base_path
        self._max_blocks = max_blocks
        self._tier_name = tier_name
        self._n_read_agents = n_read_agents
        self._n_write_agents = n_write_agents
        self._tmp_dir = tmp_dir or base_path

        # Pool is created lazily on first store/load call.
        self._pool: _NixlDualQueuePool | None = None

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
    # Pool / agent management
    # ------------------------------------------------------------------

    def _make_nixl_agent(self, index: int):
        """Create one NIXL agent. Called once per thread at pool startup."""
        conf = nixl_agent_config(
            enable_prog_thread=True,
            backends=["POSIX"],
        )
        return nixl_agent(
            agent_name=f"vllm_kv_nixl_{self._tier_name}_{index}",
            nixl_conf=conf,
            instantiate_all=False,
        )

    def _get_pool(self) -> _NixlDualQueuePool:
        """Lazily create the dual-queue NIXL thread pool."""
        if self._pool is None:
            self._pool = _NixlDualQueuePool(
                self._n_read_agents,
                self._n_write_agents,
                self._make_nixl_agent,
                thread_name_prefix="vllm_kv_nixl_fs",
            )
        return self._pool

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

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit a store job: enqueue one write task per block to the write queue.
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
            return

        keys_to_store, block_ids_to_store = map(list, zip(*pairs))

        # LRU eviction: free space for the incoming blocks.
        num_to_evict = len(keys_to_store) - (
            self._max_blocks - len(self._blocks)
        )
        if num_to_evict > 0:
            if self._evictable_count < num_to_evict:
                logger.warning(
                    "FileSystemTierManagerNixl(%s): insufficient evictable "
                    "blocks (%d evictable, %d needed); dropping store job %s.",
                    self._tier_name,
                    self._evictable_count,
                    num_to_evict,
                    job_id,
                )
                return

            protected = set(all_keys)
            evicted = []
            for key in self._blocks:  # oldest-first (OrderedDict)
                if key not in protected and key not in self._in_flight:
                    evicted.append(key)
                    if len(evicted) == num_to_evict:
                        break
            else:
                logger.warning(
                    "FileSystemTierManagerNixl(%s): could not find %d "
                    "evictable blocks (protected overlap reduced candidates); "
                    "dropping store job %s.",
                    self._tier_name,
                    num_to_evict,
                    job_id,
                )
                return
            for key in evicted:
                file_path = self.get_file_name(get_offload_block_hash(key))
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                del self._blocks[key]
                self._evictable_count -= 1

        # Cast to 1D byte view and obtain the base memory address.
        # The cast shares the same memory as self._primary_view (no copy).
        buffer = self._primary_view.cast("B")
        buffer_addr = _memview_addr(buffer)
        state = _JobState(len(keys_to_store))

        pool = self._get_pool()
        for key, bid in zip(keys_to_store, block_ids_to_store):
            dest_path = self.get_file_name(get_offload_block_hash(key))
            tmp_path = self._tmp_path(dest_path, job_id)

            def _make_task(
                bs=self._block_size, ba=buffer_addr, bi=bid,
                dp=dest_path, tp=tmp_path, st=state,
            ):
                def task(agent):
                    try:
                        _nixl_write_block(agent, bs, ba, bi, dp, tp)
                        st.task_done(True)
                    except Exception as exc:
                        st.task_done(False, exc)
                return task

            pool.enqueue_write(_make_task())

        for key in keys_to_store:
            self._in_flight[key] = job_id

        self._active_jobs[job_id] = _ActiveJob(
            is_store=True,
            keys=keys_to_store,
            buffer=buffer,
            state=state,
        )

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
                    "FileSystemTierManagerNixl(%s): block %s not found on "
                    "disk; dropping load job %s.",
                    self._tier_name,
                    key,
                    job_id,
                )
                return
        
        buffer = self._primary_view.cast("B")
        buffer_addr = _memview_addr(buffer)
        state = _JobState(len(keys))

        pool = self._get_pool()
        for key, bid in zip(keys, block_ids):
            source_path = self.get_file_name(get_offload_block_hash(key))

            def _make_task(bs=self._block_size, ba=buffer_addr, bi=bid,
                           sp=source_path, st=state):
                def task(agent):
                    try:
                        _nixl_read_block(agent, bs, ba, bi, sp)
                        st.task_done(True)
                    except Exception as exc:
                        st.task_done(False, exc)
                return task

            pool.enqueue_read(_make_task())

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
                    "FileSystemTierManagerNixl(%s): job %s NIXL I/O "
                    "failed: %s",
                    self._tier_name,
                    job_id,
                    err,
                )

            # Release memoryview now that all I/O is done.
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
