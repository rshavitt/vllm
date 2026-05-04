# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManagerNixl: NIXL-backed disk secondary tier for KV cache offloading.

Thread pool:
    Two queues (read, write) and two sets of threads:
      - Read-priority threads: drain the read queue first, then the write queue.
      - Write-priority threads: drain the write queue first, then the read queue.
    Each thread holds a dedicated NIXL agent (NIXL agents are not thread-safe).
    Load jobs are enqueued to the read queue; store jobs to the write queue.

Store path:
    Each block is written atomically via a NIXL WRITE transfer:
    data is transferred to a temp file (<dest_path_with_.tmp_instead_of_.bin>),
    then os.replace'd to the final path.

Load path:
    Each block is read from disk into the CPU buffer via a NIXL READ transfer.

File naming:  <base_path>/<hhh>/<hh>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)

NIXL agents and the thread pool are created eagerly during initialization.
"""

import collections
import os
import threading
from collections.abc import Iterable

import ctypes

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
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

    Tasks submitted to this pool must accept two positional arguments:
        task(agent, dram_handle) -> None
    where dram_handle is the pre-registered DRAM memory handle for that agent.
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
        
        # Store agents and their DRAM handles (set later via register_dram_buffer)
        self._agents: list = []
        self._dram_handles: list = []

        # Create all agents upfront so each thread gets its own.
        agents = [make_agent(i) for i in range(n_read_prio + n_write_prio)]
        self._agents = agents

        for i in range(n_read_prio):
            t = threading.Thread(
                target=self._worker,
                args=(True, i),
                name=f"{thread_name_prefix}_r{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_prio):
            t = threading.Thread(
                target=self._worker,
                args=(False, n_read_prio + i),
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

    def register_dram_buffer(self, view_addr: int, view_size: int) -> None:
        """
        Register the DRAM buffer with all agents. Must be called after pool
        creation but before any tasks are enqueued.
        """
        self._dram_handles = []
        for agent in self._agents:
            dram_handle = agent.register_memory(
                [(view_addr, view_size, 0, "")],
                mem_type="DRAM",
                backends=["POSIX"]
            )
            self._dram_handles.append(dram_handle)

    def shutdown(self, wait: bool = True) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if wait:
            for t in self._threads:
                t.join()
        
        # Deregister DRAM handles on shutdown
        for agent, dram_handle in zip(self._agents, self._dram_handles):
            if dram_handle is not None:
                try:
                    agent.deregister_memory(dram_handle)
                except Exception:
                    pass

    def _worker(self, read_priority: bool, agent_idx: int) -> None:
        agent = self._agents[agent_idx]
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
            
            # Pass both agent and its pre-registered DRAM handle to the task
            dram_handle = self._dram_handles[agent_idx] if self._dram_handles else None
            task(agent, dram_handle)


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
# Per-block NIXL I/O callbacks — module-level so they are not re-created each loop
# ---------------------------------------------------------------------------

def _run_nixl_store(
    agent,
    dram_handle,
    block_size: int,
    view_addr: int,
    block_idx: int,
    dest_path: str,
    state: "_JobState",
) -> None:
    """
    Store callback: write one KV block via NIXL WRITE transfer.
    
    Writes to a temp file then atomically replaces the destination.
    Checks if block exists first to avoid redundant writes.
    Uses pre-registered DRAM handle to avoid repeated registration overhead.
    """
    tmp_fname = dest_path.replace('.bin', '.tmp')
    fd = -1
    xfer_handle = None
    file_handle = None
    try:
        # Check if block already exists to avoid redundant writes
        if os.path.exists(dest_path):
            state.task_done(True)
            return
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        fd = os.open(
            tmp_fname,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )

        block_addr = view_addr + block_idx * block_size
        local_descs = [(block_addr, block_size, 0)]
        remote_descs = [(0, block_size, fd, "")]

        # Use pre-registered DRAM handle instead of registering again
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
            xfer_state = agent.check_xfer_state(xfer_handle)
            if xfer_state == "DONE":
                break
            if xfer_state == "ERR":
                raise RuntimeError(
                    f"NIXL WRITE transfer failed for block {block_idx} -> {dest_path}"
                )

        os.close(fd)
        fd = -1
        os.replace(tmp_fname, dest_path)
        state.task_done(True)

    except Exception as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        state.task_done(False, exc)
    finally:
        if xfer_handle is not None:
            agent.release_xfer_handle(xfer_handle)
        if file_handle is not None:
            agent.deregister_memory(file_handle)


def _run_nixl_load(
    agent,
    dram_handle,
    block_size: int,
    view_addr: int,
    block_idx: int,
    source_path: str,
    state: "_JobState",
) -> None:
    """
    Load callback: read one KV block via NIXL READ transfer.
    Uses pre-registered DRAM handle to avoid repeated registration overhead.
    """
    fd = -1
    xfer_handle = None
    file_handle = None
    try:
        fd = os.open(source_path, os.O_RDONLY | os.O_DIRECT)

        block_addr = view_addr + block_idx * block_size
        local_descs = [(block_addr, block_size, 0)]
        remote_descs = [(0, block_size, fd, "")]

        # Use pre-registered DRAM handle instead of registering again
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
            xfer_state = agent.check_xfer_state(xfer_handle)
            if xfer_state == "DONE":
                break
            if xfer_state == "ERR":
                raise RuntimeError(
                    f"NIXL READ transfer failed for block {block_idx} <- {source_path}"
                )

        os.close(fd)
        fd = -1
        state.task_done(True)

    except Exception as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        state.task_done(False, exc)
    finally:
        if xfer_handle is not None:
            agent.release_xfer_handle(xfer_handle)
        if file_handle is not None:
            agent.deregister_memory(file_handle)


# ---------------------------------------------------------------------------
# FileSystemTierManagerNixl
# ---------------------------------------------------------------------------

class FileSystemTierManagerNixl(SecondaryTierManager):
    """
    NIXL-backed disk secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.  Each thread holds a dedicated NIXL agent.

    The thread pool and agents are initialized eagerly during construction.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished() polls job completion and returns completed JobResults.

    """

    def __init__(
        self,
        base_path: str,
        n_read_agents: int = 16,
        n_write_agents: int = 16,
    ):
        """
        Args:
            base_path: Root directory for block files.
            n_read_agents: Number of read-priority threads (each with its own
                NIXL agent).
            n_write_agents: Number of write-priority threads (each with its own
                NIXL agent).

        """
        self._base_path = base_path
        self._n_read_agents = n_read_agents
        self._n_write_agents = n_write_agents

        # Create pool eagerly with NIXL agents
        self._pool = _NixlDualQueuePool(
            n_read_agents,
            n_write_agents,
            self._make_nixl_agent,
            thread_name_prefix="vllm_kv_nixl_fs",
        )

        # job_id -> _JobState for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, _JobState] = {}

        # Long-lived memoryview of the primary CPU tensor (set once by TieringOffloadingManager).
        self._primary_view: memoryview | None = None

    def _make_nixl_agent(self, index: int):
        """Create one NIXL agent. Called once per thread at pool startup."""
        conf = nixl_agent_config(
            enable_prog_thread=True,
            backends=["POSIX"],
        )
        return nixl_agent(
            agent_name=f"vllm_kv_nixl_{index}",
            nixl_conf=conf,
            instantiate_all=False,
        )

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
        self._block_size = view.strides[0]
        self._primary_view = view.cast("B")

        # Register the DRAM buffer with all agents once
        self._view_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._primary_view))
        view_size = len(self._primary_view)
        self._pool.register_dram_buffer(self._view_addr, view_size)

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

        if all(self.lookup(key) for key in job_metadata.keys):
            return False

        state = _JobState(len(job_metadata.keys))

        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            dest_path = self.get_file_name(key)
            # Create closure that captures parameters and calls _run_nixl_store
            def _make_task(bs=self._block_size, va=self._view_addr, bi=bid,
                          dp=dest_path, st=state):
                return lambda agent, dram_handle: _run_nixl_store(agent, dram_handle, bs, va, bi, dp, st)
            
            self._pool.enqueue_write(_make_task())

        self._active_jobs[job_metadata.job_id] = state
        return True

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job: enqueue one read task per block to the read queue.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_load()"
        )

        state = _JobState(len(job_metadata.keys))

        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            source_path = self.get_file_name(key)
            # Create closure that captures parameters and calls _run_nixl_load
            def _make_task(bs=self._block_size, va=self._view_addr, bi=bid,
                          sp=source_path, st=state):
                return lambda agent, dram_handle: _run_nixl_load(agent, dram_handle, bs, va, bi, sp, st)
            
            self._pool.enqueue_read(_make_task())

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
                    "FileSystemTierManagerNixl: job %s NIXL I/O "
                    "failed: %s",
                    job_id,
                    err,
                )

            del self._active_jobs[job_id]

            results.append(JobResult(job_id=job_id, success=success))

        return results


    @staticmethod
    def get_tier_type() -> str:
        return "file_system_nixl"
