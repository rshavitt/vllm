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

import os
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
from vllm.v1.kv_offload.tiering.fs.thread_pool_nixl import NixlDualQueuePool
from vllm.v1.kv_offload.tiering.fs.io_nixl import _run_nixl_store, _run_nixl_load
from vllm.v1.kv_offload.tiering.fs.state import JobState
try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    raise ImportError(
        "FileSystemTierManagerNixl requires the 'nixl' package. "
        "Install it with: pip install nixl"
    ) from e
    
logger = init_logger(__name__)


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
        self._pool = NixlDualQueuePool(
            n_read_agents,
            n_write_agents,
            self._make_nixl_agent,
            thread_name_prefix="vllm_kv_nixl_fs",
        )

        # job_id -> JobState for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, JobState] = {}

        # Long-lived memoryview of the primary CPU tensor (set once by TieringOffloadingManager).
        self._primary_view: memoryview | None = None
        self._block_size: int = 0

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
        assert view.strides is not None, "view.strides cannot be None"
        self._block_size = view.strides[0]
        self._primary_view = view.cast("B")

        # Register the DRAM buffer with all agents once
        self._view_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._primary_view))
        view_size = len(self._primary_view)
        self._pool.register_dram_buffer(self._view_addr, view_size)

    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None) -> bool | None:
        file_path = self.get_file_name(key)
        return os.path.exists(file_path)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit a store job: enqueue one write task per block to the write queue.

        Returns True if an async job was submitted, False if dropped.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_store()"
        )

        if all(self.lookup(key) for key in job_metadata.keys):
            return

        state = JobState(len(job_metadata.keys))

        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            dest_path = self.get_file_name(key)
            # Create closure that captures parameters and calls _run_nixl_store
            def _make_task(bs=self._block_size, va=self._view_addr, bi=bid,
                          dp=dest_path, st=state):
                return lambda agent, dram_handle: _run_nixl_store(agent, dram_handle, bs, va, bi, dp, st)
            
            self._pool.enqueue_write(_make_task())

        self._active_jobs[job_metadata.job_id] = state
        return

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job: enqueue one read task per block to the read queue.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_load()"
        )

        state = JobState(len(job_metadata.keys))

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
        Collect completed jobs by polling JobState.is_done.
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
        