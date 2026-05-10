# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManagerPython: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path_with_.tmp_instead_of_.bin>) via os.write, 
    then os.replace'd to the final path.

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>/<hhh>/<hh>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import os
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.fs.io import _store_block, _load_block
from vllm.v1.kv_offload.tiering.fs.state import JobState

logger = init_logger(__name__)


class FileSystemTierManager(SecondaryTierManager):
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

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        # job_id -> _JobState for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, JobState] = {}

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
        assert view.strides is not None, "view.strides cannot be None"
        self._block_size = view.strides[0]
        self._primary_view = view.cast("B")

    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None) -> bool | None:
        file_path = self.get_file_name(key)
        return os.path.exists(file_path)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit a store job: enqueue one write task per block to the write queue.

        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_store()"
        )

        state = JobState(len(job_metadata.keys))

        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            dest_path = self.get_file_name(key)
            offset = bid * self._block_size
            self._pool.enqueue_write(
                functools.partial(_store_block, dest_path, self._primary_view, offset, self._block_size, state)
            )

        self._active_jobs[job_metadata.job_id] = state

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job: enqueue one read task per block to the read queue.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_load()"
        )
        block_ids = [int(bid) for bid in job_metadata.block_ids]
        state = JobState(len(job_metadata.keys))

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
