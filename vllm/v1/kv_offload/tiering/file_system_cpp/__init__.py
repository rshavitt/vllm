# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Disk-backed secondary tier for KV cache offloading.

Each KV block is stored as a single file of exactly block_stride_bytes bytes.
Filenames are derived from the block hash using hash-based subdirectories
to limit directory fan-out:
  <base_path>/<hhh>/<hh>/<hash_hex>.bin

Store and load jobs are both submitted to the C++ thread pool immediately.
Read-vs-write priority is handled by the C++ dual-queue thread pool: threads
that prioritise reads drain the read queue first, so loads are processed
ahead of stores even when both are queued concurrently.

I/O is performed by the _kv_file_system_ops C++ extension (pread/pwrite,
internal dual-queue thread pool). One C++ task is enqueued per block file,
tracked by a shared JobState. get_finished_jobs() polls which jobs are done.

"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Sequence

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)

logger = init_logger(__name__)

# Module-level stash for finished C++ jobs that belong to a different tier
# instance than the one that happened to call cpp_get_finished_jobs() first.
# Since all get_finished() calls come from a single Python scheduler thread,
# no locking is needed.
_g_finished_stash: dict[int, bool] = {}  # cpp_job_id -> success
_g_iid_counter: int = 0

try:
    from vllm._kv_file_system_ops import get_finished_jobs as cpp_get_finished_jobs
    from vllm._kv_file_system_ops import set_thread_count as cpp_set_thread_count
    from vllm._kv_file_system_ops import submit_load_job as cpp_submit_load_job
    from vllm._kv_file_system_ops import submit_store_job as cpp_submit_store_job
except ImportError as e:
    raise ImportError(
        "FileSystemTierManager requires the _kv_file_system_ops C++ extension. "
        "Rebuild vLLM with the kv_file_system_ops target enabled."
    ) from e


class FileSystemTierManagerCpp(SecondaryTierManager):
    """
    Disk-backed secondary tier that stores each KV block as a single file.

    Both store and load jobs are submitted to the C++ thread pool immediately.
    Read-vs-write ordering is enforced by the dual-queue pool: read-priority
    threads drain the read queue first, so loads are processed before stores
    even when both are queued concurrently.

    TieredOffloadingManager calls primary.finalize_blocks() after
    get_finished() reports a load as successful.

    """

    def __init__(
        self,
        base_path: str,
        n_read_threads: int | None = 16,
        n_write_threads: int | None = 16,
    ):
        """
        Args:
            base_path: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads. If provided
                together with n_write_threads, overrides the default (32/16).
            n_write_threads: Number of write-priority I/O threads. Must be
                provided together with n_read_threads.
        """
        self._base_path = base_path

        cpp_set_thread_count(n_read_threads, n_write_threads)

        # job_id -> _ActiveJob for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, bool] = {}

        # Long-lived memoryview of the primary CPU tensor (set once by TieringOffloadingManager).
        self._primary_view: memoryview | None = None

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
        self._primary_view = view

    def lookup(self, key: OffloadKey, req_context: ReqContext | None = None) -> bool | None:
        file_path = self.get_file_name(key)
        return os.path.exists(file_path)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit a store job to the C++ thread pool immediately.
        Read-vs-write priority is handled by the dual-queue pool.

        Returns True if an async job was submitted, False if dropped.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_store()"
        )
        dest_files = [self.get_file_name(key) for key in job_metadata.keys]

        cpp_submit_store_job(job_metadata.job_id, self._primary_view, self._block_size,
                       job_metadata.block_ids, dest_files)

        self._active_jobs[job_metadata.job_id] = True

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job to the C++ thread pool immediately.
        """
        assert self._primary_view is not None, (
            "set_primary_view() must be called before submit_load()"
        )

        source_files = [self.get_file_name(key) for key in job_metadata.keys]
        cpp_submit_load_job(job_metadata.job_id, self._primary_view, self._block_size, job_metadata.block_ids, source_files)        

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs reported by the C++ extention.

        """
        return [JobResult(job_id=jid, success=ok) for jid, ok in cpp_get_finished_jobs()]

    @staticmethod
    def get_tier_type() -> str:
        return "file_system_cpp"
    