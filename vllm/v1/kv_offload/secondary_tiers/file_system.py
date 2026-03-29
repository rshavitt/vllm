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

I/O is performed by the _kv_storage_ops C++ extension (pread/pwrite,
internal dual-queue thread pool). One C++ task is enqueued per block file,
tracked by a shared JobState. get_finished_jobs() polls which jobs are done.

Memory safety: _ActiveJob.buffers holds memoryviews of the CPU tensors,
keeping them alive until the C++ tasks finish. Views are released once
get_finished_jobs() reports the job as done.
"""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    JobId,
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.mediums import CPUMemoryViewLoadStoreSpec

logger = init_logger(__name__)

try:
    from vllm._kv_storage_ops import get_finished_jobs as cpp_get_finished_jobs
    from vllm._kv_storage_ops import set_thread_count as cpp_set_thread_count
    from vllm._kv_storage_ops import submit_load_job as cpp_submit_load_job
    from vllm._kv_storage_ops import submit_store_job as cpp_submit_store_job
except ImportError as e:
    raise ImportError(
        "FileSystemTierManager requires the _kv_storage_ops C++ extension. "
        "Rebuild vLLM with the kv_storage_ops target enabled."
    ) from e


@dataclass
class _ActiveJob:
    """A job whose tasks have been enqueued in the C++ pool."""
    is_store: bool
    hashes:   list[BlockHash]
    buffer:   memoryview  # kept alive until get_finished_jobs() reports done


class FileSystemTierManager(SecondaryTierManager):
    """
    Disk-backed secondary tier that stores each KV block as a single file.

    Both store and load jobs are submitted to the C++ thread pool immediately.
    Read-vs-write ordering is enforced by the dual-queue pool: read-priority
    threads drain the read queue first, so loads are processed before stores
    even when both are queued concurrently.

    TieredOffloadingManager calls primary.finalize_blocks() after
    get_finished() reports a load as successful.

    Eviction:
        LRU via OrderedDict. Blocks in-flight are never evicted.
    """

    def __init__(
        self,
        base_path: str,
        max_blocks: int,
        tier_name: str = "Storage",
        n_read_threads: int | None = None,
        n_write_threads: int | None = None,
    ):
        """
        Args:
            base_path: Root directory for block files.
            max_blocks: Maximum number of blocks to keep indexed (LRU eviction).
            tier_name: Identifier string returned by get_tier_name().
            n_read_threads: Number of read-priority I/O threads. If provided
                together with n_write_threads, overrides the default (32/16).
            n_write_threads: Number of write-priority I/O threads. Must be
                provided together with n_read_threads.
        """
        self._base_path = base_path
        self._max_blocks = max_blocks
        self._tier_name = tier_name

        if n_read_threads is not None and n_write_threads is not None:
            cpp_set_thread_count(n_read_threads, n_write_threads)

        # LRU ordered set: block_hash -> True
        self._blocks: OrderedDict[BlockHash, bool] = OrderedDict()

        # block_hash -> job_id (prevents eviction mid-transfer)
        self._in_flight: dict[BlockHash, JobId] = {}

        # job_id -> _ActiveJob for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, _ActiveJob] = {}

        # Number of blocks in _blocks that are NOT currently in _in_flight.
        self._evictable_count: int = 0

    # ------------------------------------------------------------------
    # File-naming
    # ------------------------------------------------------------------

    def get_file_name(self, block_hash: int | bytes) -> str:
        """
        Return the file path for a KV block.
        <base>/<hhh>/<hh>/<hash>.bin — hash-based subdirectories to limit
        directory fan-out.
        """
        if isinstance(block_hash, bytes):
            block_hash = int.from_bytes(block_hash, "big")
        assert isinstance(block_hash, int)
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self._base_path}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        count = 0
        for bh in block_hashes:
            if bh in self._in_flight:
                return None  # transfer in progress — caller should retry
            if bh not in self._blocks:
                break
            count += 1
        return count

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit a store job to the C++ thread pool immediately.
        Read-vs-write priority is handled by the dual-queue pool.
        """
        assert isinstance(job_metadata.spec, CPUMemoryViewLoadStoreSpec), (
            f"Expected CPUMemoryViewLoadStoreSpec, got {type(job_metadata.spec)}"
        )
        spec: CPUMemoryViewLoadStoreSpec = job_metadata.spec
        job_id = job_metadata.job_id
        all_hashes = job_metadata.block_hashes

        # Keep only blocks not yet on disk and not currently being stored.
        pairs = [
            (bh, int(bid))
            for bh, bid in zip(all_hashes, spec.block_ids)
            if bh not in self._blocks and bh not in self._in_flight
        ]
        if not pairs:
            return

        hashes_to_store, block_ids_to_store = map(list, zip(*pairs))

        # LRU eviction: free space for the incoming blocks.
        num_to_evict = len(hashes_to_store) - (
            self._max_blocks - len(self._blocks)
        )
        if num_to_evict > 0:
            if self._evictable_count < num_to_evict:
                logger.warning(
                    "FileSystemTierManager(%s): insufficient evictable blocks "
                    "(%d evictable, %d needed); dropping store job %s.",
                    self._tier_name,
                    self._evictable_count,
                    num_to_evict,
                    job_id,
                )
                return

            protected = set(all_hashes)
            evicted = []
            for bh in self._blocks:  # oldest-first (OrderedDict)
                if bh not in protected and bh not in self._in_flight:
                    evicted.append(bh)
                    if len(evicted) == num_to_evict:
                        break
            else:
                logger.warning(
                    "FileSystemTierManager(%s): could not find %d evictable "
                    "blocks (protected overlap reduced candidates); "
                    "dropping store job %s.",
                    self._tier_name,
                    num_to_evict,
                    job_id,
                )
                return
            for bh in evicted:
                del self._blocks[bh]
                self._evictable_count -= 1

        buffer = spec.tensor_view
        dest_files = [self.get_file_name(bh) for bh in hashes_to_store]

        for bh in hashes_to_store:
            self._in_flight[bh] = job_id

        try:
            cpp_submit_store_job(job_id, buffer, spec.block_stride_bytes,
                                 block_ids_to_store, dest_files)
        except Exception:
            for bh in hashes_to_store:
                del self._in_flight[bh]
            logger.exception(
                "FileSystemTierManager(%s): failed to submit store job %s.",
                self._tier_name,
                job_id,
            )
            raise

        self._active_jobs[job_id] = _ActiveJob(
            is_store=True,
            hashes=hashes_to_store,
            buffer=buffer,
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job to the C++ thread pool immediately.
        """
        assert isinstance(job_metadata.spec, CPUMemoryViewLoadStoreSpec), (
            f"Expected CPUMemoryViewLoadStoreSpec, got {type(job_metadata.spec)}"
        )
        spec: CPUMemoryViewLoadStoreSpec = job_metadata.spec
        job_id = job_metadata.job_id
        block_hashes = list(job_metadata.block_hashes)
        block_ids = [int(bid) for bid in spec.block_ids]

        for bh in block_hashes:
            if bh not in self._blocks:
                logger.warning(
                    "FileSystemTierManager(%s): block %s not found on disk; "
                    "dropping load job %s.",
                    self._tier_name,
                    bh,
                    job_id,
                )
                return

        buffer = spec.tensor_view
        block_size = spec.block_stride_bytes
        source_files = [self.get_file_name(bh) for bh in block_hashes]

        for bh in block_hashes:
            self._in_flight[bh] = job_id
            self._evictable_count -= 1  # block is in-flight, not evictable

        try:
            cpp_submit_load_job(job_id, buffer, block_size, block_ids,
                                source_files)
        except Exception:
            # Roll back all state mutations so the tier stays consistent.
            for bh in block_hashes:
                del self._in_flight[bh]
                self._evictable_count += 1
            logger.exception(
                "FileSystemTierManager(%s): failed to submit load job %s.",
                self._tier_name,
                job_id,
            )
            raise

        self._active_jobs[job_id] = _ActiveJob(
            is_store=False,
            hashes=block_hashes,
            buffer=buffer,
        )

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs reported by the C++ pool.
        """
        results: list[JobResult] = []

        for job_id, success in cpp_get_finished_jobs():
            job = self._active_jobs.pop(job_id)

            # Release memoryview now that the C++ tasks are done.
            job.buffer.release()

            for bh in job.hashes:
                del self._in_flight[bh]
                if job.is_store and success:
                    self._blocks[bh] = True
                    self._evictable_count += 1  # on disk, not in-flight
                elif not job.is_store:
                    # Load complete (success or failure): block stays in
                    # _blocks and is no longer in-flight → evictable again.
                    self._evictable_count += 1

            results.append(JobResult(job_id=job_id, success=success))

        return results

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        for bh in reversed(list(block_hashes)):
            if bh in self._blocks:
                self._blocks.move_to_end(bh)

    def get_tier_name(self) -> str:
        return self._tier_name

    # ------------------------------------------------------------------
    # Diagnostics (not part of the abstract interface)
    # ------------------------------------------------------------------

    def get_num_blocks(self) -> int:
        return len(self._blocks)

    def get_num_in_flight(self) -> int:
        return len(self._in_flight)
