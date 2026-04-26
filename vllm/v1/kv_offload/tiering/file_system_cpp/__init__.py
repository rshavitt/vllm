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

Memory safety: _ActiveJob.buffers holds memoryviews of the CPU tensors,
keeping them alive until the C++ tasks finish. Views are released once
get_finished_jobs() reports the job as done.
"""

import os
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import (
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
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

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
    from vllm._kv_file_system_ops import submit_load_job_bulk as cpp_submit_load_job_bulk
    from vllm._kv_file_system_ops import submit_store_job as cpp_submit_store_job
    from vllm._kv_file_system_ops import submit_store_job_bulk as cpp_submit_store_job_bulk
except ImportError as e:
    raise ImportError(
        "FileSystemTierManager requires the _kv_file_system_ops C++ extension. "
        "Rebuild vLLM with the kv_file_system_ops target enabled."
    ) from e


@dataclass
class _ActiveJob:
    """A job whose tasks have been enqueued in the C++ pool."""
    is_store: bool
    keys:     list[OffloadKey]
    buffer:   memoryview  # kept alive until get_finished_jobs() reports done


class FileSystemTierManagerCpp(SecondaryTierManager):
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
        tier_name: str = "StorageCpp",
        n_read_threads: int | None = None,
        n_write_threads: int | None = None,
        bulk_mode: bool = False,
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
            bulk_mode: If True, submit all blocks of a job as a single C++ task
                instead of one task per block. Reduces queue overhead at the
                cost of within-job parallelism.
        """
        self._base_path = base_path
        self._max_blocks = max_blocks
        self._tier_name = tier_name
        self._bulk_mode = bulk_mode

        if n_read_threads is not None and n_write_threads is not None:
            cpp_set_thread_count(n_read_threads, n_write_threads)

        # Unique per-instance ID used to namespace C++ job IDs so that two
        # FileSystemTierManagerCpp instances never collide in the global pool.
        global _g_iid_counter
        _g_iid_counter += 1
        self._iid: int = _g_iid_counter

        # Maps C++ job id → Python job id for results dispatching.
        self._cpp_to_py: dict[int, JobId] = {}

        # LRU ordered set: key -> True
        self._blocks: OrderedDict[OffloadKey, bool] = OrderedDict()

        # key -> job_id (prevents eviction mid-transfer)
        self._in_flight: dict[OffloadKey, JobId] = {}

        # job_id -> _ActiveJob for all submitted (in-flight) jobs
        self._active_jobs: dict[JobId, _ActiveJob] = {}

        # Number of blocks in _blocks that are NOT currently in _in_flight.
        self._evictable_count: int = 0

        # Long-lived memoryview of the primary CPU tensor (set once by TieringOffloadingManager).
        self._primary_view: memoryview | None = None

    # ------------------------------------------------------------------
    # C++ job-id namespacing
    # ------------------------------------------------------------------

    def _make_cpp_job_id(self, py_job_id: JobId) -> int:
        """Encode (instance_id, py_job_id) into a single int for the C++ pool."""
        return (self._iid << 32) | (int(py_job_id) & 0xFFFFFFFF)

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
        if not isinstance(block_hash, int):
            raise TypeError(
                f"block_hash must be int or bytes, got {type(block_hash).__name__}"
            )
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self._base_path}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

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
        Submit a store job to the C++ thread pool immediately.
        Read-vs-write priority is handled by the dual-queue pool.
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
                    "FileSystemTierManager(%s): insufficient evictable blocks "
                    "(%d evictable, %d needed); dropping store job %s.",
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
                    "FileSystemTierManager(%s): could not find %d evictable "
                    "blocks (protected overlap reduced candidates); "
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

        buffer = self._primary_view
        dest_files = [self.get_file_name(get_offload_block_hash(key)) for key in keys_to_store]

        for key in keys_to_store:
            self._in_flight[key] = job_id

        cpp_jid = self._make_cpp_job_id(job_id)
        self._cpp_to_py[cpp_jid] = job_id
        try:
            _submit_fn = (cpp_submit_store_job_bulk if self._bulk_mode
                          else cpp_submit_store_job)
            _submit_fn(cpp_jid, buffer, self._block_size,
                       block_ids_to_store, dest_files)
        except Exception:
            del self._cpp_to_py[cpp_jid]
            for key in keys_to_store:
                del self._in_flight[key]
            logger.exception(
                "FileSystemTierManager(%s): failed to submit store job %s.",
                self._tier_name,
                job_id,
            )
            raise

        self._active_jobs[job_id] = _ActiveJob(
            is_store=True,
            keys=keys_to_store,
            buffer=buffer,
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit a load job to the C++ thread pool immediately.
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
                    "FileSystemTierManager(%s): block %s not found on disk; "
                    "dropping load job %s.",
                    self._tier_name,
                    key,
                    job_id,
                )
                return

        buffer = self._primary_view
        source_files = [self.get_file_name(get_offload_block_hash(key)) for key in keys]

        for key in keys:
            self._in_flight[key] = job_id
            self._evictable_count -= 1  # block is in-flight, not evictable

        cpp_jid = self._make_cpp_job_id(job_id)
        self._cpp_to_py[cpp_jid] = job_id
        try:
            _submit_fn = (cpp_submit_load_job_bulk if self._bulk_mode
                          else cpp_submit_load_job)
            _submit_fn(cpp_jid, buffer, self._block_size, block_ids, source_files)
        except Exception:
            del self._cpp_to_py[cpp_jid]
            # Roll back all state mutations so the tier stays consistent.
            for key in keys:
                del self._in_flight[key]
                self._evictable_count += 1
            logger.exception(
                "FileSystemTierManager(%s): failed to submit load job %s.",
                self._tier_name,
                job_id,
            )
            raise

        self._active_jobs[job_id] = _ActiveJob(
            is_store=False,
            keys=keys,
            buffer=buffer,
        )

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs reported by the C++ pool.

        cpp_get_finished_jobs() drains the global C++ registry.  When multiple
        FileSystemTierManagerCpp instances are live, the first caller may
        receive results that belong to a different instance.  Those are stashed
        in _g_finished_stash (keyed by the namespaced cpp job id) and picked up
        the next time the owning instance calls get_finished().
        """
        results: list[JobResult] = []

        # Drain new results from C++ into the global stash.
        for cpp_jid, success in cpp_get_finished_jobs():
            _g_finished_stash[cpp_jid] = success

        # Process results that belong to this instance.
        for cpp_jid in list(self._cpp_to_py.keys()):
            if cpp_jid not in _g_finished_stash:
                continue
            success = _g_finished_stash.pop(cpp_jid)
            job_id = self._cpp_to_py.pop(cpp_jid)
            job = self._active_jobs.pop(job_id)

            for key in job.keys:
                del self._in_flight[key]
                if job.is_store and success:
                    self._blocks[key] = True
                    self._evictable_count += 1  # on disk, not in-flight
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
    # Diagnostics (not part of the abstract interface)
    # ------------------------------------------------------------------

    def get_num_blocks(self) -> int:
        return len(self._blocks)

    def get_num_in_flight(self) -> int:
        return len(self._in_flight)
