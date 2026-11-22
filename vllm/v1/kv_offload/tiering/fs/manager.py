# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            hits = list(batch_lookup_C(paths))
        else:
            hits = [os.path.exists(p) for p in paths]

        stored_count = sum(hits)
        request_blocks = len(keys)

        self._tier._total_lookup_blocks += request_blocks
        self._tier._storage_hit_blocks += stored_count
        self._tier._lookup_call_count += 1

        if self._tier._lookup_call_count % self._tier._stats_log_interval == 0:
            s_total = self._tier._total_lookup_blocks
            s_hits = self._tier._storage_hit_blocks
            s_prefill = self._tier._forced_prefill_blocks
            s_hit_pct = 100.0 * s_hits / s_total if s_total else 0.0
            s_pfx_pct = 100.0 * s_prefill / s_total if s_total else 0.0
            logger.info(
                "FileSystemTierManager stats: "
                "total_lookups=%d, storage_hits=%d (%.1f%%), "
                "forced_prefill=%d (%.1f%%)",
                s_total,
                s_hits,
                s_hit_pct,
                s_prefill,
                s_pfx_pct,
            )

        if self._tier._enable_queue_balancing and stored_count > 0:
            load_queue_len = len(self._tier._pool._load_q)
            with self._tier._queue_balance_lock:
                # Only count prefill blocks from requests that have
                # no pending loads — blocked requests can't prefill.
                prefill_len = sum(
                    blks
                    for rid, blks in self._tier._req_prefill_blocks.items()
                    if rid not in self._tier._reqs_with_pending_loads
                )
                prefill_target = load_queue_len * self._tier._prefill_ratio
                if prefill_len + request_blocks < prefill_target:
                    self._tier._prefill_queue_blocks += request_blocks
                    self._tier._req_prefill_blocks[req_context.req_id] = (
                        self._tier._req_prefill_blocks.get(req_context.req_id, 0)
                        + request_blocks
                    )
                    self._tier._forced_prefill_blocks += stored_count
                    logger.info(
                        "Queue balancing: forcing prefill for req %s "
                        "(%d blocks). prefill_q=%d, load_q=%d",
                        req_context.req_id,
                        request_blocks,
                        prefill_len + request_blocks,
                        load_queue_len,
                    )
                    return [False] * request_blocks

        return hits


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[str] = MEDIUM_FS

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        enable_queue_balancing: bool = False,
        prefill_ratio: float = 1.0,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            enable_queue_balancing: When True, batch_lookup may force MISS
                for all blocks of a request to keep the prefill queue stocked
                relative to the load queue.
            prefill_ratio: Target ratio of prefill-queue blocks to load-queue
                blocks. Used when enable_queue_balancing is True.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._enable_queue_balancing = enable_queue_balancing
        self._prefill_ratio: float = prefill_ratio
        self._prefill_queue_blocks: int = 0
        self._req_prefill_blocks: dict[str, int] = {}
        self._reqs_with_pending_loads: set[str] = set()
        self._load_job_req: dict[int, str] = {}
        self._req_pending_load_jobs: dict[str, int] = {}
        self._queue_balance_lock = threading.Lock()
        self._total_lookup_blocks: int = 0
        self._storage_hit_blocks: int = 0
        self._forced_prefill_blocks: int = 0
        self._lookup_call_count: int = 0
        self._stats_log_interval: int = 100

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        req_id = job_metadata.req_context.req_id
        self._load_job_req[job_metadata.job_id] = req_id
        self._req_pending_load_jobs[req_id] = (
            self._req_pending_load_jobs.get(req_id, 0) + 1
        )
        self._reqs_with_pending_loads.add(req_id)
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            req_id = self._load_job_req.pop(job_id, None)
            if req_id is not None:
                remaining = self._req_pending_load_jobs.get(req_id, 1) - 1
                if remaining <= 0:
                    self._req_pending_load_jobs.pop(req_id, None)
                    self._reqs_with_pending_loads.discard(req_id)
                else:
                    self._req_pending_load_jobs[req_id] = remaining
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        self._reqs_with_pending_loads.discard(req_context.req_id)
        self._req_pending_load_jobs.pop(req_context.req_id, None)
        with self._queue_balance_lock:
            prefill_blocks = self._req_prefill_blocks.pop(req_context.req_id, 0)
            if prefill_blocks:
                self._prefill_queue_blocks -= prefill_blocks

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        total = self._total_lookup_blocks
        hits = self._storage_hit_blocks
        prefill = self._forced_prefill_blocks
        if total > 0:
            hit_pct = 100.0 * hits / total
            prefill_pct = 100.0 * prefill / total
        else:
            hit_pct = prefill_pct = 0.0
        logger.info(
            "FileSystemTierManager stats: "
            "total_lookups=%d, storage_hits=%d (%.1f%%), "
            "forced_prefill=%d (%.1f%%)",
            total,
            hits,
            hit_pct,
            prefill,
            prefill_pct,
        )
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
