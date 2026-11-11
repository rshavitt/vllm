# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SimulatedTierManager: Simulated storage secondary tier for benchmarking.

Tracks block existence in an in-memory dictionary (no content stored).
All read operations return zero tensors. Configurable delays simulate
real storage latency for lookups, reads, and writes.
"""

import functools
import threading
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.sim.io import (
    BandwidthLimiter,
    _load_block,
    _store_block,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class SimAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for SimulatedTierManager."""

    def __init__(self, tier: "SimulatedTierManager", tier_type: str) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        if self._tier._lookup_delay_s > 0:
            time.sleep(self._tier._lookup_delay_s * len(keys))
        with self._tier._stored_keys_lock:
            hits = [key in self._tier._stored_keys for key in keys]
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
                    "SimulatedTierManager stats: "
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


class SimulatedTierManager(SecondaryTierManager):
    """
    Simulated storage tier for benchmarking offloading overhead.

    Stores only block keys (not content) in a dictionary. Reads return
    zeros. Configurable delays on lookup, read, and write operations
    simulate real storage latency.
    """

    medium: ClassVar[str] = "SIM"

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        lookup_delay_ms: float = 0.0,
        read_bandwidth_mb_s: float = 1000.0,
        write_bandwidth_mb_s: float = 1000.0,
        read_overhead_ms: float = 0.0,
        write_overhead_ms: float = 0.0,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_queue_balancing: bool = True,
        prefill_ratio: float = 1.0,
    ):
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        self._lookup_delay_s = lookup_delay_ms / 1000.0
        self._read_overhead_s = read_overhead_ms / 1000.0
        self._write_overhead_s = write_overhead_ms / 1000.0

        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        self._read_limiter = BandwidthLimiter(read_bandwidth_mb_s * 1024 * 1024)
        self._write_limiter = BandwidthLimiter(write_bandwidth_mb_s * 1024 * 1024)

        self.locality = None
        self._stored_keys: set[OffloadKey] = set()
        self._stored_keys_lock = threading.Lock()

        self._enable_queue_balancing = enable_queue_balancing
        self._prefill_ratio = prefill_ratio
        self._prefill_queue_blocks: int = 0
        self._req_prefill_blocks: dict[str, int] = {}
        self._reqs_with_pending_loads: frozenset[str] = frozenset()

        self._total_lookup_blocks: int = 0
        self._storage_hit_blocks: int = 0
        self._forced_prefill_blocks: int = 0
        self._lookup_call_count: int = 0
        self._stats_log_interval: int = 100

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_sim",
        )
        self._lookup_manager = SimAsyncLookupManager(
            tier=self, tier_type=self.tier_type
        )

        self.events: list[OffloadingEvent] | None = None
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        logger.info(
            "SimulatedTierManager initialized: lookup=%.1fms, "
            "read_bw=%.1f MB/s, write_bw=%.1f MB/s, "
            "read_overhead=%.1fms, write_overhead=%.1fms, "
            "threads=%d+%d, queue_balancing=%s, prefill_ratio=%.2f",
            lookup_delay_ms,
            read_bandwidth_mb_s,
            write_bandwidth_mb_s,
            read_overhead_ms,
            write_overhead_ms,
            n_read_threads,
            n_write_threads,
            enable_queue_balancing,
            prefill_ratio,
        )

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
        if self._write_overhead_s > 0:
            self._write_limiter.reserve_overhead(self._write_overhead_s)
        tasks = (
            functools.partial(
                _store_block,
                key,
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._write_limiter,
                self._stored_keys,
                self._stored_keys_lock,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        if self._read_overhead_s > 0:
            self._read_limiter.reserve_overhead(self._read_overhead_s)
        tasks = (
            functools.partial(
                _load_block,
                key,
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._read_limiter,
                self._stored_keys,
                self._stored_keys_lock,
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
                if success and keys and self.medium is not None:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        with self._stored_keys_lock:
            prefill_blocks = self._req_prefill_blocks.pop(req_context.req_id, 0)
            if prefill_blocks:
                self._prefill_queue_blocks -= prefill_blocks

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._reqs_with_pending_loads = context.reqs_with_pending_promotions
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        total = self._total_lookup_blocks
        hits = self._storage_hit_blocks
        prefill = self._forced_prefill_blocks
        if total > 0:
            hit_pct = 100.0 * hits / total
            prefill_pct = 100.0 * prefill / total
        else:
            hit_pct = prefill_pct = 0.0
        logger.info(
            "SimulatedTierManager stats: "
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
