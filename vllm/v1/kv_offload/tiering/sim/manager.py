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
            return [key in self._tier._stored_keys for key in keys]


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
        read_delay_ms: float = 0.0,
        write_delay_ms: float = 0.0,
        base_overhead_ms: float = 0.0,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        self._lookup_delay_s = lookup_delay_ms / 1000.0
        self._read_delay_s = read_delay_ms / 1000.0
        self._write_delay_s = write_delay_ms / 1000.0
        self._base_overhead_s = base_overhead_ms / 1000.0

        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        self.locality = None
        self._stored_keys: set[OffloadKey] = set()
        self._stored_keys_lock = threading.Lock()

        self._zero_buf = bytes(self._block_size)
        self._use_o_direct = False

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
            "read=%.1fms, write=%.1fms, base_overhead=%.1fms, threads=%d+%d",
            lookup_delay_ms,
            read_delay_ms,
            write_delay_ms,
            base_overhead_ms,
            n_read_threads,
            n_write_threads,
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
        tasks = (
            functools.partial(
                _store_block,
                key,
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._write_delay_s,
                self._stored_keys,
                self._stored_keys_lock,
                self._use_o_direct,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                _load_block,
                key,
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._read_delay_s,
                self._stored_keys,
                self._stored_keys_lock,
                self._use_o_direct,
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

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
