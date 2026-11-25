# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for SimulatedTierManager.

Tests verify:
1. Lookup returns RETRY on first call, HIT/MISS after flush+drain
2. submit_store registers keys; get_finished_jobs reports success
3. submit_load writes zeros into the primary_kv_view slot
4. Multiple blocks handled correctly
5. Configured delays are applied
6. Shutdown completes cleanly
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.fs.thread_pool import JobState
from vllm.v1.kv_offload.tiering.queue_balance import CapacityRatioController
from vllm.v1.kv_offload.tiering.sim.manager import SimulatedTierManager

_CTX = ReqContext(req_id="test-sim")
_MOCK_SPEC = MagicMock()
_MOCK_SPEC.tokens_per_block = (16,)
_NUM_BLOCKS = 8
_BLOCK_BYTES = 64


def _make_primary_view(
    num_blocks: int = _NUM_BLOCKS, block_bytes: int = _BLOCK_BYTES
) -> memoryview:
    arr = np.zeros((num_blocks, block_bytes), dtype=np.int8)
    return memoryview(arr)


def _make_tier(
    primary_view: memoryview | None = None,
    lookup_delay_ms: float = 0.0,
    n_read_threads: int = 2,
    n_write_threads: int = 2,
) -> SimulatedTierManager:
    if primary_view is None:
        primary_view = _make_primary_view()
    return SimulatedTierManager(
        offloading_spec=_MOCK_SPEC,
        primary_kv_view=primary_view,
        tier_type="sim",
        lookup_delay_ms=lookup_delay_ms,
        n_read_threads=n_read_threads,
        n_write_threads=n_write_threads,
    )


def _to_key(i: int) -> bytes:
    return make_offload_key(str(i).encode(), 0)


def _make_job(job_id: int, block_ids: list[int]) -> JobMetadata:
    return JobMetadata(
        job_id=job_id,
        keys=[_to_key(bid) for bid in block_ids],
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=False,
        req_context=_CTX,
    )


def _flush(tier: SimulatedTierManager) -> None:
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))


def _resolve_lookup(
    tier: SimulatedTierManager, key, max_wait_s: float = 2.0
) -> LookupResult:
    """Poll until lookup resolves past RETRY, or timeout.

    flush() is called on every iteration because _need_to_drain is reset
    to False after drain_results() runs, even when results aren't available
    yet — subsequent lookup() calls would skip draining without it.
    """
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        _flush(tier)
        result = tier.lookup(key, _CTX)
        if result is not LookupResult.RETRY:
            return result
        time.sleep(0.01)
    raise TimeoutError(f"lookup did not resolve within {max_wait_s}s")


@pytest.fixture
def tier():
    t = _make_tier()
    yield t
    t.shutdown()


class TestLookup:
    def test_first_lookup_returns_retry(self, tier):
        assert tier.lookup(_to_key(1), _CTX) is LookupResult.RETRY

    def test_unknown_key_resolves_to_miss(self, tier):
        key = _to_key(99)
        tier.lookup(key, _CTX)
        assert _resolve_lookup(tier, key) is LookupResult.MISS

    def test_stored_key_resolves_to_hit(self, tier):
        key = _to_key(1)
        tier.submit_store(_make_job(0, [1]))
        tier.drain_jobs()

        tier.lookup(key, _CTX)
        assert _resolve_lookup(tier, key) is LookupResult.HIT

    def test_unstored_key_is_miss_after_store_of_other(self, tier):
        tier.submit_store(_make_job(0, [1]))
        tier.drain_jobs()

        key_missing = _to_key(2)
        tier.lookup(key_missing, _CTX)
        assert _resolve_lookup(tier, key_missing) is LookupResult.MISS

    def test_multiple_keys_resolve_correctly(self, tier):
        tier.submit_store(_make_job(0, [0, 1, 2]))
        tier.drain_jobs()

        for i in range(5):
            tier.lookup(_to_key(i), _CTX)
        _flush(tier)
        time.sleep(0.05)

        for i in range(3):
            assert tier.lookup(_to_key(i), _CTX) is LookupResult.HIT
        for i in range(3, 5):
            assert tier.lookup(_to_key(i), _CTX) is LookupResult.MISS


class TestStore:
    def test_store_completes_successfully(self, tier):
        tier.submit_store(_make_job(42, [0, 1, 2]))
        tier.drain_jobs()
        results = list(tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 42
        assert results[0].success is True

    def test_multiple_store_jobs(self, tier):
        for job_id in range(4):
            tier.submit_store(_make_job(job_id, [job_id]))
        tier.drain_jobs()
        results = {r.job_id: r for r in tier.get_finished_jobs()}
        assert len(results) == 4
        assert all(r.success for r in results.values())

    def test_store_registers_all_keys(self, tier):
        block_ids = [0, 1, 2, 3]
        tier.submit_store(_make_job(0, block_ids))
        tier.drain_jobs()

        for bid in block_ids:
            tier.lookup(_to_key(bid), _CTX)
        _flush(tier)
        time.sleep(0.05)

        for bid in block_ids:
            assert tier.lookup(_to_key(bid), _CTX) is LookupResult.HIT


class TestLoad:
    def test_load_writes_zeros_to_primary(self):
        arr = np.zeros((_NUM_BLOCKS, _BLOCK_BYTES), dtype=np.int8)
        view = memoryview(arr)
        t = _make_tier(primary_view=view)
        try:
            arr[3, :] = 0x7F

            t.submit_store(_make_job(0, [3]))
            t.drain_jobs()
            list(t.get_finished_jobs())

            t.submit_load(_make_job(1, [3]))
            t.drain_jobs()
            list(t.get_finished_jobs())

            assert np.all(arr[3, :] == 0)
        finally:
            t.shutdown()

    def test_load_completes_with_success(self, tier):
        tier.submit_store(_make_job(0, [0]))
        tier.drain_jobs()
        list(tier.get_finished_jobs())

        tier.submit_load(_make_job(1, [0]))
        tier.drain_jobs()
        results = list(tier.get_finished_jobs())
        assert len(results) == 1
        assert results[0].job_id == 1
        assert results[0].success is True

    def test_multiple_load_jobs(self, tier):
        for i in range(4):
            tier.submit_store(_make_job(i, [i]))
        tier.drain_jobs()
        list(tier.get_finished_jobs())

        for i in range(4):
            tier.submit_load(_make_job(10 + i, [i]))
        tier.drain_jobs()
        results = {r.job_id: r for r in tier.get_finished_jobs()}
        assert len(results) == 4
        assert all(r.success for r in results.values())


class TestDelays:
    def test_write_overhead_is_applied(self):
        overhead_ms = 50
        t = SimulatedTierManager(
            offloading_spec=_MOCK_SPEC,
            primary_kv_view=_make_primary_view(),
            tier_type="sim",
            write_overhead_ms=overhead_ms,
            n_read_threads=2,
            n_write_threads=2,
        )
        try:
            start = time.monotonic()
            t.submit_store(_make_job(0, [0]))
            t.drain_jobs()
            elapsed_ms = (time.monotonic() - start) * 1000
            assert elapsed_ms >= overhead_ms * 0.9
        finally:
            t.shutdown()

    def test_read_overhead_is_applied(self):
        overhead_ms = 50
        t = SimulatedTierManager(
            offloading_spec=_MOCK_SPEC,
            primary_kv_view=_make_primary_view(),
            tier_type="sim",
            read_overhead_ms=overhead_ms,
            n_read_threads=2,
            n_write_threads=2,
        )
        try:
            t.submit_store(_make_job(0, [0]))
            t.drain_jobs()
            list(t.get_finished_jobs())

            start = time.monotonic()
            t.submit_load(_make_job(1, [0]))
            t.drain_jobs()
            elapsed_ms = (time.monotonic() - start) * 1000
            assert elapsed_ms >= overhead_ms * 0.9
        finally:
            t.shutdown()


class TestQueueBalancing:
    def _make_balancing_tier(
        self, enable: bool = True, prefill_ratio: float = 1.0
    ) -> SimulatedTierManager:
        t = SimulatedTierManager(
            offloading_spec=_MOCK_SPEC,
            primary_kv_view=_make_primary_view(),
            tier_type="sim",
            n_read_threads=2,
            n_write_threads=2,
            enable_queue_balancing=enable,
        )
        t._ratio_controller.ratio = prefill_ratio
        return t

    def _fill_load_queue(self, tier: SimulatedTierManager, count: int):
        """Add no-op tasks to the load queue to simulate backlog."""
        from vllm.v1.kv_offload.tiering.fs.thread_pool import JobState

        with tier._pool._condition:
            for _ in range(count):
                tier._pool._load_q.append((lambda: None, JobState(-1, 1)))

    def test_forces_prefill_when_load_queue_larger(self):
        t = self._make_balancing_tier(enable=True)
        try:
            t._stored_keys = {_to_key(i) for i in range(4)}
            self._fill_load_queue(t, 20)

            keys = [_to_key(i) for i in range(4)]
            ctx = ReqContext(req_id="bal-1")
            results = list(t._lookup_manager.batch_lookup(keys, ctx))

            assert results == [False, False, False, False]
            assert t._prefill_queue_blocks == 4
            assert t._req_prefill_blocks["bal-1"] == 4
        finally:
            t.shutdown()

    def test_normal_hit_when_queues_balanced(self):
        t = self._make_balancing_tier(enable=True)
        try:
            t._stored_keys = {_to_key(i) for i in range(4)}

            keys = [_to_key(i) for i in range(4)]
            ctx = ReqContext(req_id="bal-2")
            results = list(t._lookup_manager.batch_lookup(keys, ctx))

            assert results == [True, True, True, True]
            assert t._prefill_queue_blocks == 0
        finally:
            t.shutdown()

    def test_prefill_counter_decrements_on_request_finish(self):
        t = self._make_balancing_tier(enable=True)
        try:
            t._stored_keys = {_to_key(i) for i in range(4)}
            self._fill_load_queue(t, 20)

            keys = [_to_key(i) for i in range(4)]
            ctx = ReqContext(req_id="bal-3")
            list(t._lookup_manager.batch_lookup(keys, ctx))
            assert t._prefill_queue_blocks == 4

            t.on_request_finished(ctx)
            assert t._prefill_queue_blocks == 0
            assert "bal-3" not in t._req_prefill_blocks
        finally:
            t.shutdown()

    def test_disabled_queue_balancing(self):
        t = self._make_balancing_tier(enable=False)
        try:
            t._stored_keys = {_to_key(i) for i in range(4)}
            self._fill_load_queue(t, 20)

            keys = [_to_key(i) for i in range(4)]
            ctx = ReqContext(req_id="bal-4")
            results = list(t._lookup_manager.batch_lookup(keys, ctx))

            assert results == [True, True, True, True]
        finally:
            t.shutdown()

    def test_no_balancing_when_no_stored_blocks(self):
        t = self._make_balancing_tier(enable=True)
        try:
            self._fill_load_queue(t, 20)

            keys = [_to_key(i) for i in range(4)]
            ctx = ReqContext(req_id="bal-5")
            results = list(t._lookup_manager.batch_lookup(keys, ctx))

            assert results == [False, False, False, False]
            assert t._prefill_queue_blocks == 0
        finally:
            t.shutdown()

    def test_low_prefill_ratio(self):
        # ratio=0.25, load_q=20 → prefill target = 5 blocks
        # First 4-block request fits (4 < 5), second doesn't (8 >= 5)
        t = self._make_balancing_tier(prefill_ratio=0.25)
        try:
            t._stored_keys = {_to_key(i) for i in range(8)}
            self._fill_load_queue(t, 20)

            keys1 = [_to_key(i) for i in range(4)]
            r1 = list(t._lookup_manager.batch_lookup(keys1, ReqContext(req_id="r1")))
            assert r1 == [False, False, False, False]
            assert t._prefill_queue_blocks == 4

            keys2 = [_to_key(i) for i in range(4, 8)]
            r2 = list(t._lookup_manager.batch_lookup(keys2, ReqContext(req_id="r2")))
            assert r2 == [True, True, True, True]
            assert t._prefill_queue_blocks == 4
        finally:
            t.shutdown()

    def test_high_prefill_ratio(self):
        # ratio=2.0, load_q=20 → prefill target = 40 blocks
        # Two 4-block requests both fit (4 < 40, 8 < 40)
        t = self._make_balancing_tier(prefill_ratio=2.0)
        try:
            t._stored_keys = {_to_key(i) for i in range(8)}
            self._fill_load_queue(t, 20)

            keys1 = [_to_key(i) for i in range(4)]
            r1 = list(t._lookup_manager.batch_lookup(keys1, ReqContext(req_id="r1")))
            assert r1 == [False, False, False, False]

            keys2 = [_to_key(i) for i in range(4, 8)]
            r2 = list(t._lookup_manager.batch_lookup(keys2, ReqContext(req_id="r2")))
            assert r2 == [False, False, False, False]
            assert t._prefill_queue_blocks == 8
        finally:
            t.shutdown()


class TestCapacityRatioController:
    """ratio == prefill capacity / load capacity, sampled only while busy."""

    def _make(self, **kw):
        kw.setdefault("tokens_per_block", 16)
        kw.setdefault("ema_alpha", 1.0)
        kw.setdefault("min_busy_s", 1.0)
        kw.setdefault("init_ratio", 1.0)
        return CapacityRatioController(**kw)

    def _prefill_steps(self, c, t0, n, tokens=200, dt=0.125, bound=True):
        """n steps spaced dt apart (exact in binary, so busy time sums
        exactly); 200 tok / 16 per block / 0.125 s = 100 blk/s."""
        t = t0
        for _ in range(n + 1):
            c.on_schedule_end(t, tokens, bound, 0, 0.0)
            t += dt
        return t - dt

    def test_ratio_is_capacity_quotient(self):
        c = self._make()
        # prefill: 100 blk/s; load: 50 blk over 1 s.
        t = self._prefill_steps(c, 0.0, 10)
        c.on_schedule_end(t, 0, False, 50, 1.0)
        assert c.ratio == pytest.approx(2.0)

    def test_ratio_held_until_both_capacities_known(self):
        c = self._make(init_ratio=3.0)
        self._prefill_steps(c, 0.0, 10)
        assert c.ratio == 3.0

    def test_idle_load_queue_does_not_change_ratio(self):
        """No busy samples (nothing routed to load) => estimate and ratio
        stay put, however long the window."""
        c = self._make()
        t = self._prefill_steps(c, 0.0, 10)
        c.on_schedule_end(t, 0, False, 50, 1.0)
        assert c.ratio == pytest.approx(2.0)
        t = self._prefill_steps(c, t + 0.1, 100)
        for _ in range(10):
            c.on_schedule_end(t, 0, False, 0, 0.0)
        assert c.ratio == pytest.approx(2.0)

    def test_steps_below_token_budget_are_not_capacity(self):
        c = self._make()
        c.on_schedule_end(0.0, 10, False, 50, 1.0)
        self._prefill_steps(c, 0.1, 30, tokens=10, bound=False)
        assert c.ratio == 1.0

    def test_idle_gap_between_steps_is_ignored(self):
        c = self._make()
        c.on_schedule_end(0.0, 160, True, 50, 1.0)
        c.on_schedule_end(0.1, 160, True, 0, 0.0)
        c.on_schedule_end(30.0, 160, True, 0, 0.0)
        assert c._prefill._busy_s == pytest.approx(0.1)

    def test_ratio_follows_capacity_change(self):
        c = self._make()
        t = self._prefill_steps(c, 0.0, 10)
        c.on_schedule_end(t, 0, False, 50, 1.0)
        assert c.ratio == pytest.approx(2.0)
        c.on_schedule_end(t, 0, False, 100, 1.0)
        assert c.ratio == pytest.approx(1.0)

    def test_ratio_clamped(self):
        c = self._make()
        t = self._prefill_steps(c, 0.0, 10)
        c.on_schedule_end(t, 0, False, 1, 1.0)
        assert c.ratio == pytest.approx(100.0)


class TestLoadCapacityMeter:
    def test_load_backlog_counts_queued_and_running(self):
        from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

        pool = DualQueueThreadPool(1, 0)
        gate = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            gate.wait(5)

        try:
            pool.enqueue_load(1, 3, [blocker, lambda: None, lambda: None])
            assert started.wait(5)
            assert pool.load_backlog() == 3
            gate.set()
            pool.wait_idle()
            assert pool.load_backlog() == 0
        finally:
            gate.set()
            pool.shutdown()

    def test_sample_only_counts_saturated_time(self):
        from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

        pool = DualQueueThreadPool(1, 0)
        try:
            pool.enqueue_load(1, 1, [lambda: time.sleep(0.05)])
            pool.wait_idle()
            time.sleep(0.2)
            blocks, busy_s = pool.pop_load_capacity_sample()
            assert blocks == 0
            assert busy_s < 0.15

            n = 10
            pool.enqueue_load(2, n, [lambda: time.sleep(0.02)] * n)
            pool.wait_idle()
            blocks, busy_s = pool.pop_load_capacity_sample()
            assert blocks >= n - 2
            assert 0.1 < busy_s < 0.3
        finally:
            pool.shutdown()


class TestPrefillBacklog:
    def _tier(self):
        t = SimulatedTierManager(
            offloading_spec=_MOCK_SPEC,
            primary_kv_view=_make_primary_view(),
            tier_type="sim",
            n_read_threads=2,
            n_write_threads=2,
            enable_queue_balancing=True,
        )
        t._stored_keys = {_to_key(i) for i in range(4)}
        return t

    def test_backlog_shrinks_as_blocks_are_prefilled(self):
        t = self._tier()
        try:
            with t._pool._condition:
                for _ in range(20):
                    t._pool._load_q.append((lambda: None, JobState(-1, 1)))
            keys = [_to_key(i) for i in range(4)]
            list(t._lookup_manager.batch_lookup(keys, ReqContext(req_id="r")))
            assert t._req_prefill_blocks["r"] == 4
            # 16 tokens/block: 40 tokens -> 3 blocks, then clamp at 0.
            t.on_schedule_end(
                ScheduleEndContext(
                    new_req_ids=[],
                    preempted_req_ids=(),
                    prefill_tokens_by_req={"r": 40},
                )
            )
            assert t._req_prefill_blocks["r"] == 1
            assert t._prefill_queue_blocks == 1
            t.on_schedule_end(
                ScheduleEndContext(
                    new_req_ids=[],
                    preempted_req_ids=(),
                    prefill_tokens_by_req={"r": 400},
                )
            )
            assert t._req_prefill_blocks["r"] == 0
            assert t._prefill_queue_blocks == 0
        finally:
            t.shutdown()

    def test_running_loads_count_as_load_backlog(self):
        """With work in flight on the load path (nothing queued), a request
        is still diverted to prefill when the prefill backlog is small."""
        t = self._tier()
        try:
            t._ratio_controller.ratio = 10.0
            t._pool._load_active = 2
            keys = [_to_key(i) for i in range(4)]
            res = list(t._lookup_manager.batch_lookup(keys, ReqContext(req_id="x")))
            assert res == [False] * 4
        finally:
            t.shutdown()


class TestShutdown:
    def test_shutdown_after_store(self):
        t = _make_tier()
        t.submit_store(_make_job(0, [0, 1, 2]))
        t.drain_jobs()
        t.shutdown()

    def test_shutdown_idle(self):
        t = _make_tier()
        t.shutdown()
