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
from vllm.v1.kv_offload.tiering.sim.manager import SimulatedTierManager

_CTX = ReqContext(req_id="test-sim")
_MOCK_SPEC = MagicMock()
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
    read_delay_ms: float = 0.0,
    write_delay_ms: float = 0.0,
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
        read_delay_ms=read_delay_ms,
        write_delay_ms=write_delay_ms,
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
    def test_write_delay_is_applied(self):
        delay_ms = 50
        t = _make_tier(write_delay_ms=delay_ms)
        try:
            start = time.monotonic()
            t.submit_store(_make_job(0, [0]))
            t.drain_jobs()
            elapsed_ms = (time.monotonic() - start) * 1000
            assert elapsed_ms >= delay_ms * 0.9
        finally:
            t.shutdown()

    def test_read_delay_is_applied(self):
        delay_ms = 50
        t = _make_tier(read_delay_ms=delay_ms)
        try:
            t.submit_store(_make_job(0, [0]))
            t.drain_jobs()
            list(t.get_finished_jobs())

            start = time.monotonic()
            t.submit_load(_make_job(1, [0]))
            t.drain_jobs()
            elapsed_ms = (time.monotonic() - start) * 1000
            assert elapsed_ms >= delay_ms * 0.9
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
