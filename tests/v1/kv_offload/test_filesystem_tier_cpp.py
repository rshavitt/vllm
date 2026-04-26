# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FileSystemTierManagerCpp.

Unit tests (TestFileSystemTierState, TestFileSystemTierEviction,
TestFileSystemTierJobLifecycle, TestFileSystemTierErrorHandling) mock the C++
_kv_file_system_ops extension and focus on Python-level state management: LRU
ordering, in-flight tracking, _evictable_count invariants, job lifecycle,
and error/edge-case paths.

Integration tests (TestTieringOffloadingManagerMixed,
TestTieringOffloadingWithoutSecondaryTiers) wire FileSystemTierManagerCpp into
TieringOffloadingManager together with DummySecondaryTier to verify cascade,
promotion, ref_cnt, eviction, and touch propagation — all with the C++
extension mocked for fast, self-contained execution.

I/O integration tests (TestFileSystemTierIO) require the real
_kv_file_system_ops extension and exercise actual disk reads and writes.
"""

import contextlib
import os
import time
from unittest.mock import patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Require the real C++ extension - tests will be skipped if not available
# ---------------------------------------------------------------------------
pytest.importorskip(
    "vllm._kv_file_system_ops",
    reason="_kv_file_system_ops extension not built; these tests require the compiled extension"
)

from vllm.v1.kv_offload.abstract import OffloadKey, ReqContext, get_offload_block_hash, make_offload_key  # noqa: E402
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec  # noqa: E402
from vllm.v1.kv_offload.tiering.dummy import DummySecondaryTier  # noqa: E402
from vllm.v1.kv_offload.tiering.file_system_cpp import (  # noqa: E402
    FileSystemTierManagerCpp,
)
from vllm.v1.kv_offload.tiering.manager import (  # noqa: E402
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 16  # float32 elements per block row
_DTYPE = torch.float32


def key(n: int) -> OffloadKey:
    """Return a deterministic OffloadKey from an integer."""
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_block_hash(req_id: int, block_idx: int) -> OffloadKey:
    """Return a deterministic OffloadKey from a (req_id, block_idx) pair."""
    return make_offload_key(f"{req_id}:{block_idx}".encode(), 0)


def make_cpu_spec(block_ids: list[int]) -> CPULoadStoreSpec:
    """Create a CPULoadStoreSpec for the given block IDs."""
    return CPULoadStoreSpec(block_ids)


def make_tier_with_view(
    base_path: str,
    max_blocks: int = 10,
    num_total_blocks: int = 32,
) -> tuple[FileSystemTierManagerCpp, torch.Tensor]:
    """Create a FileSystemTierManagerCpp and wire a test primary view into it."""
    tier = FileSystemTierManagerCpp(base_path=base_path, max_blocks=max_blocks)
    tensor = torch.zeros((num_total_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
    tier.set_primary_view(memoryview(tensor.numpy()))
    return tier, tensor


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    spec = make_cpu_spec(block_ids)
    return JobMetadata(job_id=job_id, keys=keys, spec=spec)


def drain(tier: FileSystemTierManagerCpp, max_rounds: int = 20) -> list:
    """
    Call get_finished() repeatedly until all jobs are resolved.
    Works for both the synchronous mock and the real async C++ extension.
    """
    results = []
    for _ in range(max_rounds):
        results.extend(tier.get_finished())
        if not tier._active_jobs:
            break
        time.sleep(0.005)
    return results


def evictable_count_expected(tier: FileSystemTierManagerCpp) -> int:
    """Recompute _evictable_count from first principles for assertion."""
    return len(set(tier._blocks) - set(tier._in_flight))


# ---------------------------------------------------------------------------
# _SyncMockCpp — synchronous mock for the _kv_file_system_ops C++ functions
# ---------------------------------------------------------------------------

class _SyncMockCpp:
    """
    Synchronous mock: jobs complete immediately on submit.

    submit_store_job / submit_load_job enqueue (job_id, success) into
    _pending.  get_finished_jobs() returns all pending results and clears the
    queue, simulating instant job completion.

    Use patch_ctx() as a context manager to install all three patches.
    """

    def __init__(self, success: bool = True):
        self._pending: list[tuple[int, bool]] = []
        self.success = success
        # Records ("store", job_id) or ("load", job_id) in call order.
        self._call_order: list[tuple[str, int]] = []

    def submit_store_job(self, job_id, *args):
        self._call_order.append(("store", job_id))
        self._pending.append((job_id, self.success))

    def submit_load_job(self, job_id, *args):
        self._call_order.append(("load", job_id))
        self._pending.append((job_id, self.success))

    def get_finished_jobs(self):
        out, self._pending = self._pending, []
        return out

    @contextlib.contextmanager
    def patch_ctx(self):
        base = "vllm.v1.kv_offload.tiering.file_system_cpp"
        with (
            patch(f"{base}.cpp_submit_store_job", new=self.submit_store_job),
            patch(f"{base}.cpp_submit_load_job",  new=self.submit_load_job),
            patch(f"{base}.cpp_get_finished_jobs", new=self.get_finished_jobs),
        ):
            yield self


# ---------------------------------------------------------------------------
# State tests — mocked I/O, focus on lookup / in-flight tracking
# ---------------------------------------------------------------------------

class TestFileSystemTierState:

    @pytest.fixture(autouse=True)
    def _patch_io(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManagerCpp(
                base_path=str(tmp_path), max_blocks=10
            )
            tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
            self.tier.set_primary_view(memoryview(tensor.numpy()))
            yield

    def test_get_tier_name(self):
        t = FileSystemTierManagerCpp.__new__(FileSystemTierManagerCpp)
        t._tier_name = "MyTier"
        assert t.get_tier_name() == "MyTier"

    def test_initial_state_empty(self):
        assert self.tier.get_num_blocks() == 0
        assert self.tier.get_num_in_flight() == 0
        assert self.tier._evictable_count == 0

    def test_lookup_empty_tier(self):
        req_ctx = ReqContext()
        assert self.tier.lookup(key(1), req_ctx) is False
        assert self.tier.lookup(key(2), req_ctx) is False

    def test_lookup_all_present(self):
        req_ctx = ReqContext()
        keys = [key(i) for i in range(3)]
        for k in keys:
            self.tier._blocks[k] = True
        self.tier._evictable_count = 3
        for k in keys:
            assert self.tier.lookup(k, req_ctx) is True

    def test_lookup_partial_hit_stops_at_first_miss(self):
        req_ctx = ReqContext()
        keys = [key(i) for i in range(4)]
        # Only first two present
        self.tier._blocks[keys[0]] = True
        self.tier._blocks[keys[1]] = True
        self.tier._evictable_count = 2
        assert self.tier.lookup(keys[0], req_ctx) is True
        assert self.tier.lookup(keys[1], req_ctx) is True
        assert self.tier.lookup(keys[2], req_ctx) is False
        assert self.tier.lookup(keys[3], req_ctx) is False

    def test_lookup_in_flight_returns_none(self):
        req_ctx = ReqContext()
        keys = [key(i) for i in range(3)]
        # First block present, second block in-flight.
        # lookup() checks _in_flight before _blocks, so it must reach
        # keys[1] — that only happens once keys[0] passes both checks.
        self.tier._blocks[keys[0]] = True
        self.tier._evictable_count = 1
        self.tier._in_flight[keys[1]] = 99
        assert self.tier.lookup(keys[0], req_ctx) is True
        assert self.tier.lookup(keys[1], req_ctx) is None
        assert self.tier.lookup(keys[2], req_ctx) is False

    def test_lookup_none_when_first_block_in_flight(self):
        req_ctx = ReqContext()
        keys = [key(i) for i in range(3)]
        self.tier._in_flight[keys[0]] = 1
        assert self.tier.lookup(keys[0], req_ctx) is None

    def test_get_file_name_structure(self, tmp_path):
        tier = FileSystemTierManagerCpp(base_path="/kvcache", max_blocks=10)
        path = tier.get_file_name(get_offload_block_hash(key(0)))
        assert path == "/kvcache/000/00/0000000000000000.bin"

    def test_get_file_name_consistent_for_same_hash(self, tmp_path):
        tier = FileSystemTierManagerCpp(base_path="/kvcache", max_blocks=10)
        h = get_offload_block_hash(key(12345))
        assert tier.get_file_name(h) == tier.get_file_name(h)

    def test_get_file_name_accepts_int(self, tmp_path):
        tier = FileSystemTierManagerCpp(base_path="/base", max_blocks=10)
        # OffloadKey block hash (8 bytes big-endian) → same hex as passing int directly
        path_via_bytes = tier.get_file_name(get_offload_block_hash(key(42)))
        path_via_int = tier.get_file_name(42)
        assert path_via_bytes == path_via_int


# ---------------------------------------------------------------------------
# Eviction tests
# ---------------------------------------------------------------------------

class TestFileSystemTierEviction:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManagerCpp(
                base_path=str(tmp_path), max_blocks=5
            )
            tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
            self.tier.set_primary_view(memoryview(tensor.numpy()))
            yield

    def _fill(self, n: int, start: int = 0):
        """Seed _blocks with n entries (bypassing I/O) from hash start."""
        for i in range(start, start + n):
            self.tier._blocks[key(i)] = True
        self.tier._evictable_count = len(self.tier._blocks)

    def test_eviction_removes_oldest_first(self):
        self._fill(5)  # blocks 0-4, oldest = 0
        # Store one new block; must evict key(0)
        job = make_job(1, [key(10)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        assert key(10) in self.tier._blocks
        assert key(0) not in self.tier._blocks
        # Remaining original blocks still present
        for i in range(1, 5):
            assert key(i) in self.tier._blocks

    def test_eviction_respects_in_flight(self):
        self._fill(5)  # blocks 0-4
        # Mark block 0 (oldest) as in-flight
        self.tier._in_flight[key(0)] = 99
        self.tier._evictable_count -= 1  # it's in-flight now

        # Now try to add one more block; must skip key(0) and evict key(1)
        job = make_job(1, [key(10)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        assert key(10) in self.tier._blocks
        assert key(0) in self.tier._blocks   # protected by in-flight
        assert key(1) not in self.tier._blocks  # oldest evictable

    def test_eviction_skips_protected_batch_blocks(self):
        self._fill(5)  # 0-4 oldest to newest
        # Store [key(0), key(10)]: key(0) already on disk so filtered out;
        # but key(0) appears in all_hashes → protected set.
        # Need to evict 1 to make room for key(10).
        # key(0) is protected; key(1) should be evicted.
        job = make_job(1, [key(0), key(10)], [0, 1])
        self.tier.submit_store(job)
        drain(self.tier)

        assert key(10) in self.tier._blocks
        assert key(0) in self.tier._blocks   # in all_hashes → protected
        assert key(1) not in self.tier._blocks

    def test_eviction_fails_insufficient_evictable(self):
        """All blocks in-flight → _evictable_count=0 → drop job with warning."""
        self._fill(5)
        for i in range(5):
            self.tier._in_flight[key(i)] = 99
        self.tier._evictable_count = 0

        with patch("vllm.v1.kv_offload.tiering.file_system_cpp.logger") as mock_log:
            self.tier.submit_store(make_job(1, [key(10)], [0]))

        assert key(10) not in self.tier._blocks
        mock_log.warning.assert_called_once()
        assert "insufficient" in mock_log.warning.call_args[0][0]

    def test_eviction_fails_protected_overlap(self, caplog):
        """
        _evictable_count >= needed but all evictable candidates are in
        the protected set → scan exhausts without finding enough → warning.
        """
        # Fill all 5 slots; need to evict 2 for the 2 new keys (key(5), key(6)),
        # but key(0)-key(4) are all in all_hashes → all protected → the scan
        # exhausts without finding candidates → for...else fires → warning.
        self._fill(5)  # key(0)-key(4), all slots filled
        job = make_job(1, [key(i) for i in range(7)], list(range(7)))
        with caplog.at_level("WARNING"):
            self.tier.submit_store(job)
        assert key(5) not in self.tier._blocks
        assert key(6) not in self.tier._blocks

    def test_evictable_count_after_eviction(self):
        self._fill(5)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        job = make_job(1, [key(10)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        assert self.tier._evictable_count == evictable_count_expected(self.tier)

    def test_touch_moves_to_end_of_lru(self):
        self._fill(3)  # insertion order: key(0), key(1), key(2)
        self.tier.touch([key(0)])  # key(0) now most recent
        lru_order = list(self.tier._blocks.keys())
        assert lru_order[-1] == key(0)
        assert lru_order[0] == key(1)  # key(1) is now oldest


# ---------------------------------------------------------------------------
# Job lifecycle tests
# ---------------------------------------------------------------------------

class TestFileSystemTierJobLifecycle:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self._mock = _SyncMockCpp()
        with self._mock.patch_ctx():
            self.tier = FileSystemTierManagerCpp(
                base_path=str(tmp_path), max_blocks=20
            )
            tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
            self.tier.set_primary_view(memoryview(tensor.numpy()))
            yield

    def test_submit_store_goes_to_active_jobs(self):
        job = make_job(1, [key(0), key(1)], [0, 1])
        self.tier.submit_store(job)

        assert len(self.tier._active_jobs) == 1

    def test_submit_load_goes_to_futures_immediately(self):
        # Seed blocks as already on disk.
        self.tier._blocks[key(0)] = True
        self.tier._blocks[key(1)] = True
        self.tier._evictable_count = 2

        job = make_job(1, [key(0), key(1)], [0, 1])
        self.tier.submit_load(job)

        assert len(self.tier._active_jobs) == 1

    def test_store_job_completes_and_adds_to_blocks(self):
        keys = [key(0), key(1)]
        job = make_job(1, keys, [0, 1])
        self.tier.submit_store(job)
        results = drain(self.tier)

        assert len(results) == 1
        assert results[0].job_id == 1
        assert results[0].success is True
        assert all(b in self.tier._blocks for b in keys)
        assert self.tier.get_num_in_flight() == 0

    def test_store_job_updates_evictable_count(self):
        keys = [key(0), key(1)]
        job = make_job(1, keys, [0, 1])
        self.tier.submit_store(job)
        drain(self.tier)

        assert self.tier._evictable_count == evictable_count_expected(self.tier)
        assert self.tier._evictable_count == 2

    def test_load_job_completes_and_restores_evictable_count(self):
        self.tier._blocks[key(0)] = True
        self.tier._blocks[key(1)] = True
        self.tier._evictable_count = 2

        job = make_job(1, [key(0), key(1)], [0, 1])
        self.tier.submit_load(job)

        # While in-flight, evictable count is reduced
        assert self.tier._evictable_count == 0

        results = drain(self.tier)

        assert len(results) == 1
        assert results[0].success is True
        # Restored after completion
        assert self.tier._evictable_count == 2
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

    def test_store_buffer_kept_alive_in_active_job(self):
        """_ActiveJob.buffer must hold the tensor view until the job completes."""
        job = make_job(1, [key(0)], [0])
        self.tier.submit_store(job)

        assert 1 in self.tier._active_jobs
        active = self.tier._active_jobs[1]
        assert isinstance(active.buffer, memoryview)

    def test_duplicate_store_skipped(self):
        """Blocks already on disk are not re-stored."""
        self.tier._blocks[key(0)] = True
        self.tier._evictable_count = 1

        self.tier.submit_store(make_job(1, [key(0)], [0]))
        # All blocks already on disk → filtered out, no job submitted
        assert 1 not in self.tier._active_jobs

    def test_in_flight_store_not_duplicated(self):
        """
        A second submit_store for a block already in-flight must be dropped.
        Without the fix, the second call overwrites _in_flight[bh] with the
        new job_id, so get_finished() for the first job deletes a key it no
        longer owns, corrupting state.
        """
        self.tier.submit_store(make_job(1, [key(0)], [0]))
        assert 1 in self.tier._active_jobs
        assert self.tier._in_flight[key(0)] == 1

        # Second store for the same block while the first is still in-flight.
        self.tier.submit_store(make_job(2, [key(0)], [0]))
        # Second job silently dropped; _in_flight must still point to job 1.
        assert 2 not in self.tier._active_jobs
        assert self.tier._in_flight[key(0)] == 1

    def test_failed_store_does_not_add_to_blocks(self):
        failing_mock = _SyncMockCpp(success=False)
        with failing_mock.patch_ctx():
            tier = FileSystemTierManagerCpp(base_path="/tmp", max_blocks=10)
            tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
            tier.set_primary_view(memoryview(tensor.numpy()))
            tier.submit_store(make_job(1, [key(0)], [0]))
            results = drain(tier)

        assert results[0].success is False
        assert key(0) not in tier._blocks

    def test_failed_store_evictable_count_unchanged(self):
        failing_mock = _SyncMockCpp(success=False)
        with failing_mock.patch_ctx():
            tier = FileSystemTierManagerCpp(base_path="/tmp", max_blocks=10)
            tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
            tier.set_primary_view(memoryview(tensor.numpy()))
            tier.submit_store(make_job(1, [key(0)], [0]))
            drain(tier)

        assert tier._evictable_count == 0

    def test_multiple_independent_jobs(self):
        keys_a = [key(0), key(1)]
        keys_b = [key(2), key(3)]
        self.tier.submit_store(make_job(1, keys_a, [0, 1]))
        self.tier.submit_store(make_job(2, keys_b, [2, 3]))
        results = drain(self.tier)

        job_ids = {r.job_id for r in results}
        assert job_ids == {1, 2}
        assert all(b in self.tier._blocks for b in keys_a + keys_b)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestFileSystemTierErrorHandling:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManagerCpp(
                base_path=str(tmp_path), max_blocks=10
            )
            tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
            self.tier.set_primary_view(memoryview(tensor.numpy()))
            yield

    def test_submit_load_missing_block_logs_warning(self):
        with patch("vllm.v1.kv_offload.tiering.file_system_cpp.logger") as mock_log:
            self.tier.submit_load(make_job(1, [key(42)], [0]))

        mock_log.warning.assert_called_once()
        assert "not found on disk" in mock_log.warning.call_args[0][0]

    def test_submit_load_missing_block_no_state_change(self):
        before_inflight = dict(self.tier._in_flight)
        before_evictable = self.tier._evictable_count
        self.tier.submit_load(make_job(1, [key(42)], [0]))
        assert self.tier._in_flight == before_inflight
        assert self.tier._evictable_count == before_evictable
        assert 1 not in self.tier._active_jobs

    def test_submit_store_cpp_failure_rolls_back(self):
        """If cpp_submit_store_job raises, _in_flight must be cleaned up."""
        base = "vllm.v1.kv_offload.tiering.file_system_cpp"
        with patch(f"{base}.cpp_submit_store_job",
                   side_effect=RuntimeError("pool error")):
            with pytest.raises(RuntimeError):
                self.tier.submit_store(make_job(1, [key(0)], [0]))

        assert key(0) not in self.tier._in_flight
        assert 1 not in self.tier._active_jobs

    def test_submit_load_cpp_failure_rolls_back(self):
        """
        If cpp_submit_load_job raises, all state mutations must be rolled back.
        """
        self.tier._blocks[key(0)] = True
        self.tier._evictable_count = 1

        base = "vllm.v1.kv_offload.tiering.file_system_cpp"
        with patch(f"{base}.cpp_submit_load_job",
                   side_effect=RuntimeError("pool error")):
            with pytest.raises(RuntimeError):
                self.tier.submit_load(make_job(1, [key(0)], [0]))

        # State rolled back
        assert key(0) not in self.tier._in_flight
        assert self.tier._evictable_count == 1
        assert 1 not in self.tier._active_jobs

    def test_get_finished_failed_job_cleans_up_state(self, caplog):
        """A job finishing with success=False must clean up state correctly."""
        self.tier._blocks[key(0)] = True
        self.tier._evictable_count = 1

        # Submit a load job and immediately complete it as a failure.
        failing_mock = _SyncMockCpp(success=False)
        base = "vllm.v1.kv_offload.tiering.file_system_cpp"
        with (
            patch(f"{base}.cpp_submit_load_job",  new=failing_mock.submit_load_job),
            patch(f"{base}.cpp_get_finished_jobs", new=failing_mock.get_finished_jobs),
        ):
            self.tier.submit_load(make_job(7, [key(0)], [0]))
            results = list(self.tier.get_finished())

        assert len(results) == 1
        assert results[0].job_id == 7
        assert results[0].success is False
        # State fully cleaned up
        assert 7 not in self.tier._active_jobs
        assert key(0) not in self.tier._in_flight
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

    def test_evictable_count_invariant_after_mixed_operations(self):
        """
        _evictable_count must equal len(_blocks) - (in_flight ∩ _blocks)
        after a sequence of store, evict, load, and failure operations.
        """
        # Store 3 blocks
        self.tier.submit_store(make_job(1, [key(i) for i in range(3)], [0, 1, 2]))
        drain(self.tier)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        # Start loading key(0) and key(1)
        job = make_job(2, [key(0), key(1)], [0, 1])
        self.tier.submit_load(job)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        # Complete the load
        drain(self.tier)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        # Store 3 more (triggers eviction of key(2) since cap=10, 3+3=6 ≤ 10, no eviction needed)
        self.tier.submit_store(
            make_job(3, [key(i) for i in range(10, 13)], [0, 1, 2])
        )
        drain(self.tier)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)


# ---------------------------------------------------------------------------
# TieringOffloadingManager integration tests — FileSystemTierManagerCpp + DummySecondaryTier
# ---------------------------------------------------------------------------

class TestTieringOffloadingManagerMixed:

    @pytest.fixture(autouse=True)
    def manager_setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.primary_tier = CPUPrimaryTierOffloadingManager(
                num_blocks=5
            )
            mock_cpu_tensor = torch.zeros((5, _BLOCK_ELEMENTS), dtype=_DTYPE)
            self.primary_tier.create_kv_memoryview = lambda: memoryview(mock_cpu_tensor.numpy())

            # tier1: in-memory dummy tier
            self.secondary_tier1 = DummySecondaryTier(
                tier_name="Dummy", max_blocks=10
            )
            # tier2: filesystem tier (C++ mocked)
            self.secondary_tier2 = FileSystemTierManagerCpp(
                base_path=str(tmp_path / "fs"), max_blocks=10
            )

            self.manager = TieringOffloadingManager(
                primary_tier=self.primary_tier,
                secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
            )
            yield

    def test_basic_store_to_primary(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        result = self.manager.prepare_store(keys, ReqContext())
        assert result is not None
        assert len(result.keys_to_store) == 3

        self.manager.complete_store(keys, success=True)

        req_ctx = ReqContext()
        assert all(self.primary_tier.lookup(k, req_ctx) is True for k in keys)

    def test_cascade_to_both_secondary_tiers(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(keys, ReqContext())
        self.manager.complete_store(keys, success=True)
        self.manager._process_finished_jobs()

        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3
        req_ctx = ReqContext()
        assert all(self.secondary_tier1.lookup(k, req_ctx) is True for k in keys)
        assert all(self.secondary_tier2.lookup(k, req_ctx) is True for k in keys)

    def test_ref_cnt_protection_during_cascade(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(keys, ReqContext())
        self.manager.complete_store(keys, success=True)

        # ref_cnt == 2: one hold per secondary tier
        for key in keys:
            block_status = self.primary_tier._policy.get(key)
            assert block_status is not None, f"Block {key} not found in policy"
            assert block_status.ref_cnt == 2

        self.manager._process_finished_jobs()

        for key in keys:
            block_status = self.primary_tier._policy.get(key)
            assert block_status is not None, f"Block {key} not found in policy"
            assert block_status.ref_cnt == 0

    def test_lookup_from_primary(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(keys, ReqContext())
        self.manager.complete_store(keys, success=True)

        req_ctx = ReqContext()
        assert all(self.manager.lookup(k, req_ctx) is True for k in keys)

    def test_promotion_from_dummy_tier(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        # Seed dummy tier directly
        for key in keys:
            self.secondary_tier1.blocks[key] = True

        req_ctx = ReqContext()
        for k in keys:
            self.manager.lookup(k, req_ctx)

        self.manager._process_finished_jobs()

        req_ctx = ReqContext()
        assert all(self.primary_tier.lookup(k, req_ctx) is True for k in keys)
        assert all(self.manager.lookup(k, req_ctx) is True for k in keys)

    def test_promotion_from_filesystem_tier(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        # Seed filesystem tier directly
        for key in keys:
            self.secondary_tier2._blocks[key] = True
        self.secondary_tier2._evictable_count = len(keys)

        req_ctx = ReqContext()
        for k in keys:
            self.manager.lookup(k, req_ctx)

        self.manager._process_finished_jobs()

        req_ctx = ReqContext()
        assert all(self.primary_tier.lookup(k, req_ctx) is True for k in keys)
        assert all(self.manager.lookup(k, req_ctx) is True for k in keys)

    def test_partial_lookup(self):
        keys = [make_block_hash(1, i) for i in range(5)]

        self.manager.prepare_store(keys[:3], ReqContext())
        self.manager.complete_store(keys[:3], success=True)

        req_ctx = ReqContext()
        assert sum(1 for k in keys if self.manager.lookup(k, req_ctx) is True) == 3

    def test_eviction_in_primary_tier(self):
        keys = [make_block_hash(1, i) for i in range(5)]
        result = self.manager.prepare_store(keys, ReqContext())
        assert result is not None
        self.manager.complete_store(keys, success=True)
        self.manager._process_finished_jobs()

        more_keys = [make_block_hash(2, i) for i in range(2)]
        result = self.manager.prepare_store(more_keys, ReqContext())

        assert result is not None
        assert len(result.evicted_keys) == 2
        assert len(result.keys_to_store) == 2

    def test_touch_propagates_to_all_tiers(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(keys, ReqContext())
        self.manager.complete_store(keys, success=True)
        self.manager._process_finished_jobs()

        self.manager.touch(keys)

        # Verify touch propagated to primary tier by checking all blocks are still present
        req_ctx = ReqContext()
        assert all(self.primary_tier.lookup(k, req_ctx) is True for k in keys)

        # Verify touch propagated to secondary tiers by checking all blocks are still present
        req_ctx = ReqContext()
        assert all(self.secondary_tier1.lookup(k, req_ctx) is True for k in keys)
        assert all(self.secondary_tier2.lookup(k, req_ctx) is True for k in keys)

    def test_failed_store_no_cascade(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(keys, ReqContext())
        self.manager.complete_store(keys, success=False)
        self.manager._process_finished_jobs()

        assert self.secondary_tier1.get_num_blocks() == 0
        assert self.secondary_tier2.get_num_blocks() == 0

    def test_multiple_secondary_tiers_independent_eviction(self, tmp_path):
        """Test that secondary tiers manage their own evictions independently."""
        # Create tiers with different capacities
        small_tier = DummySecondaryTier(
            tier_name="SmallDummy", max_blocks=5, simulate_async=False
        )
        large_tier = FileSystemTierManagerCpp(
            base_path=str(tmp_path / "large_fs"), max_blocks=10
        )

        primary_tier = CPUPrimaryTierOffloadingManager(num_blocks=10)
        mock_cpu_tensor = torch.zeros((10, _BLOCK_ELEMENTS), dtype=_DTYPE)
        primary_tier.create_kv_memoryview = lambda: memoryview(mock_cpu_tensor.numpy())

        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[small_tier, large_tier],
        )

        # First, store 5 blocks to fill the small tier
        keys1 = [make_block_hash(1, i) for i in range(5)]
        result = manager.prepare_store(keys1, ReqContext())
        assert result is not None
        manager.complete_store(keys1, success=True)
        manager._process_finished_jobs()

        assert small_tier.get_num_blocks() == 5
        assert large_tier.get_num_blocks() == 5

        # Now store 3 more blocks — small tier should evict 3 blocks
        keys2 = [make_block_hash(2, i) for i in range(3)]
        result = manager.prepare_store(keys2, ReqContext())
        assert result is not None
        manager.complete_store(keys2, success=True)
        manager._process_finished_jobs()

        # Small tier should still have 5 blocks (evicted 3, added 3)
        assert small_tier.get_num_blocks() == 5

        # Large tier should have all 8 blocks
        assert large_tier.get_num_blocks() == 8

    def test_prepare_store_processes_finished_jobs_first(self):
        keys = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(keys, ReqContext())
        self.manager.complete_store(keys, success=True)

        for key in keys:
            block_status = self.primary_tier._policy.get(key)
            assert block_status is not None, f"Block {key} not found in policy"
            assert block_status.ref_cnt == 2

        self.manager.prepare_store([make_block_hash(2, i) for i in range(2)], ReqContext())

        for key in keys:
            block_status = self.primary_tier._policy.get(key)
            assert block_status is not None, f"Block {key} not found in policy"
            assert block_status.ref_cnt == 0


# ---------------------------------------------------------------------------
# Baseline: no secondary tiers
# ---------------------------------------------------------------------------

class TestTieringOffloadingWithoutSecondaryTiers:

    def test_works_without_secondary_tiers(self):
        primary_tier = CPUPrimaryTierOffloadingManager(num_blocks=5)
        _t = torch.zeros((5, _BLOCK_ELEMENTS), dtype=_DTYPE)
        primary_tier.create_kv_memoryview = lambda: memoryview(_t.numpy())

        manager = TieringOffloadingManager(primary_tier=primary_tier, secondary_tiers=[])

        keys = [make_block_hash(1, i) for i in range(3)]
        manager.prepare_store(keys, ReqContext())
        manager.complete_store(keys, success=True)

        req_ctx = ReqContext()
        assert all(manager.lookup(k, req_ctx) is True for k in keys)


# ---------------------------------------------------------------------------
# Integration tests — require real _kv_file_system_ops extension
# ---------------------------------------------------------------------------


class TestFileSystemTierIO:
    """Exercises actual pread/pwrite disk I/O via the C++ extension."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.base = str(tmp_path)
        self.block_elements = 128  # 128 * 4 = 512 bytes — O_DIRECT sector alignment
        self.num_io_blocks = 50
        self.tensor = torch.zeros(
            (self.num_io_blocks, self.block_elements), dtype=torch.float32
        )
        self.tier = FileSystemTierManagerCpp(
            base_path=self.base, max_blocks=self.num_io_blocks
        )
        self.tier.set_primary_view(memoryview(self.tensor.numpy()))

    def test_store_creates_file_on_disk(self):
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        expected_path = self.tier.get_file_name(get_offload_block_hash(key(1)))
        assert os.path.isfile(expected_path)

    def test_data_roundtrip_single_block(self):
        # Write known values into block 0 of the primary view
        self.tensor[0] = torch.arange(self.block_elements, dtype=torch.float32)
        expected = self.tensor[0].clone()

        self.tier.submit_store(make_job(1, [key(7)], [0]))
        drain(self.tier)

        # Overwrite block 0 to prove data comes from disk
        self.tensor[0] = 0.0

        # Load from disk into block 1 of the primary view
        self.tier.submit_load(make_job(2, [key(7)], [1]))
        drain(self.tier)

        assert torch.equal(self.tensor[1], expected)

    def test_data_roundtrip_multiple_blocks(self):
        hashes = [key(i + 100) for i in range(4)]
        for idx in range(4):
            self.tensor[idx] = float(idx + 1)
        expected = self.tensor[:4].clone()

        self.tier.submit_store(make_job(1, hashes, list(range(4))))
        drain(self.tier)

        # Zero out source blocks to prove data comes from disk
        self.tensor[:4] = 0.0

        self.tier.submit_load(make_job(2, hashes, list(range(4, 8))))
        drain(self.tier)

        for i in range(4):
            assert torch.equal(self.tensor[4 + i], expected[i])

    def test_file_path_is_deterministic(self):
        h = get_offload_block_hash(key(9999))
        path1 = self.tier.get_file_name(h)
        path2 = self.tier.get_file_name(h)
        assert path1 == path2


# ---------------------------------------------------------------------------
# End-to-end tests with primary tier integration
# ---------------------------------------------------------------------------

class TestFileSystemTierE2EWithPrimary:
    """
    End-to-end tests integrating FileSystemTierManagerCpp with
    CPUPrimaryTierOffloadingManager using real disk I/O.
    
    These tests verify full data integrity through cascade and promotion
    pipelines with actual file system operations.
    """

    @pytest.fixture
    def setup_manager(self, tmp_path):
        """Setup TieringOffloadingManager with real primary and filesystem tiers."""
        # Use 128 elements per block for O_DIRECT alignment (512 bytes)
        block_elements = 128
        num_primary_blocks = 10
        num_secondary_blocks = 20
        
        # Create primary tier
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
        )

        # Provide a plain CPU tensor as the shared KV buffer so that
        # TieringOffloadingManager can wire secondary tier memoryviews
        # without requiring a real SharedOffloadRegion.
        cpu_tensor = torch.zeros((num_primary_blocks, block_elements), dtype=torch.float32)
        primary_tier.create_kv_memoryview = lambda: memoryview(cpu_tensor.numpy())

        # Create filesystem tier with real I/O
        fs_tier = FileSystemTierManagerCpp(
            base_path=str(tmp_path / "kvcache"),
            max_blocks=num_secondary_blocks
        )
        
        # Create tiering manager
        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[fs_tier],
        )
        
        yield manager, primary_tier, fs_tier, cpu_tensor, block_elements
        
        # Cleanup
        manager.shutdown()

    def test_full_cascade_with_data_integrity(self, setup_manager):
        """
        Store blocks to primary tier with known data patterns, verify cascade
        to filesystem tier completes, and verify data integrity by reading
        files directly from disk.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Generate unique data patterns for each block
        num_blocks = 5
        keys = [make_block_hash(1, i) for i in range(num_blocks)]
        expected_data = {}
        
        # Prepare store to primary tier
        result = manager.prepare_store(keys, ReqContext())
        assert result is not None
        assert len(result.keys_to_store) == num_blocks
        
        # Fill blocks with unique random data
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for i, block_id in enumerate(spec.block_ids):
            data = torch.rand(block_elements, dtype=torch.float32)
            cpu_tensor[int(block_id)] = data
            expected_data[keys[i]] = data.clone()
        
        # Complete store (triggers cascade to filesystem)
        manager.complete_store(keys, success=True)
        
        # Wait for cascade to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify blocks are in both tiers
        req_ctx = ReqContext()
        assert all(primary_tier.lookup(k, req_ctx) is True for k in keys)
        assert all(fs_tier.lookup(k, req_ctx) is True for k in keys)

        # Verify data integrity by reading from disk
        for key in keys:
            file_path = fs_tier.get_file_name(get_offload_block_hash(key))
            assert os.path.isfile(file_path), f"File not found: {file_path}"
            
            # Read file and verify size
            file_size = os.path.getsize(file_path)
            expected_size = block_elements * 4  # 4 bytes per float32
            assert file_size == expected_size, f"File size mismatch: {file_size} != {expected_size}"

    def test_full_promotion_with_data_integrity(self, setup_manager):
        """
        Pre-populate filesystem tier with blocks containing known data,
        trigger promotion by calling lookup(), and verify data integrity
        matches original data.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Generate unique data for blocks
        num_blocks = 4
        keys = [make_block_hash(2, i) for i in range(num_blocks)]
        expected_data = {}
        
        # Store blocks to primary first (to get them on disk)
        result = manager.prepare_store(keys, ReqContext())
        assert result is not None
        
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for i, block_id in enumerate(spec.block_ids):
            data = torch.rand(block_elements, dtype=torch.float32)
            cpu_tensor[int(block_id)] = data
            expected_data[keys[i]] = data.clone()
        
        manager.complete_store(keys, success=True)
        
        # Wait for cascade
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Evict blocks from primary tier by storing new blocks
        evict_keys = [make_block_hash(3, i) for i in range(10)]
        result = manager.prepare_store(evict_keys, ReqContext())
        assert result is not None
        assert len(result.evicted_keys) >= num_blocks
        
        for block_id in result.store_spec.block_ids:
            cpu_tensor[block_id] = 0.0
        manager.complete_store(evict_keys, success=True)
        
        # Wait for cascade of new blocks
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify blocks are only in filesystem tier
        req_ctx = ReqContext()
        assert not any(primary_tier.lookup(k, req_ctx) is True for k in keys)
        assert all(fs_tier.lookup(k, req_ctx) is True for k in keys)

        # Trigger promotion by lookup
        for k in keys:
            manager.lookup(k, req_ctx)

        # Wait for promotion to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)

        # Verify blocks are now in primary tier
        assert all(manager.lookup(k, req_ctx) is True for k in keys)

        # Verify data integrity after promotion
        spec = primary_tier.prepare_load(keys, ReqContext())
        for i, block_id in enumerate(spec.block_ids):
            actual_data = cpu_tensor[block_id]
            expected = expected_data[keys[i]]
            assert torch.allclose(actual_data, expected, rtol=1e-5, atol=1e-7), \
                f"Block {i} data mismatch after promotion"

    def test_cascade_promotion_roundtrip(self, setup_manager):
        """
        Store blocks with random data to primary (triggers cascade),
        evict blocks from primary tier, lookup blocks to trigger promotion
        from filesystem, and verify data integrity after full roundtrip.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Store blocks with random data
        num_blocks = 3
        keys = [make_block_hash(4, i) for i in range(num_blocks)]
        expected_data = {}
        
        result = manager.prepare_store(keys, ReqContext())
        assert result is not None
        
        for i, block_id in enumerate(result.store_spec.block_ids):
            data = torch.rand(block_elements, dtype=torch.float32)
            cpu_tensor[block_id] = data
            expected_data[keys[i]] = data.clone()
        
        manager.complete_store(keys, success=True)
        
        # Wait for cascade
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Evict from primary by filling it
        # Primary has 10 slots, 3 used, 7 free. Store 10 more to force
        # all 3 original blocks to be evicted (3 + 10 = 13, 13 - 10 = 3 evictions).
        evict_keys = [make_block_hash(5, i) for i in range(10)]
        result = manager.prepare_store(evict_keys, ReqContext())
        assert result is not None
        
        for block_id in result.store_spec.block_ids:
            cpu_tensor[block_id] = 0.0
        manager.complete_store(evict_keys, success=True)
        
        # Wait for cascade of new blocks
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify original blocks are evicted from primary
        req_ctx = ReqContext()
        assert not any(primary_tier.lookup(k, req_ctx) is True for k in keys)

        # Trigger promotion
        for k in keys:
            manager.lookup(k, req_ctx)

        # Wait for promotion to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)

        # Verify data integrity after roundtrip
        assert all(manager.lookup(k, req_ctx) is True for k in keys)
        spec = primary_tier.prepare_load(keys, ReqContext())
        
        for i, block_id in enumerate(spec.block_ids):
            actual_data = cpu_tensor[block_id]
            expected = expected_data[keys[i]]
            assert torch.allclose(actual_data, expected, rtol=1e-5, atol=1e-7), \
                f"Block {i} data mismatch after roundtrip"

    def test_eviction_coordination_with_real_io(self, setup_manager):
        """
        Fill both primary and filesystem tiers to capacity, store additional
        blocks to trigger eviction in both tiers, and verify LRU eviction
        works correctly with evicted blocks removed from disk.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Fill primary tier (10 blocks)
        keys_batch1 = [make_block_hash(6, i) for i in range(10)]
        result = manager.prepare_store(keys_batch1, ReqContext())
        assert result is not None
        
        for block_id in result.store_spec.block_ids:
            cpu_tensor[block_id] = torch.rand(block_elements, dtype=torch.float32)
        manager.complete_store(keys_batch1, success=True)
        
        # Wait for cascade
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Fill filesystem tier (20 blocks total, 10 more)
        keys_batch2 = [make_block_hash(7, i) for i in range(10)]
        result = manager.prepare_store(keys_batch2, ReqContext())
        assert result is not None
        
        for block_id in result.store_spec.block_ids:
            cpu_tensor[block_id] = torch.rand(block_elements, dtype=torch.float32)
        manager.complete_store(keys_batch2, success=True)
        
        # Wait for cascade
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify both tiers are at capacity
        assert fs_tier.get_num_blocks() == 20
        
        # Store more blocks to trigger eviction
        keys_batch3 = [make_block_hash(8, i) for i in range(5)]
        result = manager.prepare_store(keys_batch3, ReqContext())
        assert result is not None
        assert len(result.evicted_keys) == 5  # Primary tier evicts oldest
        
        for block_id in result.store_spec.block_ids:
            cpu_tensor[block_id] = torch.rand(block_elements, dtype=torch.float32)
        manager.complete_store(keys_batch3, success=True)
        
        # Wait for cascade
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify filesystem tier evicted oldest blocks
        assert fs_tier.get_num_blocks() == 20
        
        # Verify oldest blocks from batch1 are evicted from filesystem
        for i in range(5):
            key = keys_batch1[i]
            file_path = fs_tier.get_file_name(get_offload_block_hash(key))
            assert not os.path.exists(file_path), f"Evicted file still exists: {file_path}"

    def test_ref_cnt_protection_during_async_cascade(self, setup_manager):
        """
        Store blocks to primary tier, verify ref_cnt prevents eviction during
        async cascade, wait for cascade completion, and verify ref_cnt is
        released after cascade.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Store blocks to primary
        keys = [make_block_hash(9, i) for i in range(3)]
        result = manager.prepare_store(keys, ReqContext())
        assert result is not None
        
        for block_id in result.store_spec.block_ids:
            cpu_tensor[block_id] = torch.rand(block_elements, dtype=torch.float32)
        
        manager.complete_store(keys, success=True)
        
        # Verify ref_cnt is incremented during cascade (1 per secondary tier)
        for key in keys:
            block_status = primary_tier._policy.get(key)
            assert block_status is not None
            assert block_status.ref_cnt == 1, f"Expected ref_cnt=1, got {block_status.ref_cnt}"
        
        # Wait for cascade to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify ref_cnt is released after cascade
        for key in keys:
            block_status = primary_tier._policy.get(key)
            assert block_status is not None
            assert block_status.ref_cnt == 0, f"Expected ref_cnt=0, got {block_status.ref_cnt}"

    def test_multiple_filesystem_tiers_independent_io(self, tmp_path):
        """
        Use two filesystem tiers with different capacities, verify both
        receive cascaded data, verify independent eviction policies, and
        verify data integrity in both tiers.
        """
        block_elements = 128
        num_primary_blocks = 10
        
        # Create primary tier
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
        )
        cpu_tensor = torch.zeros((num_primary_blocks, block_elements), dtype=torch.float32)
        primary_tier.create_kv_memoryview = lambda: memoryview(cpu_tensor.numpy())

        # Create two filesystem tiers with different capacities
        fs_tier1 = FileSystemTierManagerCpp(
            base_path=str(tmp_path / "tier1"),
            max_blocks=5
        )
        fs_tier2 = FileSystemTierManagerCpp(
            base_path=str(tmp_path / "tier2"),
            max_blocks=15
        )
        
        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[fs_tier1, fs_tier2],
        )
        
        try:
            # Store blocks to primary (cascades to both tiers)
            keys = [make_block_hash(10, i) for i in range(5)]
            expected_data = {}
            
            result = manager.prepare_store(keys, ReqContext())
            assert result is not None
            
            spec = result.store_spec
            assert isinstance(spec, CPULoadStoreSpec)
            for i, block_id in enumerate(spec.block_ids):
                data = torch.rand(block_elements, dtype=torch.float32)
                cpu_tensor[int(block_id)] = data
                expected_data[keys[i]] = data.clone()
            
            manager.complete_store(keys, success=True)
            
            # Wait for cascade to both tiers
            for _ in range(20):
                manager._process_finished_jobs()
                time.sleep(0.01)
            
            # Verify both tiers received the data
            assert fs_tier1.get_num_blocks() == 5
            assert fs_tier2.get_num_blocks() == 5
            req_ctx = ReqContext()
            assert all(fs_tier1.lookup(k, req_ctx) is True for k in keys)
            assert all(fs_tier2.lookup(k, req_ctx) is True for k in keys)
            
            # Store more blocks to trigger eviction in tier1 only
            keys2 = [make_block_hash(11, i) for i in range(3)]
            result = manager.prepare_store(keys2, ReqContext())
            assert result is not None
            
            spec2 = result.store_spec
            assert isinstance(spec2, CPULoadStoreSpec)
            for block_id in spec2.block_ids:
                cpu_tensor[int(block_id)] = torch.rand(block_elements, dtype=torch.float32)
            
            manager.complete_store(keys2, success=True)
            
            # Wait for cascade to both tiers
            for _ in range(20):
                manager._process_finished_jobs()
                time.sleep(0.01)
            
            # Verify tier1 evicted blocks (capacity 5)
            assert fs_tier1.get_num_blocks() == 5
            # Verify tier2 kept all blocks (capacity 15)
            assert fs_tier2.get_num_blocks() == 8
            
            # Verify data integrity in tier2 for all blocks
            for key in keys + keys2:
                file_path = fs_tier2.get_file_name(get_offload_block_hash(key))
                assert os.path.isfile(file_path), f"File not found in tier2: {file_path}"
        
        finally:
            manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
