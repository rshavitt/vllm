# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FileSystemTierManager.

Unit tests (TestStorageTierState, TestStorageTierEviction,
TestStorageTierJobLifecycle, TestStorageTierErrorHandling) mock the C++
_kv_storage_ops extension and focus on Python-level state management: LRU
ordering, in-flight tracking, _evictable_count invariants, job lifecycle,
and error/edge-case paths.

Integration tests (TestStorageTierIO) require the real _kv_storage_ops
extension and exercise actual disk reads and writes.
"""

import contextlib
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Pre-install a lightweight stub so FileSystemTierManager can be imported even
# without a compiled _kv_storage_ops extension.  Unit tests patch the module-
# level names (cpp_submit_store_job / cpp_submit_load_job / cpp_get_finished_jobs)
# in place; integration tests detect whether the real extension is present.
# ---------------------------------------------------------------------------
if "vllm._kv_storage_ops" not in sys.modules:
    _stub = MagicMock()
    _stub.submit_store_job.return_value = None
    _stub.submit_load_job.return_value  = None
    _stub.get_finished_jobs.return_value = []
    sys.modules["vllm._kv_storage_ops"] = _stub

from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402
from vllm.v1.kv_offload.abstract import JobMetadata  # noqa: E402
from vllm.v1.kv_offload.mediums import CPUMemoryViewLoadStoreSpec  # noqa: E402
from vllm.v1.kv_offload.secondary_tiers.file_system import (  # noqa: E402
    FileSystemTierManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 16  # float32 elements per block row
_DTYPE = torch.float32


def bh(n: int) -> BlockHash:
    """Return a deterministic BlockHash from an integer."""
    return BlockHash(n.to_bytes(8, "big"))


def make_spec(
    block_ids: list[int],
    num_total_blocks: int = 32,
) -> CPUMemoryViewLoadStoreSpec:
    """
    Create a CPUMemoryViewLoadStoreSpec backed by a single CPU tensor.
    Each row is one block; stride = _BLOCK_ELEMENTS * 4 bytes.
    """
    tensor = torch.zeros((num_total_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
    return CPUMemoryViewLoadStoreSpec(block_ids, tensor)


def make_job(
    job_id: int,
    hashes: list[BlockHash],
    block_ids: list[int] | None = None,
    num_total_blocks: int = 32,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(hashes)))
    spec = make_spec(block_ids, num_total_blocks=num_total_blocks)
    return JobMetadata(job_id=job_id, block_hashes=hashes, spec=spec)


def drain(tier: FileSystemTierManager, max_rounds: int = 20) -> list:
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


def evictable_count_expected(tier: FileSystemTierManager) -> int:
    """Recompute _evictable_count from first principles for assertion."""
    return len(set(tier._blocks) - set(tier._in_flight))


# ---------------------------------------------------------------------------
# _SyncMockCpp — synchronous mock for the _kv_storage_ops C++ functions
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
        base = "vllm.v1.kv_offload.secondary_tiers.file_system"
        with (
            patch(f"{base}.cpp_submit_store_job", new=self.submit_store_job),
            patch(f"{base}.cpp_submit_load_job",  new=self.submit_load_job),
            patch(f"{base}.cpp_get_finished_jobs", new=self.get_finished_jobs),
        ):
            yield self


# ---------------------------------------------------------------------------
# State tests — mocked I/O, focus on lookup / in-flight tracking
# ---------------------------------------------------------------------------

class TestStorageTierState:

    @pytest.fixture(autouse=True)
    def _patch_io(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManager(
                base_path=str(tmp_path), max_blocks=10
            )
            yield

    def test_get_tier_name(self):
        t = FileSystemTierManager.__new__(FileSystemTierManager)
        t._tier_name = "MyTier"
        assert t.get_tier_name() == "MyTier"

    def test_initial_state_empty(self):
        assert self.tier.get_num_blocks() == 0
        assert self.tier.get_num_in_flight() == 0
        assert self.tier._evictable_count == 0

    def test_lookup_empty_tier(self):
        assert self.tier.lookup([bh(1), bh(2)]) == 0

    def test_lookup_all_present(self):
        blocks = [bh(i) for i in range(3)]
        for b in blocks:
            self.tier._blocks[b] = True
        self.tier._evictable_count = 3
        assert self.tier.lookup(blocks) == 3

    def test_lookup_partial_hit_stops_at_first_miss(self):
        blocks = [bh(i) for i in range(4)]
        # Only first two present
        self.tier._blocks[blocks[0]] = True
        self.tier._blocks[blocks[1]] = True
        self.tier._evictable_count = 2
        assert self.tier.lookup(blocks) == 2

    def test_lookup_in_flight_returns_none(self):
        blocks = [bh(i) for i in range(3)]
        # First block present, second block in-flight.
        # lookup() checks _in_flight before _blocks, so it must reach
        # blocks[1] — that only happens once blocks[0] passes both checks.
        self.tier._blocks[blocks[0]] = True
        self.tier._evictable_count = 1
        self.tier._in_flight[blocks[1]] = 99
        assert self.tier.lookup(blocks) is None

    def test_lookup_none_when_first_block_in_flight(self):
        blocks = [bh(i) for i in range(3)]
        self.tier._in_flight[blocks[0]] = 1
        assert self.tier.lookup(blocks) is None

    def test_get_file_name_structure(self, tmp_path):
        tier = FileSystemTierManager(base_path="/kvcache", max_blocks=10)
        # bh(0) → int 0 → hex "0000000000000000"
        path = tier.get_file_name(bh(0))
        assert path == "/kvcache/000/00/0000000000000000.bin"

    def test_get_file_name_consistent_for_same_hash(self, tmp_path):
        tier = FileSystemTierManager(base_path="/kvcache", max_blocks=10)
        h = bh(12345)
        assert tier.get_file_name(h) == tier.get_file_name(h)

    def test_get_file_name_accepts_int(self, tmp_path):
        tier = FileSystemTierManager(base_path="/base", max_blocks=10)
        # BlockHash(n.to_bytes(8)) → int.from_bytes → same hex as passing int directly
        path_via_bytes = tier.get_file_name(bh(42))
        path_via_int = tier.get_file_name(42)
        assert path_via_bytes == path_via_int


# ---------------------------------------------------------------------------
# Eviction tests
# ---------------------------------------------------------------------------

class TestStorageTierEviction:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManager(
                base_path=str(tmp_path), max_blocks=5
            )
            yield

    def _fill(self, n: int, start: int = 0):
        """Seed _blocks with n entries (bypassing I/O) from hash start."""
        for i in range(start, start + n):
            self.tier._blocks[bh(i)] = True
        self.tier._evictable_count = len(self.tier._blocks)

    def test_eviction_removes_oldest_first(self):
        self._fill(5)  # blocks 0-4, oldest = 0
        # Store one new block; must evict bh(0)
        job = make_job(1, [bh(10)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        assert bh(10) in self.tier._blocks
        assert bh(0) not in self.tier._blocks
        # Remaining original blocks still present
        for i in range(1, 5):
            assert bh(i) in self.tier._blocks

    def test_eviction_respects_in_flight(self):
        self._fill(5)  # blocks 0-4
        # Mark block 0 (oldest) as in-flight
        self.tier._in_flight[bh(0)] = 99
        self.tier._evictable_count -= 1  # it's in-flight now

        # Now try to add one more block; must skip bh(0) and evict bh(1)
        job = make_job(1, [bh(10)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        assert bh(10) in self.tier._blocks
        assert bh(0) in self.tier._blocks   # protected by in-flight
        assert bh(1) not in self.tier._blocks  # oldest evictable

    def test_eviction_skips_protected_batch_blocks(self):
        self._fill(5)  # 0-4 oldest to newest
        # Store [bh(0), bh(10)]: bh(0) already on disk so filtered out;
        # but bh(0) appears in all_hashes → protected set.
        # Need to evict 1 to make room for bh(10).
        # bh(0) is protected; bh(1) should be evicted.
        job = make_job(1, [bh(0), bh(10)], [0, 1])
        self.tier.submit_store(job)
        drain(self.tier)

        assert bh(10) in self.tier._blocks
        assert bh(0) in self.tier._blocks   # in all_hashes → protected
        assert bh(1) not in self.tier._blocks

    def test_eviction_fails_insufficient_evictable(self):
        """All blocks in-flight → _evictable_count=0 → drop job with warning."""
        self._fill(5)
        for i in range(5):
            self.tier._in_flight[bh(i)] = 99
        self.tier._evictable_count = 0

        with patch("vllm.v1.kv_offload.secondary_tiers.file_system.logger") as mock_log:
            self.tier.submit_store(make_job(1, [bh(10)], [0]))

        assert bh(10) not in self.tier._blocks
        mock_log.warning.assert_called_once()
        assert "insufficient" in mock_log.warning.call_args[0][0]

    def test_eviction_fails_protected_overlap(self, caplog):
        """
        _evictable_count >= needed but all evictable candidates are in
        the protected set → scan exhausts without finding enough → warning.
        """
        # 3 blocks on disk, 2 needed for eviction, but both evictable ones
        # are in the current batch (protected).
        self._fill(3)  # bh(0), bh(1), bh(2)
        # Store bh(3) and bh(4): need to evict 2, but bh(0),bh(1),bh(2) are
        # all in all_hashes → all protected → can't find candidates.
        job = make_job(1, [bh(0), bh(1), bh(2), bh(3), bh(4)], [0, 1, 2, 3, 4])
        with caplog.at_level("WARNING"):
            self.tier.submit_store(job)
        # bh(3) and bh(4) are new blocks but eviction failed
        assert bh(3) not in self.tier._blocks
        assert bh(4) not in self.tier._blocks

    def test_evictable_count_after_eviction(self):
        self._fill(5)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        job = make_job(1, [bh(10)], [0])
        self.tier.submit_store(job)
        drain(self.tier)

        assert self.tier._evictable_count == evictable_count_expected(self.tier)

    def test_touch_moves_to_end_of_lru(self):
        self._fill(3)  # insertion order: bh(0), bh(1), bh(2)
        self.tier.touch([bh(0)])  # bh(0) now most recent
        lru_order = list(self.tier._blocks.keys())
        assert lru_order[-1] == bh(0)
        assert lru_order[0] == bh(1)  # bh(1) is now oldest


# ---------------------------------------------------------------------------
# Job lifecycle tests
# ---------------------------------------------------------------------------

class TestStorageTierJobLifecycle:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self._mock = _SyncMockCpp()
        with self._mock.patch_ctx():
            self.tier = FileSystemTierManager(
                base_path=str(tmp_path), max_blocks=20
            )
            yield

    def test_submit_store_goes_to_active_jobs(self):
        job = make_job(1, [bh(0), bh(1)], [0, 1])
        self.tier.submit_store(job)

        assert len(self.tier._active_jobs) == 1

    def test_submit_load_goes_to_futures_immediately(self):
        # Seed blocks as already on disk.
        self.tier._blocks[bh(0)] = True
        self.tier._blocks[bh(1)] = True
        self.tier._evictable_count = 2

        job = make_job(1, [bh(0), bh(1)], [0, 1])
        self.tier.submit_load(job)

        assert len(self.tier._active_jobs) == 1

    def test_store_job_completes_and_adds_to_blocks(self):
        blocks = [bh(0), bh(1)]
        job = make_job(1, blocks, [0, 1])
        self.tier.submit_store(job)
        results = drain(self.tier)

        assert len(results) == 1
        assert results[0].job_id == 1
        assert results[0].success is True
        assert all(b in self.tier._blocks for b in blocks)
        assert self.tier.get_num_in_flight() == 0

    def test_store_job_updates_evictable_count(self):
        blocks = [bh(0), bh(1)]
        job = make_job(1, blocks, [0, 1])
        self.tier.submit_store(job)
        drain(self.tier)

        assert self.tier._evictable_count == evictable_count_expected(self.tier)
        assert self.tier._evictable_count == 2

    def test_load_job_completes_and_restores_evictable_count(self):
        self.tier._blocks[bh(0)] = True
        self.tier._blocks[bh(1)] = True
        self.tier._evictable_count = 2

        job = make_job(1, [bh(0), bh(1)], [0, 1])
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
        job = make_job(1, [bh(0)], [0])
        self.tier.submit_store(job)

        assert 1 in self.tier._active_jobs
        active = self.tier._active_jobs[1]
        assert isinstance(active.buffer, memoryview)

    def test_duplicate_store_skipped(self):
        """Blocks already on disk are not re-stored."""
        self.tier._blocks[bh(0)] = True
        self.tier._evictable_count = 1

        self.tier.submit_store(make_job(1, [bh(0)], [0]))
        # All blocks already on disk → filtered out, no job submitted
        assert 1 not in self.tier._active_jobs

    def test_in_flight_store_not_duplicated(self):
        """
        A second submit_store for a block already in-flight must be dropped.
        Without the fix, the second call overwrites _in_flight[bh] with the
        new job_id, so get_finished() for the first job deletes a key it no
        longer owns, corrupting state.
        """
        self.tier.submit_store(make_job(1, [bh(0)], [0]))
        assert 1 in self.tier._active_jobs
        assert self.tier._in_flight[bh(0)] == 1

        # Second store for the same block while the first is still in-flight.
        self.tier.submit_store(make_job(2, [bh(0)], [0]))
        # Second job silently dropped; _in_flight must still point to job 1.
        assert 2 not in self.tier._active_jobs
        assert self.tier._in_flight[bh(0)] == 1

    def test_failed_store_does_not_add_to_blocks(self):
        failing_mock = _SyncMockCpp(success=False)
        with failing_mock.patch_ctx():
            tier = FileSystemTierManager(base_path="/tmp", max_blocks=10)
            tier.submit_store(make_job(1, [bh(0)], [0]))
            results = drain(tier)

        assert results[0].success is False
        assert bh(0) not in tier._blocks

    def test_failed_store_evictable_count_unchanged(self):
        failing_mock = _SyncMockCpp(success=False)
        with failing_mock.patch_ctx():
            tier = FileSystemTierManager(base_path="/tmp", max_blocks=10)
            tier.submit_store(make_job(1, [bh(0)], [0]))
            drain(tier)

        assert tier._evictable_count == 0

    def test_multiple_independent_jobs(self):
        blocks_a = [bh(0), bh(1)]
        blocks_b = [bh(2), bh(3)]
        self.tier.submit_store(make_job(1, blocks_a, [0, 1]))
        self.tier.submit_store(make_job(2, blocks_b, [2, 3]))
        results = drain(self.tier)

        job_ids = {r.job_id for r in results}
        assert job_ids == {1, 2}
        assert all(b in self.tier._blocks for b in blocks_a + blocks_b)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestStorageTierErrorHandling:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManager(
                base_path=str(tmp_path), max_blocks=10
            )
            yield

    def test_submit_load_missing_block_logs_warning(self):
        with patch("vllm.v1.kv_offload.secondary_tiers.file_system.logger") as mock_log:
            self.tier.submit_load(make_job(1, [bh(42)], [0]))

        mock_log.warning.assert_called_once()
        assert "not found on disk" in mock_log.warning.call_args[0][0]

    def test_submit_load_missing_block_no_state_change(self):
        before_inflight = dict(self.tier._in_flight)
        before_evictable = self.tier._evictable_count
        self.tier.submit_load(make_job(1, [bh(42)], [0]))
        assert self.tier._in_flight == before_inflight
        assert self.tier._evictable_count == before_evictable
        assert 1 not in self.tier._active_jobs

    def test_submit_store_cpp_failure_rolls_back(self):
        """If cpp_submit_store_job raises, _in_flight must be cleaned up."""
        base = "vllm.v1.kv_offload.secondary_tiers.file_system"
        with patch(f"{base}.cpp_submit_store_job",
                   side_effect=RuntimeError("pool error")):
            with pytest.raises(RuntimeError):
                self.tier.submit_store(make_job(1, [bh(0)], [0]))

        assert bh(0) not in self.tier._in_flight
        assert 1 not in self.tier._active_jobs

    def test_submit_load_cpp_failure_rolls_back(self):
        """
        If cpp_submit_load_job raises, all state mutations must be rolled back.
        """
        self.tier._blocks[bh(0)] = True
        self.tier._evictable_count = 1

        base = "vllm.v1.kv_offload.secondary_tiers.file_system"
        with patch(f"{base}.cpp_submit_load_job",
                   side_effect=RuntimeError("pool error")):
            with pytest.raises(RuntimeError):
                self.tier.submit_load(make_job(1, [bh(0)], [0]))

        # State rolled back
        assert bh(0) not in self.tier._in_flight
        assert self.tier._evictable_count == 1
        assert 1 not in self.tier._active_jobs

    def test_get_finished_failed_job_cleans_up_state(self, caplog):
        """A job finishing with success=False must clean up state correctly."""
        self.tier._blocks[bh(0)] = True
        self.tier._evictable_count = 1

        # Submit a load job and immediately complete it as a failure.
        failing_mock = _SyncMockCpp(success=False)
        base = "vllm.v1.kv_offload.secondary_tiers.file_system"
        with (
            patch(f"{base}.cpp_submit_load_job",  new=failing_mock.submit_load_job),
            patch(f"{base}.cpp_get_finished_jobs", new=failing_mock.get_finished_jobs),
        ):
            self.tier.submit_load(make_job(7, [bh(0)], [0]))
            results = list(self.tier.get_finished())

        assert len(results) == 1
        assert results[0].job_id == 7
        assert results[0].success is False
        # State fully cleaned up
        assert 7 not in self.tier._active_jobs
        assert bh(0) not in self.tier._in_flight
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

    def test_evictable_count_invariant_after_mixed_operations(self):
        """
        _evictable_count must equal len(_blocks) - (in_flight ∩ _blocks)
        after a sequence of store, evict, load, and failure operations.
        """
        # Store 3 blocks
        self.tier.submit_store(make_job(1, [bh(i) for i in range(3)], [0, 1, 2]))
        drain(self.tier)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        # Start loading bh(0) and bh(1)
        job = make_job(2, [bh(0), bh(1)], [0, 1])
        self.tier.submit_load(job)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        # Complete the load
        drain(self.tier)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)

        # Store 3 more (triggers eviction of bh(2) since cap=10, 3+3=6 ≤ 10, no eviction needed)
        self.tier.submit_store(
            make_job(3, [bh(i) for i in range(10, 13)], [0, 1, 2])
        )
        drain(self.tier)
        assert self.tier._evictable_count == evictable_count_expected(self.tier)


# ---------------------------------------------------------------------------
# Integration tests — require real _kv_storage_ops extension
# ---------------------------------------------------------------------------

def _is_real_extension() -> bool:
    """True if the real C extension (not our MagicMock stub) is loaded."""
    mod = sys.modules.get("vllm._kv_storage_ops")
    return mod is not None and not isinstance(mod, MagicMock)


pytestmark_io = pytest.mark.skipif(
    not _is_real_extension(),
    reason="_kv_storage_ops extension not built; skipping I/O integration tests",
)


@pytestmark_io
class TestStorageTierIO:
    """Exercises actual pread/pwrite disk I/O via the C++ extension."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.base = str(tmp_path)
        self.block_elements = 16
        self.tier = FileSystemTierManager(
            base_path=self.base, max_blocks=50
        )

    def _make_tensor(self, num_blocks: int, fill_value: float = 0.0):
        t = torch.full(
            (num_blocks, self.block_elements), fill_value, dtype=torch.float32
        )
        return t

    def test_store_creates_file_on_disk(self, tmp_path):
        import os

        tensor = self._make_tensor(4)
        spec = CPUMemoryViewLoadStoreSpec([0], tensor)
        job = JobMetadata(job_id=1, block_hashes=[bh(1)], spec=spec)
        self.tier.submit_store(job)
        drain(self.tier)

        expected_path = self.tier.get_file_name(bh(1))
        assert os.path.isfile(expected_path)

    def test_data_roundtrip_single_block(self, tmp_path):
        import os

        num_blocks = 4
        block_size = self.block_elements * 4  # float32

        # Write tensor with known values
        src_tensor = self._make_tensor(num_blocks)
        src_tensor[0] = torch.arange(self.block_elements, dtype=torch.float32)

        spec_store = CPUMemoryViewLoadStoreSpec([0], src_tensor, readonly=True)
        self.tier.submit_store(
            JobMetadata(job_id=1, block_hashes=[bh(7)], spec=spec_store)
        )
        drain(self.tier)

        # Read back into a fresh tensor
        dst_tensor = self._make_tensor(num_blocks)
        spec_load = CPUMemoryViewLoadStoreSpec([1], dst_tensor)
        self.tier.submit_load(
            JobMetadata(job_id=2, block_hashes=[bh(7)], spec=spec_load)
        )
        drain(self.tier)

        # Block 1 of dst should equal block 0 of src
        assert torch.equal(dst_tensor[1], src_tensor[0])

    def test_data_roundtrip_multiple_blocks(self):
        num_blocks = 8
        src_tensor = self._make_tensor(num_blocks)
        hashes = [bh(i + 100) for i in range(4)]
        for idx, h in enumerate(hashes):
            src_tensor[idx] = float(idx + 1)

        spec_store = CPUMemoryViewLoadStoreSpec(
            list(range(4)), src_tensor, readonly=True
        )
        self.tier.submit_store(
            JobMetadata(job_id=1, block_hashes=hashes, spec=spec_store)
        )
        drain(self.tier)

        dst_tensor = self._make_tensor(num_blocks)
        spec_load = CPUMemoryViewLoadStoreSpec(list(range(4, 8)), dst_tensor)
        self.tier.submit_load(
            JobMetadata(job_id=2, block_hashes=hashes, spec=spec_load)
        )
        drain(self.tier)

        for i in range(4):
            assert torch.equal(dst_tensor[4 + i], src_tensor[i])

    def test_file_path_is_deterministic(self):
        h = bh(9999)
        path1 = self.tier.get_file_name(h)
        path2 = self.tier.get_file_name(h)
        assert path1 == path2

