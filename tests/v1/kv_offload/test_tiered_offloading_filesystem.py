# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TiersOffloadingManager tests using one DummySecondaryTier and one
FileSystemTierManager as the secondary tiers.

The C++ _kv_storage_ops extension is replaced with _SyncMockCpp so jobs
complete synchronously without real disk I/O, keeping the tests fast and
self-contained.
"""

import contextlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Pre-install a lightweight stub so FileSystemTierManager can be imported even
# without a compiled _kv_storage_ops extension.
# ---------------------------------------------------------------------------
if "vllm._kv_storage_ops" not in sys.modules:
    _stub = MagicMock()
    _stub.submit_store_job.return_value = None
    _stub.submit_load_job.return_value  = None
    _stub.get_finished_jobs.return_value = []
    sys.modules["vllm._kv_storage_ops"] = _stub

from vllm.v1.core.kv_cache_utils import BlockHash  # noqa: E402
from vllm.v1.kv_offload.secondary_tiers.dummy import DummySecondaryTier  # noqa: E402
from vllm.v1.kv_offload.secondary_tiers.file_system import (  # noqa: E402
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiered import CPUPrimaryTierOffloadingManager  # noqa: E402
from vllm.v1.kv_offload.tiered_manager import TiersOffloadingManager  # noqa: E402


def make_block_hash(req_id: int, block_idx: int) -> BlockHash:
    return BlockHash(f"{req_id}:{block_idx}".encode())


# ---------------------------------------------------------------------------
# _SyncMockCpp — synchronous mock for the _kv_storage_ops C++ functions
# ---------------------------------------------------------------------------

class _SyncMockCpp:
    """Jobs complete immediately on submit."""

    def __init__(self, success: bool = True):
        self._pending: list[tuple[int, bool]] = []
        self.success = success

    def submit_store_job(self, job_id, *args):
        self._pending.append((job_id, self.success))

    def submit_load_job(self, job_id, *args):
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
# TestFileSystemTierDirect — mirrors TestDummySecondaryTier from
# test_tiered_offloading.py, exercising FileSystemTierManager directly.
# ---------------------------------------------------------------------------

class TestFileSystemTierDirect:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.tier = FileSystemTierManager(
                base_path=str(tmp_path), max_blocks=10
            )
            yield

    def test_basic_store_and_lookup(self):
        blocks = [make_block_hash(1, i) for i in range(3)]
        assert self.tier.lookup(blocks) == 0

        # Seed blocks directly (bypass I/O)
        self.tier._blocks[blocks[0]] = True
        self.tier._blocks[blocks[1]] = True
        self.tier._evictable_count = 2

        assert self.tier.lookup(blocks) == 2
        assert self.tier.lookup([blocks[2]]) == 0

    def test_in_flight_blocks_return_none(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.tier._in_flight[blocks[0]] = 1

        assert self.tier.lookup(blocks) is None

    def test_lru_eviction(self, tmp_path):
        mock_tensor = torch.zeros((4, 16), dtype=torch.float32)

        # Use a small-capacity tier so adding a 4th block triggers eviction.
        tier = FileSystemTierManager(base_path=str(tmp_path / "lru"), max_blocks=3)

        blocks = [make_block_hash(1, i) for i in range(3)]
        for b in blocks:
            tier._blocks[b] = True
        tier._evictable_count = 3

        assert tier.get_num_blocks() == 3

        # Touch blocks[0] so it becomes most-recently-used
        tier.touch([blocks[0]])

        from vllm.v1.kv_offload.abstract import JobMetadata
        from vllm.v1.kv_offload.mediums import CPUMemoryViewLoadStoreSpec

        new_block = make_block_hash(1, 3)
        tier.submit_store(
            JobMetadata(
                job_id=1,
                block_hashes=[new_block],
                spec=CPUMemoryViewLoadStoreSpec([0], mock_tensor),
            )
        )
        list(tier.get_finished())

        assert new_block in tier._blocks
        assert blocks[1] not in tier._blocks  # LRU victim (oldest after touch)
        assert blocks[0] in tier._blocks
        assert blocks[2] in tier._blocks

    def test_jobs_complete_via_get_finished(self):
        """Store jobs submitted to C++ complete when get_finished() is called."""
        from vllm.v1.kv_offload.abstract import JobMetadata
        from vllm.v1.kv_offload.mediums import CPUMemoryViewLoadStoreSpec

        mock_tensor = torch.zeros((10, 16), dtype=torch.float32)
        blocks = [make_block_hash(1, i) for i in range(2)]

        self.tier.submit_store(
            JobMetadata(
                job_id=1,
                block_hashes=blocks,
                spec=CPUMemoryViewLoadStoreSpec([0, 1], mock_tensor),
            )
        )

        # In-flight while job is pending
        assert self.tier.get_num_in_flight() == 2
        assert self.tier.get_num_blocks() == 0

        completed = list(self.tier.get_finished())
        assert len(completed) == 1
        assert completed[0].job_id == 1
        assert completed[0].success is True

        assert self.tier.get_num_blocks() == 2
        assert self.tier.get_num_in_flight() == 0


# ---------------------------------------------------------------------------
# Tests: one DummySecondaryTier (tier1) + one FileSystemTierManager (tier2)
# ---------------------------------------------------------------------------

class TestTiersOffloadingManagerMixed:

    @pytest.fixture(autouse=True)
    def manager_setup(self, tmp_path):
        mock = _SyncMockCpp()
        with mock.patch_ctx():
            self.primary_tier = CPUPrimaryTierOffloadingManager(
                block_size=16, num_blocks=5
            )
            mock_cpu_tensor = torch.zeros((5, 16), dtype=torch.float32)
            self.primary_tier.get_primary_kv_tensors = lambda: mock_cpu_tensor

            # tier1: in-memory dummy tier
            self.secondary_tier1 = DummySecondaryTier(
                tier_name="Dummy", max_blocks=10
            )
            # tier2: filesystem tier (C++ mocked)
            self.secondary_tier2 = FileSystemTierManager(
                base_path=str(tmp_path / "fs"), max_blocks=10
            )

            self.manager = TiersOffloadingManager(
                primary_tier=self.primary_tier,
                secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
            )
            yield

    def test_basic_store_to_primary(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        result = self.manager.prepare_store(blocks)
        assert result is not None
        assert len(result.block_hashes_to_store) == 3

        self.manager.complete_store(blocks, success=True)

        assert self.primary_tier.lookup(blocks) == 3

    def test_cascade_to_both_secondary_tiers(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)
        self.manager._process_finished_jobs()

        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3
        assert self.secondary_tier1.lookup(blocks) == 3
        assert self.secondary_tier2.lookup(blocks) == 3

    def test_ref_cnt_protection_during_cascade(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)

        # ref_cnt == 2: one hold per secondary tier
        for bh in blocks:
            assert self.primary_tier._policy.get(bh).ref_cnt == 2

        self.manager._process_finished_jobs()

        for bh in blocks:
            assert self.primary_tier._policy.get(bh).ref_cnt == 0

    def test_lookup_from_primary(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)

        assert self.manager.lookup(blocks) == 3

    def test_promotion_from_dummy_tier(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Seed dummy tier directly
        for bh in blocks:
            self.secondary_tier1.blocks[bh] = True

        result = self.manager.lookup(blocks)
        assert result is None  # retry later

        self.manager._process_finished_jobs()

        assert self.primary_tier.lookup(blocks) == 3
        assert self.manager.lookup(blocks) == 3

    def test_promotion_from_filesystem_tier(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Seed filesystem tier directly
        for bh in blocks:
            self.secondary_tier2._blocks[bh] = True
        self.secondary_tier2._evictable_count = len(blocks)

        result = self.manager.lookup(blocks)
        assert result is None  # retry later

        self.manager._process_finished_jobs()

        assert self.primary_tier.lookup(blocks) == 3
        assert self.manager.lookup(blocks) == 3

    def test_partial_lookup(self):
        blocks = [make_block_hash(1, i) for i in range(5)]

        self.manager.prepare_store(blocks[:3])
        self.manager.complete_store(blocks[:3], success=True)

        assert self.manager.lookup(blocks) == 3

    def test_eviction_in_primary_tier(self):
        blocks = [make_block_hash(1, i) for i in range(5)]
        result = self.manager.prepare_store(blocks)
        assert result is not None
        self.manager.complete_store(blocks, success=True)
        self.manager._process_finished_jobs()

        more_blocks = [make_block_hash(2, i) for i in range(2)]
        result = self.manager.prepare_store(more_blocks)

        assert result is not None
        assert len(result.block_hashes_evicted) == 2
        assert len(result.block_hashes_to_store) == 2

    def test_touch_propagates_to_all_tiers(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)
        self.manager._process_finished_jobs()

        self.manager.touch(blocks)

        primary_keys = list(self.primary_tier._policy.blocks.keys())
        assert primary_keys[-3:] == list(reversed(blocks))

        assert list(self.secondary_tier1.blocks.keys())[-3:] == list(reversed(blocks))
        assert list(self.secondary_tier2._blocks.keys())[-3:] == list(reversed(blocks))

    def test_failed_store_no_cascade(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=False)
        self.manager._process_finished_jobs()

        assert self.secondary_tier1.get_num_blocks() == 0
        assert self.secondary_tier2.get_num_blocks() == 0

    def test_multiple_secondary_tiers_independent_eviction(self, tmp_path):
        """Test that secondary tiers manage their own evictions independently."""
        # Create tiers with different capacities
        small_tier = DummySecondaryTier(
            tier_name="SmallDummy", max_blocks=5, simulate_async=False
        )
        large_tier = FileSystemTierManager(
            base_path=str(tmp_path / "large_fs"), max_blocks=10
        )

        # Create a fresh primary tier for this test
        primary_tier = CPUPrimaryTierOffloadingManager(block_size=16, num_blocks=10)

        # Mock get_primary_kv_tensors to return test tensor
        mock_cpu_tensor = torch.zeros((10, 16), dtype=torch.float32)
        primary_tier.get_primary_kv_tensors = lambda: mock_cpu_tensor

        manager = TiersOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[small_tier, large_tier],
        )

        # First, store 5 blocks to fill the small tier
        blocks1 = [make_block_hash(1, i) for i in range(5)]
        result = manager.prepare_store(blocks1)
        assert result is not None
        manager.complete_store(blocks1, success=True)
        manager._process_finished_jobs()

        # Both tiers should have 5 blocks
        assert small_tier.get_num_blocks() == 5
        assert large_tier.get_num_blocks() == 5

        # Now store 3 more blocks - small tier should evict 3 blocks
        blocks2 = [make_block_hash(2, i) for i in range(3)]
        result = manager.prepare_store(blocks2)
        assert result is not None
        manager.complete_store(blocks2, success=True)
        manager._process_finished_jobs()

        # Small tier should still have 5 blocks (evicted 3, added 3)
        assert small_tier.get_num_blocks() == 5

        # Large tier should have all 8 blocks
        assert large_tier.get_num_blocks() == 8

    def test_prepare_store_processes_finished_jobs_first(self):
        blocks = [make_block_hash(1, i) for i in range(3)]

        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)

        for bh in blocks:
            assert self.primary_tier._policy.get(bh).ref_cnt == 2

        self.manager.prepare_store([make_block_hash(2, i) for i in range(2)])

        for bh in blocks:
            assert self.primary_tier._policy.get(bh).ref_cnt == 0


# ---------------------------------------------------------------------------
# Baseline: no secondary tiers
# ---------------------------------------------------------------------------

class TestTiersOffloadingWithoutSecondaryTiers:

    def test_works_without_secondary_tiers(self):
        primary_tier = CPUPrimaryTierOffloadingManager(block_size=16, num_blocks=5)
        primary_tier.get_primary_kv_tensors = lambda: torch.zeros((5, 16))

        manager = TiersOffloadingManager(primary_tier=primary_tier, secondary_tiers=[])

        blocks = [make_block_hash(1, i) for i in range(3)]
        manager.prepare_store(blocks)
        manager.complete_store(blocks, success=True)

        assert manager.lookup(blocks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
