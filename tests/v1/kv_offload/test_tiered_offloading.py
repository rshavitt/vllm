# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TieredOffloadingManager and DummySecondaryTier.

These tests verify:
1. Basic tiered offloading operations (store, load, lookup)
2. Cascade behavior (blocks stored to all secondary tiers)
3. Promotion behavior (blocks loaded from secondary to primary to GPU)
4. ref_cnt management (blocks protected during async transfers)
5. Eviction coordination between tiers
"""

import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import TransferDirection
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.secondary_tiers.dummy import DummySecondaryTier
from vllm.v1.kv_offload.tiered_manager import TieredOffloadingManager


def make_block_hash(req_id: int, block_idx: int) -> BlockHash:
    """Helper to create block hashes for testing."""
    return BlockHash(f"{req_id}:{block_idx}".encode())


class TestDummySecondaryTier:
    """Tests for DummySecondaryTier implementation."""

    def test_basic_store_and_lookup(self):
        """Test basic store and lookup operations."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=10)

        # Initially empty
        blocks = [make_block_hash(1, i) for i in range(3)]
        assert tier.lookup(blocks) == 0

        # Store blocks (simulate with direct insertion for testing)
        tier.blocks[blocks[0]] = True
        tier.blocks[blocks[1]] = True

        # Lookup should find 2 blocks
        assert tier.lookup(blocks) == 2

        # Third block not present
        assert tier.lookup([blocks[2]]) == 0

    def test_in_flight_blocks_return_none(self):
        """Test that in-flight blocks cause lookup to return None."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=10)

        blocks = [make_block_hash(1, i) for i in range(3)]

        # Mark first block as in-flight
        tier.in_flight[blocks[0]] = 1

        # Lookup should return None (retry later)
        assert tier.lookup(blocks) is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=3)

        # Fill tier to capacity
        blocks = [make_block_hash(1, i) for i in range(3)]
        for block in blocks:
            tier.blocks[block] = True

        assert tier.get_num_blocks() == 3

        # Touch first block (make it most recently used)
        tier.touch([blocks[0]])

        # Store new block should evict blocks[1] (least recently used)
        new_block = make_block_hash(1, 3)
        from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

        result = tier.submit_store(
            job_id=1,
            block_hashes=[new_block],
            primary_load_spec=CPULoadStoreSpec([0]),
        )

        assert result is not None
        assert len(result.block_hashes_evicted) == 1
        assert result.block_hashes_evicted[0] == blocks[1]

        # Complete the job
        tier.get_finished()

        # Verify new block is stored and old block is evicted
        assert new_block in tier.blocks
        assert blocks[1] not in tier.blocks

    def test_async_simulation(self):
        """Test simulated async behavior."""
        tier = DummySecondaryTier(tier_name="Test", max_blocks=10, simulate_async=True)

        blocks = [make_block_hash(1, i) for i in range(2)]
        from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

        # Submit store job
        tier.submit_store(
            job_id=1, block_hashes=blocks, primary_load_spec=CPULoadStoreSpec([0, 1])
        )

        # Blocks should be in-flight
        assert tier.get_num_in_flight() == 2
        assert tier.get_num_blocks() == 0

        # First get_finished() should complete the job
        completed = list(tier.get_finished())
        assert len(completed) == 1
        assert completed[0].job_id == 1
        assert completed[0].direction == TransferDirection.PRIMARY_TO_SECONDARY
        assert completed[0].success is True

        # Blocks should now be stored
        assert tier.get_num_blocks() == 2
        assert tier.get_num_in_flight() == 0


class TestTieredOffloadingManager:
    """Tests for TieredOffloadingManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create primary tier (CPU-based)
        self.cpu_backend = CPUBackend(block_size=16, num_blocks=5)
        self.primary_tier = LRUOffloadingManager(self.cpu_backend)

        # Create secondary tiers
        self.secondary_tier1 = DummySecondaryTier(tier_name="Storage", max_blocks=10)
        self.secondary_tier2 = DummySecondaryTier(tier_name="Network", max_blocks=10)

        # Create tiered manager
        self.manager = TieredOffloadingManager(
            primary_tier=self.primary_tier,
            secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
        )

    def test_basic_store_to_primary(self):
        """Test basic store operation to primary tier."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Prepare store
        result = self.manager.prepare_store(blocks)
        assert result is not None
        assert len(result.block_hashes_to_store) == 3

        # Complete store
        self.manager.complete_store(blocks, success=True)

        # Blocks should be in primary tier
        assert self.primary_tier.lookup(blocks) == 3

    def test_cascade_to_all_secondary_tiers(self):
        """Test that blocks are cascaded to ALL secondary tiers."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Store to primary
        result = self.manager.prepare_store(blocks)
        assert result is not None

        # Complete store (triggers cascade)
        self.manager.complete_store(blocks, success=True)

        # Process finished jobs to complete cascade
        self.manager._process_finished_jobs()

        # Blocks should be in both secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3

        # Verify blocks are present
        assert self.secondary_tier1.lookup(blocks) == 3
        assert self.secondary_tier2.lookup(blocks) == 3

    def test_ref_cnt_protection_during_cascade(self):
        """Test that ref_cnt protects blocks during cascade."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Store to primary
        result = self.manager.prepare_store(blocks)
        assert result is not None
        self.manager.complete_store(blocks, success=True)

        # After complete_store, blocks should have ref_cnt > 0
        # (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier.blocks[block_hash]
            # ref_cnt should be 2 (one for each secondary tier)
            assert block.ref_cnt == 2

        # Process finished jobs to complete cascade
        self.manager._process_finished_jobs()

        # After cascade completes, ref_cnt should be 0
        for block_hash in blocks:
            block = self.primary_tier.blocks[block_hash]
            assert block.ref_cnt == 0

    def test_lookup_from_primary(self):
        """Test lookup when blocks are in primary tier."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Store blocks
        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)

        # Lookup should find all blocks in primary
        assert self.manager.lookup(blocks) == 3

    def test_promotion_from_secondary(self):
        """Test promotion of blocks from secondary to primary tier."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Manually add blocks to secondary tier (simulate previous cascade)
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        # Lookup should initiate promotion
        result = self.manager.lookup(blocks)
        assert result is None  # Retry later

        # Process finished jobs to complete promotion
        self.manager._process_finished_jobs()

        # Now blocks should be in primary tier
        assert self.primary_tier.lookup(blocks) == 3

        # Next lookup should succeed
        assert self.manager.lookup(blocks) == 3

    def test_partial_lookup(self):
        """Test lookup with partial hits."""
        blocks = [make_block_hash(1, i) for i in range(5)]

        # Store first 3 blocks to primary
        self.manager.prepare_store(blocks[:3])
        self.manager.complete_store(blocks[:3], success=True)

        # Lookup all 5 blocks should return 3 (first 3 found)
        assert self.manager.lookup(blocks) == 3

    def test_eviction_in_primary_tier(self):
        """Test eviction in primary tier when capacity is exceeded."""
        # Primary tier has capacity of 5 blocks
        # First, fill the primary tier
        blocks = [make_block_hash(1, i) for i in range(5)]
        result = self.manager.prepare_store(blocks)
        assert result is not None
        assert len(result.block_hashes_to_store) == 5
        self.manager.complete_store(blocks, success=True)

        # Process finished jobs to release ref_cnt from cascade
        self.manager._process_finished_jobs()

        # Now try to store 2 more blocks (should trigger eviction)
        more_blocks = [make_block_hash(2, i) for i in range(2)]
        result = self.manager.prepare_store(more_blocks)

        # Should evict 2 blocks from primary tier
        assert result is not None
        assert len(result.block_hashes_evicted) == 2
        assert len(result.block_hashes_to_store) == 2

    def test_touch_propagates_to_all_tiers(self):
        """Test that touch() propagates to all tiers."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Store blocks
        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)
        self.manager._process_finished_jobs()

        # Touch blocks
        self.manager.touch(blocks)

        # Verify touch was called on primary tier (check LRU order)
        # In LRU, touched blocks should be at the end
        primary_keys = list(self.primary_tier.blocks.keys())
        assert primary_keys[-3:] == list(reversed(blocks))

        # Verify touch was called on all secondary tiers
        secondary1_keys = list(self.secondary_tier1.blocks.keys())
        assert secondary1_keys[-3:] == list(reversed(blocks))

        secondary2_keys = list(self.secondary_tier2.blocks.keys())
        assert secondary2_keys[-3:] == list(reversed(blocks))

    def test_failed_store_no_cascade(self):
        """Test that failed GPU→primary store doesn't cascade."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Prepare store
        result = self.manager.prepare_store(blocks)
        assert result is not None

        # Complete store with failure
        self.manager.complete_store(blocks, success=False)

        # Process finished jobs
        self.manager._process_finished_jobs()

        # Blocks should NOT be in secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 0
        assert self.secondary_tier2.get_num_blocks() == 0

    def test_multiple_secondary_tiers_independent_eviction(self):
        """Test that secondary tiers manage their own evictions."""
        # Create tier with small capacity
        small_tier = DummySecondaryTier(
            tier_name="SmallStorage", max_blocks=5, simulate_async=False
        )
        large_tier = DummySecondaryTier(
            tier_name="LargeStorage", max_blocks=10, simulate_async=False
        )

        # Create a fresh primary tier for this test
        cpu_backend = CPUBackend(block_size=16, num_blocks=10)
        primary_tier = LRUOffloadingManager(cpu_backend)

        manager = TieredOffloadingManager(
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
        """Test that prepare_store() calls _process_finished_jobs() first."""
        blocks = [make_block_hash(1, i) for i in range(3)]

        # Store blocks
        self.manager.prepare_store(blocks)
        self.manager.complete_store(blocks, success=True)

        # Blocks should have ref_cnt = 2 (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier.blocks[block_hash]
            assert block.ref_cnt == 2

        # Call prepare_store again (should process finished jobs first)
        more_blocks = [make_block_hash(2, i) for i in range(2)]
        self.manager.prepare_store(more_blocks)

        # Original blocks should now have ref_cnt = 0
        for block_hash in blocks:
            block = self.primary_tier.blocks[block_hash]
            assert block.ref_cnt == 0


class TestTieredOffloadingWithoutSecondaryTiers:
    """Test TieredOffloadingManager with no secondary tiers (backward compat)."""

    def test_works_without_secondary_tiers(self):
        """Test that manager works with empty secondary_tiers list."""
        cpu_backend = CPUBackend(block_size=16, num_blocks=5)
        primary_tier = LRUOffloadingManager(cpu_backend)

        # Create manager with no secondary tiers
        manager = TieredOffloadingManager(primary_tier=primary_tier, secondary_tiers=[])

        blocks = [make_block_hash(1, i) for i in range(3)]

        # Should work like a regular OffloadingManager
        result = manager.prepare_store(blocks)
        assert result is not None
        manager.complete_store(blocks, success=True)

        assert manager.lookup(blocks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
