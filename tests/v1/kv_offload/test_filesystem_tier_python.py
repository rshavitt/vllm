# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FileSystemTierManagerPython.

These tests use real disk I/O to verify the Python filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import os
import time

import numpy as np

import pytest
import torch

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.file_system_python import (
    FileSystemTierManagerPython,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 512 * 1024  # 2 MB per block (float32 × 512K = 2MB)
_DTYPE = torch.float32


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)

def make_tier_with_view(
    base_path: str,
    num_total_blocks: int = 32,
) -> FileSystemTierManagerPython:
    tier = FileSystemTierManagerPython(base_path=base_path)
    tensor = torch.zeros((num_total_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
    tier.set_primary_view(memoryview(tensor.numpy()))
    return tier


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(job_id=job_id, keys=keys, block_ids=np.array(block_ids, dtype=np.int64))


def drain(tier: FileSystemTierManagerPython, max_rounds: int = 40) -> list:
    """
    Call get_finished() repeatedly until all jobs are resolved or timeout.
    """
    results = []
    for _ in range(max_rounds):
        results.extend(tier.get_finished())
        if not tier._active_jobs:
            break
        time.sleep(0.01)
    return results


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------

class TestPythonFSTierBasic:
    """Tests for basic tier functionality with real I/O."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tier = FileSystemTierManagerPython(
            base_path=str(tmp_path)
        )
        self.tensor = torch.zeros((4, _BLOCK_ELEMENTS), dtype=_DTYPE)
        self.tier.set_primary_view(memoryview(self.tensor.numpy()))
        yield

    def test_lookup_empty_tier(self):
        assert self.tier.lookup(key(1)) is False
        assert self.tier.lookup(key(2)) is False

    def test_get_file_name_structure(self):
        tier = FileSystemTierManagerPython(base_path="/kvcache")
        path = tier.get_file_name(key(0))
        assert path == "/kvcache/000/00/0000000000000000.bin"

    def test_store_creates_file_and_lookup_succeeds(self):
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        # Verify file exists and lookup returns True
        assert self.tier.lookup(key(1)) is True

    def test_store_skips_already_stored_blocks(self):
        # Store once
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        drain(self.tier)
        
        # Try to store again - should skip
        job2 = make_job(2, [key(1)], [0])
        submitted = self.tier.submit_store(job2)
        assert not submitted  # Should return False since block already exists

    def test_store_then_load_roundtrip(self):
        job_s = make_job(1, [key(1), key(2)], [0, 1])
        self.tier.submit_store(job_s)
        store_results = drain(self.tier)
        assert all(r.success for r in store_results)
        assert self.tier.lookup(key(1)) is True
        assert self.tier.lookup(key(2)) is True

        job_l = make_job(2, [key(1), key(2)], [2, 3])
        self.tier.submit_load(job_l)
        load_results = drain(self.tier)
        assert all(r.success for r in load_results)
        # Blocks stay on disk after load
        assert self.tier.lookup(key(1)) is True
        assert self.tier.lookup(key(2)) is True

    def test_failed_store_with_invalid_path(self, tmp_path):
        """Test that failed store operations are reported correctly."""
        # Create a tier with an invalid base path to trigger I/O failures
        tier = FileSystemTierManagerPython(
            base_path="/dev/null/invalid_path"
        )
        tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
        tier.set_primary_view(memoryview(tensor.numpy()))
        job = make_job(1, [key(5)], [0])
        tier.submit_store(job)
        results = drain(tier)

        assert len(results) == 1
        assert not results[0].success
        assert tier.lookup(key(5)) is False

    def test_multiple_jobs_tracked_independently(self):
        job1 = make_job(1, [key(1)], [0])
        job2 = make_job(2, [key(2)], [1])
        self.tier.submit_store(job1)
        self.tier.submit_store(job2)
        results = drain(self.tier)
        job_ids = {r.job_id for r in results}
        assert job_ids == {1, 2}
        assert self.tier.lookup(key(1)) is True
        assert self.tier.lookup(key(2)) is True


# ---------------------------------------------------------------------------
# I/O integration tests — real disk
# ---------------------------------------------------------------------------

class TestPythonFSTierIO:

    def _make_tier(self, base_path, num_total_blocks: int = 8, **kwargs):
        tier = FileSystemTierManagerPython(
            base_path=str(base_path),
            **kwargs,
        )
        tensor = torch.zeros((num_total_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
        tier.set_primary_view(memoryview(tensor.numpy()))
        return tier, tensor

    def test_store_creates_file(self, tmp_path):
        tier, _ = self._make_tier(tmp_path)
        job = make_job(1, [key(1)], [0])
        tier.submit_store(job)
        results = drain(tier)
        assert results[0].success
        assert tier.lookup(key(1)) is True
        dest = tier.get_file_name(key(1))
        assert os.path.exists(dest), f"Expected file at {dest}"

    def test_store_load_data_integrity(self, tmp_path):
        """Data written by store must be exactly recovered by load."""
        num_blocks = 4
        num_total = 8
        tier, tensor = self._make_tier(tmp_path, num_total_blocks=num_total)

        # Fill source blocks with random data
        for bid in range(num_blocks):
            tensor[bid] = torch.rand((_BLOCK_ELEMENTS,), dtype=_DTYPE)
        expected = tensor[:num_blocks].clone()

        block_ids = list(range(num_blocks))
        keys = [key(i) for i in range(num_blocks)]

        tier.submit_store(make_job(1, keys, block_ids))
        results = drain(tier)
        assert all(r.success for r in results)

        # Overwrite source blocks to prove data is read from disk
        tensor[:num_blocks] = 0.0

        load_ids = list(range(num_blocks, 2 * num_blocks))
        tier.submit_load(make_job(2, keys, load_ids))
        results = drain(tier)
        assert all(r.success for r in results)

        for i, bid in enumerate(load_ids):
            assert torch.allclose(
                tensor[bid], expected[i]
            ), f"Block {bid} data mismatch after store+load"

    def test_store_load_multiple_blocks(self, tmp_path):
        num_blocks = 8
        num_total = 16
        tier, tensor = self._make_tier(tmp_path, num_total_blocks=num_total)

        for bid in range(num_blocks):
            tensor[bid] = float(bid + 1)
        expected = tensor[:num_blocks].clone()

        keys = [key(i + 100) for i in range(num_blocks)]
        tier.submit_store(make_job(10, keys, list(range(num_blocks))))
        results = drain(tier)
        assert all(r.success for r in results)

        tensor[:num_blocks] = 0.0
        load_ids = list(range(num_blocks, 2 * num_blocks))
        tier.submit_load(make_job(11, keys, load_ids))
        results = drain(tier)
        assert all(r.success for r in results)

        for i, bid in enumerate(load_ids):
            assert torch.allclose(tensor[bid], expected[i])

    def test_idempotent_store(self, tmp_path):
        """Storing the same block twice should be a no-op after first store."""
        tier, _ = self._make_tier(tmp_path)
        job1 = make_job(1, [key(7)], [0])
        tier.submit_store(job1)
        drain(tier)
        assert tier.lookup(key(7)) is True

        job2 = make_job(2, [key(7)], [0])
        submitted = tier.submit_store(job2)
        # No active job should be created since block is already stored.
        assert not submitted


# ---------------------------------------------------------------------------
# End-to-end tests with primary tier integration
# ---------------------------------------------------------------------------

class TestPythonFileSystemTierE2EWithPrimary:
    """
    End-to-end tests integrating FileSystemTierManagerPython with
    CPUPrimaryTierOffloadingManager using real disk I/O.
    
    These tests verify full data integrity through cascade and promotion
    pipelines with actual file system operations.
    """

    @pytest.fixture
    def setup_manager(self, tmp_path):
        """Setup TieringOffloadingManager with real primary and Python filesystem tiers."""
        from vllm.v1.kv_offload.tiering.manager import (
            CPUPrimaryTierOffloadingManager,
            TieringOffloadingManager,
        )

        block_elements = _BLOCK_ELEMENTS
        num_primary_blocks = 10

        # Create primary tier
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
        )

        # Provide a plain CPU tensor as the shared KV buffer so that
        # TieringOffloadingManager can wire secondary tier memoryviews
        # without requiring a real SharedOffloadRegion.
        cpu_tensor = torch.zeros((num_primary_blocks, block_elements), dtype=torch.float32)
        primary_tier.create_kv_memoryview = lambda: memoryview(cpu_tensor.numpy())
        
        # Create Python filesystem tier with real I/O
        fs_tier = FileSystemTierManagerPython(
            base_path=str(tmp_path / "kvcache"),
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
        keys = [key(100 + i) for i in range(num_blocks)]
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
        for k in keys:
            assert primary_tier.lookup(k, ReqContext()) is True
            assert fs_tier.lookup(k) is True
        
        # Verify data integrity by reading from disk
        for k in keys:
            file_path = fs_tier.get_file_name(k)
            assert os.path.isfile(file_path), f"File not found: {file_path}"
            with open(file_path, "rb") as f:
                raw = f.read(block_elements * 4)
            actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
            assert torch.allclose(actual, expected_data[k]), \
                f"Data mismatch for block {k}"

    def test_full_promotion_with_data_integrity(self, setup_manager):
        """
        Pre-populate filesystem tier with blocks containing known data,
        trigger promotion by calling lookup(), and verify data integrity
        matches original data.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Generate unique data for blocks
        num_blocks = 4
        keys = [key(200 + i) for i in range(num_blocks)]
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

        # Wait for cascade to complete before trying to evict
        for _ in range(20):
            manager._process_finished_jobs()
            # Check if cascade is done (ref_cnt == 0 for all blocks)
            all_done = all(
                primary_tier._policy.get(k).ref_cnt == 0
                for k in keys if primary_tier._policy.get(k) is not None
            )
            if all_done:
                break
            time.sleep(0.05)

        # Evict blocks from primary tier by storing new blocks.
        # Primary has 10 slots, 4 are filled, 6 free. Store 10 more to force
        # all 4 original blocks to be evicted (4 + 10 = 14, 14 - 10 = 4 evictions).
        evict_keys = [key(300 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, ReqContext())
        assert result is not None
        assert len(result.evicted_keys) >= 4  # All 4 original blocks should be evicted
        
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for block_id in spec.block_ids:
            cpu_tensor[int(block_id)] = 0.0
        manager.complete_store(evict_keys, success=True)
        
        # Wait for cascade of new blocks
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify blocks are only in filesystem tier
        for k in keys:
            assert primary_tier.lookup(k, ReqContext()) is False
            assert fs_tier.lookup(k) is True
        
        # Trigger promotion by lookup (each call initiates promotion for that key)
        for k in keys:
            manager.lookup(k, ReqContext())

        # Wait for promotion to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)

        # Verify blocks are now in primary tier
        assert all(manager.lookup(k, ReqContext()) is True for k in keys)

        # Verify data integrity after promotion
        load_spec = primary_tier.prepare_load(keys, ReqContext())
        for i, block_id in enumerate(load_spec.block_ids):
            actual_data = cpu_tensor[int(block_id)]
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
        keys = [key(400 + i) for i in range(num_blocks)]
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

        # Wait for cascade to complete before trying to evict
        for _ in range(20):
            manager._process_finished_jobs()
            all_done = all(
                primary_tier._policy.get(k).ref_cnt == 0
                for k in keys if primary_tier._policy.get(k) is not None
            )
            if all_done:
                break
            time.sleep(0.05)

        # Evict from primary by filling it (10 slots, 3 filled, store 10 more to trigger eviction)
        evict_keys = [key(500 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, ReqContext())
        assert result is not None
        assert len(result.evicted_keys) >= num_blocks  # Original blocks should be evicted
        
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for block_id in spec.block_ids:
            cpu_tensor[int(block_id)] = 0.0
        manager.complete_store(evict_keys, success=True)
        
        # Wait for cascade of new blocks
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify original blocks are evicted from primary
        for k in keys:
            assert primary_tier.lookup(k, ReqContext()) is False
        
        # Trigger promotion (each lookup initiates promotion for that key)
        for k in keys:
            manager.lookup(k, ReqContext())

        # Wait for promotion to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)

        # Verify data integrity after roundtrip
        assert all(manager.lookup(k, ReqContext()) is True for k in keys)
        load_spec = primary_tier.prepare_load(keys, ReqContext())
        
        for i, block_id in enumerate(load_spec.block_ids):
            actual_data = cpu_tensor[int(block_id)]
            expected = expected_data[keys[i]]
            assert torch.allclose(actual_data, expected, rtol=1e-5, atol=1e-7), \
                f"Block {i} data mismatch after roundtrip"

    def test_ref_cnt_protection_during_async_cascade(self, setup_manager):
        """
        Store blocks to primary tier, verify ref_cnt prevents eviction during
        async cascade, wait for cascade completion, and verify ref_cnt is
        released after cascade.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Store blocks to primary
        keys = [key(900 + i) for i in range(3)]
        result = manager.prepare_store(keys, ReqContext())
        assert result is not None
        
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for block_id in spec.block_ids:
            cpu_tensor[int(block_id)] = torch.rand(block_elements, dtype=torch.float32)
        
        manager.complete_store(keys, success=True)
        
        # Verify ref_cnt is incremented during cascade (1 per secondary tier)
        for k in keys:
            block_status = primary_tier._policy.get(k)
            assert block_status is not None
            assert block_status.ref_cnt == 1, f"Expected ref_cnt=1, got {block_status.ref_cnt}"
        
        # Wait for cascade to complete (may take multiple rounds)
        for _ in range(20):
            manager._process_finished_jobs()
            # Check if all ref_cnts are released
            all_released = all(
                primary_tier._policy.get(k).ref_cnt == 0
                for k in keys if primary_tier._policy.get(k) is not None
            )
            if all_released:
                break
            time.sleep(0.05)
        
        # Verify ref_cnt is released after cascade
        for k in keys:
            block_status = primary_tier._policy.get(k)
            assert block_status is not None
            assert block_status.ref_cnt == 0, f"Expected ref_cnt=0, got {block_status.ref_cnt}"

    def test_multiple_filesystem_tiers_independent_io(self, tmp_path):
        """
        Use two Python filesystem tiers with different capacities, verify both
        receive cascaded data, and verify data integrity in both tiers.
        """
        from vllm.v1.kv_offload.tiering.manager import (
            CPUPrimaryTierOffloadingManager,
            TieringOffloadingManager,
        )
        
        block_elements = _BLOCK_ELEMENTS
        num_primary_blocks = 10

        # Create primary tier
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
        )
        cpu_tensor = torch.zeros((num_primary_blocks, block_elements), dtype=torch.float32)
        primary_tier.create_kv_memoryview = lambda: memoryview(cpu_tensor.numpy())

        # Create two Python filesystem tiers
        fs_tier1 = FileSystemTierManagerPython(
            base_path=str(tmp_path / "tier1"),
        )
        fs_tier2 = FileSystemTierManagerPython(
            base_path=str(tmp_path / "tier2"),
        )
        
        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[fs_tier1, fs_tier2],
        )
        
        try:
            # Store blocks to primary (cascades to both tiers)
            keys = [key(1000 + i) for i in range(5)]
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
            for k in keys:
                assert fs_tier1.lookup(k) is True
                assert fs_tier2.lookup(k) is True

            # Verify data integrity in both tiers
            for k in keys:
                file_path1 = fs_tier1.get_file_name(k)
                file_path2 = fs_tier2.get_file_name(k)
                assert os.path.isfile(file_path1), f"File not found in tier1: {file_path1}"
                assert os.path.isfile(file_path2), f"File not found in tier2: {file_path2}"
        
        finally:
            manager.shutdown()