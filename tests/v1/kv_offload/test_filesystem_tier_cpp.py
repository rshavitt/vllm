# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FileSystemTierManagerCpp.

These tests use the real _kv_file_system_ops C++ extension to verify the
C++ filesystem tier implementation. The tier manager writes KV cache blocks
to disk via the C++ thread pool and reads them back, verifying data integrity
throughout the process.

Note: These tests require the _kv_file_system_ops extension to be built.
"""

import os
import time

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

_BLOCK_ELEMENTS = 512 * 1024  # 2 MB per block (float32 × 512K = 2MB)
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
    num_total_blocks: int = 4,
    **kwargs,
) -> tuple[FileSystemTierManagerCpp, torch.Tensor]:
    """Create a FileSystemTierManagerCpp and wire a test primary view into it."""
    tier = FileSystemTierManagerCpp(base_path=base_path, **kwargs)
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
    Works with the real async C++ extension.
    """
    results = []
    for _ in range(max_rounds):
        results.extend(tier.get_finished())
        if not tier._active_jobs:
            break
        time.sleep(0.005)
    return results


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------

class TestFileSystemTierBasic:
    """Tests for basic tier functionality with real C++ I/O."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tier, self.tensor = make_tier_with_view(
            str(tmp_path),
            n_read_threads=4,
            n_write_threads=4,
        )
        yield

    def test_get_tier_name(self):
        t = FileSystemTierManagerCpp(base_path="/tmp/test", tier_name="MyTier")
        assert t.get_tier_name() == "MyTier"

    def test_lookup_empty_tier(self):
        assert self.tier.lookup(key(1)) is False
        assert self.tier.lookup(key(2)) is False

    def test_get_file_name_structure(self):
        tier = FileSystemTierManagerCpp(base_path="/kvcache")
        path = tier.get_file_name(key(0))
        assert path == "/kvcache/000/00/0000000000000000.bin"

    def test_get_file_name_consistent_for_same_hash(self):
        tier = FileSystemTierManagerCpp(base_path="/kvcache")
        k = key(12345)
        assert tier.get_file_name(k) == tier.get_file_name(k)

    def test_store_creates_file_and_lookup_succeeds(self):
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        # Verify file exists and lookup returns True
        assert self.tier.lookup(key(1)) is True

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
# I/O integration tests — real C++ extension
# ---------------------------------------------------------------------------

class TestFileSystemTierIO:

    def _make_tier(self, tmp_path, num_total_blocks: int = 8, **kwargs):
        tier = FileSystemTierManagerCpp(
            base_path=str(tmp_path / "kvcache"),
            n_read_threads=4,
            n_write_threads=4,
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


# ---------------------------------------------------------------------------
# End-to-end tests with primary tier integration
# ---------------------------------------------------------------------------

class TestFileSystemTierE2EWithPrimary:
    """
    End-to-end tests integrating FileSystemTierManagerCpp with
    CPUPrimaryTierOffloadingManager using real C++ disk I/O.
    
    These tests verify full data integrity through cascade and promotion
    pipelines with actual file system operations.
    """

    @pytest.fixture
    def setup_manager(self, tmp_path):
        """Setup TieringOffloadingManager with real primary and C++ filesystem tiers."""
        block_elements = _BLOCK_ELEMENTS
        num_primary_blocks = 10

        # Create primary tier
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
        )

        # Provide a plain CPU tensor as the shared KV buffer
        cpu_tensor = torch.zeros((num_primary_blocks, block_elements), dtype=torch.float32)
        primary_tier.create_kv_memoryview = lambda: memoryview(cpu_tensor.numpy())
        
        # Create C++ filesystem tier with real I/O
        fs_tier = FileSystemTierManagerCpp(
            base_path=str(tmp_path / "kvcache"),
            n_read_threads=4,
            n_write_threads=4,
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

        # Wait for cascade to complete
        for _ in range(20):
            manager._process_finished_jobs()
            all_done = all(
                primary_tier._policy.get(k).ref_cnt == 0
                for k in keys if primary_tier._policy.get(k) is not None
            )
            if all_done:
                break
            time.sleep(0.05)

        # Evict blocks from primary tier by storing new blocks
        evict_keys = [key(300 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, ReqContext())
        assert result is not None
        assert len(result.evicted_keys) >= 4
        
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
        
        # Trigger promotion by lookup
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

        # Wait for cascade to complete
        for _ in range(20):
            manager._process_finished_jobs()
            all_done = all(
                primary_tier._policy.get(k).ref_cnt == 0
                for k in keys if primary_tier._policy.get(k) is not None
            )
            if all_done:
                break
            time.sleep(0.05)

        # Evict from primary by filling it
        evict_keys = [key(500 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, ReqContext())
        assert result is not None
        assert len(result.evicted_keys) >= num_blocks
        
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
        
        # Trigger promotion
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
