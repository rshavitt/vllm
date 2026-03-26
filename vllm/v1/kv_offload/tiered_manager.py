# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TiersOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a primary tier (with GPU access, currently
CPU-based) and zero or more secondary tiers (Storage, Network, etc.) to
provide hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Only the primary tier can directly access
   GPU memory (currently implemented using CPU memory)
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.protect_blocks() increments ref_cnt,
   protecting blocks from eviction until unprotect_blocks() is called
"""

from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    JobId,
    JobMetadata,
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.mediums import (
    BlockIDsLoadStoreSpec,
    CPUMemoryViewLoadStoreSpec,
)

logger = init_logger(__name__)


# TODO: Think of reorganizing the tiers manager feature into files/dirs
class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    # TODO: Rename to secondary tiers facing methods, or something similar...
    """CPUOffloadingManager with alias methods for use by TiersOffloadingManager."""

    def allocate_blocks(self, block_hashes) -> PrepareStoreOutput | None:
        return self.prepare_store(block_hashes)

    def finalize_blocks(self, block_hashes, success: bool = True) -> None:
        self.complete_store(block_hashes, success)

    def protect_blocks(self, block_hashes) -> LoadStoreSpec:
        return self.prepare_load(block_hashes)

    def unprotect_blocks(self, block_hashes) -> None:
        self.complete_load(block_hashes)

    def get_primary_kv_tensors(self) -> list[torch.Tensor]:
        """
        Get the primary tier's KV cache tensors.

        Returns the list of CPU tensors that store the KV cache data.
        TieredManager will pass memoryviews of these tensors to secondary tier
        managers for data transfer operations.

        TODO: This is a placeholder returning a dummy zero tensor.
        Actual implementation requires CPUOffloadingManager to maintain a
        reference to the worker's CPU tensors.

        Returns:
            List of CPU tensors storing KV cache data. Currently returns
            a dummy zero tensor as placeholder (wrong data).
        """
        # PRNOTE: This is a placeholder. The real implementation requires
        # CPUOffloadingManager to hold a reference to the worker's CPU KV
        # tensors and return them here. Until that's wired up, secondary tier
        # managers will receive memory views of a zero tensor (wrong data).
        return [torch.zeros(1)]


class TiersOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a primary tier (with GPU access, currently
    CPU-based) and zero or more secondary tiers (Storage, Network, etc.) to
    provide hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
    ):
        """
        Initialize the tiered offloading manager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Job tracking: maps job_id to metadata for each transfer direction
        # Store jobs: primary → secondary transfers
        self._store_jobs: dict[JobId, JobMetadata] = {}
        # Load jobs: secondary → primary transfers (promotions)
        self._load_jobs: dict[JobId, JobMetadata] = {}

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _create_memory_view_spec(
        self, cpu_blocks_spec: LoadStoreSpec, readonly: bool = False
    ) -> LoadStoreSpec:
        """
        Convert CPULoadStoreSpec to CPUMemoryViewLoadStoreSpec.

        Takes a spec with just block IDs and enhances it with memory views
        for direct CPU memory access by secondary tiers.

        Args:
            cpu_blocks_spec: CPULoadStoreSpec with block IDs
            readonly: If True, create readonly memory views (for read operations)

        Returns:
            CPUMemoryViewLoadStoreSpec with block IDs and memory views
        """
        # Type assertion: primary tier always returns BlockIDsLoadStoreSpec
        assert isinstance(cpu_blocks_spec, BlockIDsLoadStoreSpec)

        cpu_tensors = self.primary_tier.get_primary_kv_tensors()

        return CPUMemoryViewLoadStoreSpec(
            block_ids=cpu_blocks_spec.block_ids.tolist(),
            cpu_tensors=cpu_tensors,
            readonly=readonly,
        )

    def _process_finished_jobs(self):
        """
        Poll all secondary tiers for completed jobs and update state accordingly.

        This method:
        1. Calls get_finished() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.unprotect_blocks()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.finalize_blocks()
           to make blocks available
        """
        for tier in self.secondary_tiers:
            for completed_job in tier.get_finished():
                job_id = completed_job.job_id

                # Determine job type by checking which dictionary contains the job_id
                if job_id in self._store_jobs:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    job_metadata = self._store_jobs.pop(job_id)
                    job_metadata.spec.release()
                    self.primary_tier.unprotect_blocks(job_metadata.block_hashes)
                elif job_id in self._load_jobs:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    job_metadata = self._load_jobs.pop(job_id)
                    job_metadata.spec.release()
                    self.primary_tier.finalize_blocks(
                        job_metadata.block_hashes, completed_job.success
                    )
                else:
                    # Job ID not found in either dictionary - this shouldn't happen
                    logger.error(
                        "Received finished job for unknown job_id %d from tier %s",
                        job_id,
                        tier.get_tier_name(),
                    )

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """
        Find the length of the maximal series of blocks that are offloaded.

        Algorithm:
        1. Check primary tier first
        2. If not all blocks found, check all secondary tiers sequentially,
           promoting blocks from each tier that has hits and updating the
           remaining blocks to search for
        3. Return None to signal "retry later" if any promotions were initiated

        Args:
            block_hashes: Block hashes to look up.

        Returns:
            Number of consecutive blocks (from start) that are present,
            or None if blocks are being transferred (retry later).
        """
        # Process any completed async jobs first to ensure promoted blocks
        # are finalized and available in the primary tier
        self._process_finished_jobs()

        block_hashes_list = list(block_hashes)

        # Step 1: Check primary tier
        primary_hits = self.primary_tier.lookup(block_hashes_list)

        if primary_hits is None:
            # Primary tier is busy (blocks being transferred)
            return None

        if primary_hits == len(block_hashes_list):
            # All blocks in primary tier
            return primary_hits

        # Step 2: Check all secondary tiers for remaining blocks
        remaining_blocks = block_hashes_list[primary_hits:]

        # Track whether any promotions were initiated
        has_promotions = False

        for tier in self.secondary_tiers:
            if not remaining_blocks:
                # All blocks have been found
                break

            secondary_hits = tier.lookup(remaining_blocks)

            # Skip if tier is busy (None) or has no hits (0)
            if not secondary_hits:
                continue

            # Found blocks in this secondary tier, initiate promotion
            blocks_to_promote = remaining_blocks[:secondary_hits]
            self._initiate_promotion(tier, blocks_to_promote)
            has_promotions = True

            # Update remaining_blocks to continue searching for the rest
            remaining_blocks = remaining_blocks[secondary_hits:]

        # Step 3: If any promotions were initiated, return None to signal retry
        if has_promotions:
            return None

        # No more blocks found in any tier
        return primary_hits

    def _initiate_promotion(
        self, tier: SecondaryTierManager, block_hashes: list[BlockHash]
    ):
        """
        Initiate promotion of blocks from a secondary tier to the primary tier.

        This method:
        1. Calls primary.allocate_blocks() to allocate space in primary tier
        2. Calls tier.submit_load() to start async transfer: secondary→primary
        3. Tracks the job in _load_jobs dictionary

        Args:
            tier: The secondary tier to promote from
            block_hashes: Blocks to promote
        """
        # Allocate space in primary tier for promoted blocks
        primary_store_result = self.primary_tier.allocate_blocks(block_hashes)

        if primary_store_result is None:
            # Cannot allocate space in primary tier (full)
            # The next lookup() will retry
            return

        # Submit async load job: secondary→primary
        job_id = self._next_job_id()

        # Convert to memory view spec for secondary tier access (writable for loading)
        primary_write_spec = self._create_memory_view_spec(
            primary_store_result.store_spec, readonly=False
        )

        # Track this load job
        job_metadata = JobMetadata(
            job_id=job_id, block_hashes=block_hashes, spec=primary_write_spec
        )
        self._load_jobs[job_id] = job_metadata

        tier.submit_load(job_metadata)

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        CRITICAL: This method calls _process_finished_jobs() FIRST to ensure
        that any completed promotions have been finalized and blocks are ready.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            block_hashes: Blocks to prepare for loading.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        # Process completed promotions to ensure blocks are ready
        self._process_finished_jobs()

        return self.primary_tier.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Mark blocks as recently used in all tiers.

        Args:
            block_hashes: Blocks to mark as recently used.
        """
        block_hashes = list(block_hashes)
        self.primary_tier.touch(block_hashes)
        for tier in self.secondary_tiers:
            tier.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            block_hashes: Blocks that finished loading.
        """
        self.primary_tier.complete_load(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        Args:
            block_hashes: Blocks to prepare for storing.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # Step 1: Poll for completed async jobs FIRST
        # This decrements ref_cnt on primary blocks that have been
        # successfully transferred to secondary tiers.
        self._process_finished_jobs()

        # Step 2: Store to primary tier
        primary_result = self.primary_tier.prepare_store(block_hashes)

        # Note: Secondary tier cascading will happen in complete_store()
        # after the GPU→Primary transfer completes and blocks are ready.

        return primary_result

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.protect_blocks() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            block_hashes: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
        """
        # Materialize only if success=True (needed for cascading to secondary tiers)
        block_hashes_list = list(block_hashes) if success else block_hashes

        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(block_hashes_list, success)

        if not success:
            # If GPU→Primary transfer failed, don't cascade to secondary tiers
            return

        # At this point, success=True is guaranteed, so block_hashes_list
        # is list[BlockHash]
        assert isinstance(block_hashes_list, list)

        # Step 2: Cascade to ALL secondary tiers
        # For each secondary tier, call primary.protect_blocks() to get the
        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
        # eviction during the async transfer). One protect_blocks() call per
        # secondary tier.
        for tier in self.secondary_tiers:
            # Get spec for reading from primary tier AND increment ref_cnt
            primary_blocks_spec = self.primary_tier.protect_blocks(block_hashes_list)

            # Convert to memory view spec for secondary tier access
            # (readonly for storing)
            primary_read_spec = self._create_memory_view_spec(
                primary_blocks_spec, readonly=True
            )

            # Submit async store job: primary→secondary
            job_id = self._next_job_id()

            # Track this store job
            job_metadata = JobMetadata(
                job_id=job_id, block_hashes=block_hashes_list, spec=primary_read_spec
            )
            self._store_jobs[job_id] = job_metadata

            tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight.
        # Their completion is tracked via get_finished() / _process_finished_jobs().

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take offloading events from the primary tier.

        Note: Currently only primary tier events are tracked. Secondary tier
        events could be added in the future if needed.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()

        # Also yield events from primary tier
        yield from self.primary_tier.take_events()
