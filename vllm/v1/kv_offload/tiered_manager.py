# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager: Multi-tier KV cache offloading orchestrator.

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

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    JobId,
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
    SecondaryTierManager,
)


class TieredOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a primary tier (with GPU access, currently
    CPU-based) and zero or more secondary tiers (Storage, Network, etc.) to
    provide hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished()
      - Secondary tiers return CompletedJob objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: OffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
    ):
        """
        Initialize the tiered offloading manager.

        Args:
            primary_tier: The primary tier manager (e.g., LRUOffloadingManager
                         with CPUBackend)
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
        """
        self.primary_tier = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

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
            for completed in tier.get_finished():
                if completed.is_store:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    self.primary_tier.unprotect_blocks(completed.block_hashes)
                else:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    self.primary_tier.finalize_blocks(
                        completed.block_hashes, completed.success
                    )

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """
        Find the length of the maximal series of blocks that are offloaded.

        Algorithm:
        1. Check primary tier first
        2. If not all blocks found, check secondary tiers
        3. If found in secondary tier, initiate promotion and return None
           (retry later)

        Args:
            block_hashes: Block hashes to look up.

        Returns:
            Number of consecutive blocks (from start) that are present,
            or None if blocks are being transferred (retry later).
        """
        block_hashes_list = list(block_hashes)

        # Step 1: Check primary tier
        primary_hits = self.primary_tier.lookup(block_hashes_list)

        if primary_hits is None:
            # Primary tier is busy (blocks being transferred)
            return None

        if primary_hits == len(block_hashes_list):
            # All blocks in primary tier
            return primary_hits

        # Step 2: Check secondary tiers for remaining blocks
        remaining_blocks = block_hashes_list[primary_hits:]

        for tier in self.secondary_tiers:
            secondary_hits = tier.lookup(remaining_blocks)

            if secondary_hits is None:
                # Blocks are being transferred in this tier, retry later
                return None

            if secondary_hits > 0:
                # Found blocks in this secondary tier, initiate promotion
                blocks_to_promote = remaining_blocks[:secondary_hits]
                self._initiate_promotion(tier, blocks_to_promote)
                # Return None to signal "retry later"
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
        tier.submit_load(job_id, block_hashes, primary_store_result.store_spec)

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            block_hashes: Blocks to prepare for loading.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        return self.primary_tier.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Mark blocks as recently used in all tiers.

        Args:
            block_hashes: Blocks to mark as recently used.
        """
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
        if primary_result is None:
            return None

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

        Args:
            block_hashes: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
        """
        # IMPORTANT: Materialize the iterable BEFORE calling primary.complete_store()
        # because the iterable might be consumed by that call
        block_hashes_list = list(block_hashes)

        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(block_hashes_list, success)

        if not success:
            # If GPU→Primary transfer failed, don't cascade to secondary tiers
            return

        # Step 2: Cascade to ALL secondary tiers
        # For each secondary tier, call primary.protect_blocks() to get the
        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
        # eviction during the async transfer). One protect_blocks() call per
        # secondary tier.
        for tier in self.secondary_tiers:
            # Get spec for reading from primary tier AND increment ref_cnt
            primary_load_spec = self.primary_tier.protect_blocks(block_hashes_list)

            # Submit async store job: primary→secondary
            job_id = self._next_job_id()
            tier.submit_store(job_id, block_hashes_list, primary_load_spec)

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


# Made with Bob
