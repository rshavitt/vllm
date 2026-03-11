# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DummySecondaryTier: A simple in-memory secondary tier for testing.

This implementation provides a minimal secondary tier that stores blocks
in memory (using a dictionary) and simulates async transfers with immediate
completion. It's useful for testing the TieredOffloadingManager without
requiring actual storage or network backends.
"""

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    JobId,
    JobResult,
    LoadStoreSpec,
    PrepareStoreOutput,
    SecondaryTierManager,
    TransferDirection,
)


class DummyLoadStoreSpec(LoadStoreSpec):
    """
    Minimal LoadStoreSpec for DummySecondaryTier testing.

    This spec is never actually used for I/O since DummySecondaryTier
    stores blocks in memory. It exists to provide proper type semantics
    and serve as a template for real secondary tier implementations.
    """

    @staticmethod
    def medium() -> str:
        return "Dummy"


class DummySecondaryTier(SecondaryTierManager):
    """
    A simple in-memory secondary tier for testing.

    This implementation:
    - Stores blocks in a dictionary (block_hash -> True)
    - Simulates async transfers with immediate completion
    - Uses LRU eviction policy
    - Tracks in-flight transfers to return None from lookup()
    """

    def __init__(
        self,
        tier_name: str = "DummyStorage",
        max_blocks: int = 1000,
        simulate_async: bool = False,
    ):
        """
        Initialize the dummy secondary tier.

        Args:
            tier_name: Name of this tier (for identification)
            max_blocks: Maximum number of blocks this tier can store
            simulate_async: If True, jobs complete on next get_finished() call.
                          If False, jobs complete immediately.
        """
        self.tier_name = tier_name
        self.max_blocks = max_blocks
        self.simulate_async = simulate_async

        # block_hash -> True (only care about presence)
        self.blocks: OrderedDict[BlockHash, bool] = OrderedDict()

        # Tracks in-flight transfers: block_hash -> job_id
        self.in_flight: dict[BlockHash, JobId] = {}

        # Completed jobs waiting to be retrieved by get_finished()
        self.completed_jobs: list[JobResult] = []

        # Pending jobs (for simulated async mode)
        self.pending_jobs: list[JobResult] = []

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """
        Check which blocks exist in this secondary tier.

        Args:
            block_hashes: Block hashes to look up.

        Returns:
            Number of consecutive blocks (from start) that are present and ready,
            or None if blocks are being transferred (retry later).
        """
        hit_count = 0
        for block_hash in block_hashes:
            # Check if block is in-flight
            if block_hash in self.in_flight:
                # Block is being transferred, return None (retry later)
                return None

            # Check if block exists in this tier
            if block_hash not in self.blocks:
                break

            hit_count += 1

        return hit_count

    def submit_store(
        self,
        job_id: JobId,
        block_hashes: Iterable[BlockHash],
        primary_load_spec: LoadStoreSpec,
    ) -> PrepareStoreOutput | None:
        """
        Submit an async job to store blocks from primary tier to this tier.

        Args:
            job_id: Unique identifier for this transfer job.
            block_hashes: Blocks to store.
            primary_load_spec: Spec for reading blocks from primary tier.

        Returns:
            PrepareStoreOutput describing which blocks will be stored and
            what was evicted, or None if the store cannot proceed.
        """
        block_hashes_list = list(block_hashes)

        # Filter out blocks already present
        blocks_to_store = [bh for bh in block_hashes_list if bh not in self.blocks]

        if not blocks_to_store:
            # All blocks already present
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=DummyLoadStoreSpec(),
                block_hashes_evicted=[],
            )

        # Evict blocks if needed (LRU policy)
        num_blocks_to_evict = len(blocks_to_store) - (
            self.max_blocks - len(self.blocks)
        )

        evicted = []
        if num_blocks_to_evict > 0:
            # Evict oldest blocks (LRU)
            protected = set(block_hashes_list)
            for block_hash in list(self.blocks.keys()):
                if block_hash not in protected and block_hash not in self.in_flight:
                    del self.blocks[block_hash]
                    evicted.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break
            else:
                # Could not evict enough blocks
                return None

        # Mark blocks as in-flight
        for block_hash in blocks_to_store:
            self.in_flight[block_hash] = job_id

        # Create completed job
        completed = JobResult(
            job_id=job_id,
            block_hashes=blocks_to_store,
            direction=TransferDirection.PRIMARY_TO_SECONDARY,
            success=True,
        )

        if self.simulate_async:
            # Job will complete on next get_finished() call
            self.pending_jobs.append(completed)
        else:
            # Job completes immediately
            self._complete_store_job(completed)

        return PrepareStoreOutput(
            block_hashes_to_store=blocks_to_store,
            store_spec=DummyLoadStoreSpec(),
            block_hashes_evicted=evicted,
        )

    def submit_load(
        self,
        job_id: JobId,
        block_hashes: Iterable[BlockHash],
        primary_store_spec: LoadStoreSpec,
    ) -> LoadStoreSpec | None:
        """
        Submit an async job to load blocks from this tier to primary tier.

        Args:
            job_id: Unique identifier for this transfer job.
            block_hashes: Blocks to load.
            primary_store_spec: Spec for writing blocks into primary tier.

        Returns:
            LoadStoreSpec for reading from this tier, or None if load cannot proceed.
        """
        block_hashes_list = list(block_hashes)

        # Verify all blocks exist
        for block_hash in block_hashes_list:
            if block_hash not in self.blocks:
                return None

        # Mark blocks as in-flight
        for block_hash in block_hashes_list:
            self.in_flight[block_hash] = job_id

        # Create completed job
        completed = JobResult(
            job_id=job_id,
            block_hashes=block_hashes_list,
            direction=TransferDirection.SECONDARY_TO_PRIMARY,
            success=True,
        )

        if self.simulate_async:
            # Job will complete on next get_finished() call
            self.pending_jobs.append(completed)
        else:
            # Job completes immediately
            self._complete_load_job(completed)

        return DummyLoadStoreSpec()

    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished async jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        # Move pending jobs to completed
        if self.simulate_async and self.pending_jobs:
            for job in self.pending_jobs:
                if job.direction == TransferDirection.PRIMARY_TO_SECONDARY:
                    self._complete_store_job(job)
                else:
                    self._complete_load_job(job)
            self.pending_jobs.clear()

        # Return completed jobs
        result = self.completed_jobs
        self.completed_jobs = []
        return result

    def _complete_store_job(self, job: JobResult):
        """Complete a store job by adding blocks to storage."""
        for block_hash in job.block_hashes:
            self.blocks[block_hash] = True
            del self.in_flight[block_hash]
        self.completed_jobs.append(job)

    def _complete_load_job(self, job: JobResult):
        """Complete a load job by removing in-flight markers."""
        for block_hash in job.block_hashes:
            del self.in_flight[block_hash]
        self.completed_jobs.append(job)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Mark blocks as recently used (move to end of LRU list).

        Args:
            block_hashes: Blocks to mark as recently used.
        """
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.blocks:
                self.blocks.move_to_end(block_hash)

    def get_tier_name(self) -> str:
        """
        Get the name of this tier.

        Returns:
            Tier name string.
        """
        return self.tier_name

    def get_num_blocks(self) -> int:
        """Get the number of blocks currently stored in this tier."""
        return len(self.blocks)

    def get_num_in_flight(self) -> int:
        """Get the number of blocks currently in-flight."""
        return len(self.in_flight)

    def clear(self):
        """Clear all blocks and in-flight transfers (for testing)."""
        self.blocks.clear()
        self.in_flight.clear()
        self.completed_jobs.clear()
        self.pending_jobs.clear()


# Made with Bob
