# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - find the length of the maximal series of blocks,
        starting from the first one, that are all offloaded.
    prepare_load() - prepare given blocks to be read.
        The given blocks will be protected from eviction.
        This function returns a LoadSpec which encapsulates
        information required for performing the load.
    touch() - marks the give blocks as recently used. Can be used
        to track block's LRU. This function is separated from the
        prepare_load function to allow setting block recency even
        for blocks which do not need reading from the cache, such as
        blocks that are cached by the GPU prefix cache.
    complete_load() - mark blocks which were previously prepared to be
        loaded as done loading. This is to re-allow their eviction.
    prepare_store() - prepare the given blocks to be written.
        Returns a StoreSpec encapsulating offloading information,
        as well as a list of blocks that were evicted as a result.
    complete_store() - marks a previous store as completed.
        Following this call, the given blocks will become loadable.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

# Type alias for job IDs used in async transfer tracking
JobId = int


class LoadStoreSpec(ABC):
    """
    Abstract metadata that encapsulates information allowing a worker
    to load, and optionally also to store, blocks of KV data.
    """

    @staticmethod
    @abstractmethod
    def medium() -> str:
        """
        Returns a string representation of the medium type
        this store/load targets.
        """
        pass


@dataclass
class PrepareStoreOutput:
    block_hashes_to_store: list[BlockHash]
    store_spec: LoadStoreSpec
    block_hashes_evicted: list[BlockHash]


@dataclass
class OffloadingEvent:
    block_hashes: list[BlockHash]
    block_size: int
    medium: str
    # True if blocks are removed, False if stored
    removed: bool


@dataclass
class CompletedJob:
    """Result of a completed async transfer job."""

    job_id: JobId
    block_hashes: list[BlockHash]
    is_store: bool  # True if primary→secondary, False if secondary→primary
    success: bool


class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """
        Finds the length of the maximal series of blocks, starting from the
        first one, that are all offloaded.

        Args:
            block_hashes: the hashes identifying the blocks to lookup.

        Returns:
            An integer representing the maximal number of blocks that
            are currently offloaded, or None if the lookup should be retried
            later. Returning None will delay the request handling by the vLLM
            scheduler.
        """
        pass

    @abstractmethod
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
        """
        pass

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        return

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        return

    @abstractmethod
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
        """
        pass

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If success is False, blocks that were not marked as stored will be
        removed.

        Args:
            block_hashes: the hashes identifying the blocks.
            success: whether the blocks were stored successfully.
        """
        return

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take the offloading events from the manager.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        return ()

    # Tier-agnostic API for primary tier operations
    # These methods provide intent-based names that work regardless of
    # data flow direction, making tiered manager code self-documenting.

    def allocate_blocks(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Allocate space for blocks in this tier.

        This is a tier-agnostic alias for prepare_store() that makes the
        intent clear when used by a tiered manager during promotion
        (secondary→primary transfer).

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A PrepareStoreOutput indicating where to store blocks and what
            was evicted, or None if blocks cannot be allocated.
        """
        return self.prepare_store(block_hashes)

    def finalize_blocks(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """
        Finalize previously allocated blocks, making them available.

        This is a tier-agnostic alias for complete_store() that makes the
        intent clear when used by a tiered manager during promotion
        (secondary→primary transfer).

        Args:
            block_hashes: the hashes identifying the blocks.
            success: whether the blocks were successfully written.
        """
        self.complete_store(block_hashes, success)

    def protect_blocks(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Protect blocks from eviction and return LoadStoreSpec for reading.

        This is a tier-agnostic alias for prepare_load() that makes the
        intent clear when used by a tiered manager during cascade
        (primary→secondary transfer). Increments ref_cnt to prevent
        eviction during async operations.

        Args:
            block_hashes: the hashes identifying the blocks.

        Returns:
            A LoadStoreSpec that can be used to read the blocks.
        """
        return self.prepare_load(block_hashes)

    def unprotect_blocks(self, block_hashes: Iterable[BlockHash]):
        """
        Unprotect blocks, allowing eviction again.

        This is a tier-agnostic alias for complete_load() that makes the
        intent clear when used by a tiered manager during cascade
        (primary→secondary transfer). Decrements ref_cnt after async
        operations complete.

        Args:
            block_hashes: the hashes identifying the blocks.
        """
        self.complete_load(block_hashes)


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers cannot directly access GPU memory. All data transfers
    must go through the primary tier (implemented as CPU in current version):
      - Store: GPU → primary → secondary  (cascade)
      - Load:  secondary → primary → GPU  (promotion)

    IMPORTANT: All methods run in the Scheduler process and must be
    lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished() polls for completion.
    """

    @abstractmethod
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """
        Check which blocks exist in this secondary tier.

        Args:
            block_hashes: Block hashes to look up.

        Returns:
            Number of consecutive blocks (from start) that are present and ready,
            or None if blocks are being transferred (retry later).
        """
        pass

    @abstractmethod
    def submit_store(
        self,
        job_id: JobId,
        block_hashes: Iterable[BlockHash],
        primary_load_spec: LoadStoreSpec,
    ) -> PrepareStoreOutput | None:
        """
        Submit an async job to store blocks from the primary tier to this
        secondary tier.

        This method is lightweight: it allocates metadata and submits the
        transfer job, but does NOT perform the actual data transfer on the
        calling thread.

        The caller (TieredOffloadingManager) must have already called
        primary.prepare_load(block_hashes) to obtain primary_load_spec and
        to increment ref_cnt on those blocks. ref_cnt will be decremented
        when get_finished() reports this job_id as complete and
        primary.complete_load() is called.

        This method is responsible for:
          1. Filtering out blocks already present in this secondary tier
          2. Evicting blocks from this secondary tier if needed (secondary
             tiers are responsible for their own evictions)
          3. Allocating space in this secondary tier
          4. Submitting the async transfer: primary → secondary

        Args:
            job_id: Unique identifier for this transfer job.
            block_hashes: Blocks to store.
            primary_load_spec: Spec for reading blocks from the primary tier
                               (obtained via primary.prepare_load()).

        Returns:
            PrepareStoreOutput describing which blocks will be stored and
            what was evicted, or None if the store cannot proceed.
        """
        pass

    @abstractmethod
    def submit_load(
        self,
        job_id: JobId,
        block_hashes: Iterable[BlockHash],
        primary_store_spec: LoadStoreSpec,
    ) -> LoadStoreSpec | None:
        """
        Submit an async job to load blocks from this secondary tier to the
        primary tier.

        This method is lightweight: it marks blocks as in-flight and submits
        the transfer job, but does NOT perform the actual data transfer on
        the calling thread.

        The caller (TieredOffloadingManager) must have already called
        primary.prepare_store(block_hashes) to obtain primary_store_spec and
        to allocate space in the primary tier. When get_finished() reports
        this job_id as complete, primary.complete_store() is called to make
        the blocks available for GPU loads.

        Args:
            job_id: Unique identifier for this transfer job.
            block_hashes: Blocks to load.
            primary_store_spec: Spec for writing blocks into the primary tier
                                (obtained via primary.prepare_store()).

        Returns:
            LoadStoreSpec for reading from this secondary tier, or None if
            the load cannot proceed.
        """
        pass

    @abstractmethod
    def get_finished(self) -> Iterable[CompletedJob]:
        """
        Poll for completed async jobs (both loads and stores).

        This is the mechanism by which the TieredOffloadingManager learns
        that a transfer has completed and can:
          - Call primary.complete_load() to decrement ref_cnt (for stores)
          - Call primary.complete_store() to make blocks loadable (for loads)

        Returns:
            Iterable of CompletedJob objects for all jobs that have
            completed since the last call.
        """
        pass

    def touch(self, block_hashes: Iterable[BlockHash]):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            block_hashes: Blocks to mark as recently used.
        """
        return

    @abstractmethod
    def get_tier_name(self) -> str:
        """
        Get the name of this tier (e.g., "Storage", "Network").

        Returns:
            Tier name string.
        """
        pass
