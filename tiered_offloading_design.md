# Tiered KV Cache Offloading Architecture Design

## Executive Summary

This document presents a comprehensive design for extending vLLM's KV cache offloading system from single-tier (GPU ↔ primary tier) to multi-tier (GPU ↔ primary tier ↔ secondary tiers). The primary tier is currently implemented using CPU memory. The design maintains backward compatibility with the existing [`OffloadingManager`](vllm/v1/kv_offload/abstract.py:159) API while introducing new abstractions for secondary tiers.

## Terminology: Primary Tier vs CPU

**Important:** Throughout this document, "primary tier" refers to an **architectural abstraction** - the tier that has direct access to GPU memory and serves as the gateway for all GPU↔offload operations.

In the current implementation, the primary tier is realized using **CPU memory** via CPU-based managers like [`LRUOffloadingManager`](vllm/v1/kv_offload/lru_manager.py:16) or [`ARCOffloadingManager`](vllm/v1/kv_offload/arc_manager.py:16). However, the architecture is designed to support alternative primary tier implementations in the future.

When we refer to "CPU" in this document, we are discussing the specific implementation choice, not the architectural role.

---

**Key Design Principles:**
1. **Always offload to all tiers** — When a block is stored to the primary tier, it is cascaded to ALL secondary tiers
2. **Primary tier is the gateway** — Only the primary tier can directly access GPU memory (currently implemented using CPU memory)
3. **Staged promotion** — Blocks in secondary tiers must be promoted to the primary tier before GPU can access them
4. **Transparent retry mechanism** — Return `None` from `lookup()` to signal "data is being promoted, try later"
5. **Lightweight Scheduler methods** — All `SecondaryTierManager` methods run in the Scheduler process and must be non-blocking; actual data transfers are submitted asynchronously via `submit_load()` / `submit_store()`
6. **`ref_cnt` as eviction protection** — `primary.prepare_load()` increments `ref_cnt`, protecting blocks from eviction until `complete_load()` is called
7. **Secondary tiers own their evictions** — Each secondary tier is responsible for managing its own eviction policy

---

## 1. Current Architecture Analysis

### 1.1 Existing Components

**Core Abstractions:**
- [`OffloadingManager`](vllm/v1/kv_offload/abstract.py:159) — Scheduler-side interface for managing offloaded blocks
- [`Backend`](vllm/v1/kv_offload/backend.py:37) — Allocates storage and provides load/store specs
- [`LoadStoreSpec`](vllm/v1/kv_offload/abstract.py:37) — Worker-side metadata for actual data transfer
- [`BlockStatus`](vllm/v1/kv_offload/backend.py:11) — Tracks block state (ready/not-ready, ref count)
- [`PrepareStoreOutput`](vllm/v1/kv_offload/abstract.py:53) — Output of `prepare_store()`: blocks to store, store spec, evicted blocks

**Existing Implementations:**
- [`LRUOffloadingManager`](vllm/v1/kv_offload/lru_manager.py:16) — LRU eviction policy
- [`ARCOffloadingManager`](vllm/v1/kv_offload/arc_manager.py:16) — Adaptive Replacement Cache policy
- [`CPUBackend`](vllm/v1/kv_offload/backends/cpu.py:20) — CPU memory backend

**Current Data Flow:**
```
GPU ←→ primary tier (via OffloadingManager + CPUBackend)
     └─ Currently implemented using CPU memory
```

### 1.2 The `ref_cnt` Protection Mechanism

The [`BlockStatus`](vllm/v1/kv_offload/backend.py:11) in the primary tier tracks a `ref_cnt` for each block. This counter is the primary protection against eviction:

- **Incremented** by [`prepare_load()`](vllm/v1/kv_offload/abstract.py:178) — protects a block from being evicted while it is being read or while it is the source for a secondary-tier store
- **Decremented** by [`complete_load()`](vllm/v1/kv_offload/abstract.py:204) — releases the protection, allowing the block to be evicted again

This mechanism is critical for the tiered design: when cascading a block from the primary tier to a secondary tier, `prepare_load()` must be called on the primary tier to pin the block in primary tier memory for the duration of the transfer. `complete_load()` is called (via `get_finished()`) once the async transfer completes.

### 1.3 Extension Points

The architecture can be extended at two levels:

1. **Manager Level** — Create `TieredOffloadingManager` implementing [`OffloadingManager`](vllm/v1/kv_offload/abstract.py:159)
2. **Secondary Tier Level** — Create `SecondaryTierManager` implementations (Storage, Network, etc.)

---

## 2. SecondaryTierManager API Specification

### 2.1 Overview

[`SecondaryTierManager`](vllm/v1/kv_offload/abstract.py:69) is an abstract class for managing non-primary tiers. Unlike [`OffloadingManager`](vllm/v1/kv_offload/abstract.py:159), it cannot directly access GPU memory and must coordinate with the primary tier (currently CPU-based).

**Critical constraint:** All `SecondaryTierManager` methods are called from the **Scheduler process** and must be **lightweight and non-blocking**. They must not perform actual data transfers on the calling thread. Instead, `submit_load()` and `submit_store()` accept a `job_id` parameter and submit async jobs for tracking.

### 2.2 Relationship Between `submit_store()` and `primary.prepare_load()`

When the `TieredOffloadingManager` cascades a block from the primary tier to a secondary tier:

1. **`primary.prepare_load(block_hashes)`** is called to obtain the [`LoadStoreSpec`](vllm/v1/kv_offload/abstract.py:37) describing where the blocks live in primary tier memory. This also **increments `ref_cnt`** on those blocks, protecting them from eviction for the duration of the transfer.
2. **`secondary.submit_store(job_id, block_hashes, primary_load_spec)`** is called with the spec obtained above, submitting an async transfer job.
3. When `get_finished()` reports the job as complete, **`primary.complete_load(block_hashes)`** is called to **decrement `ref_cnt`**, releasing the eviction protection.

This means `primary.prepare_load()` serves a dual purpose:
- **For GPU loads**: provides the spec for primary→GPU transfer
- **For secondary-tier stores**: provides the spec for primary→secondary transfer AND pins the blocks in primary tier memory

When there are multiple secondary tiers, `primary.prepare_load()` must be called **once per secondary tier** to correctly increment `ref_cnt` for each pending transfer.

### 2.3 API Definition

```python
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec, PrepareStoreOutput

JobId = int

@dataclass
class CompletedJob:
    """Result of a completed async transfer job."""
    job_id: JobId
    block_hashes: list[BlockHash]
    is_store: bool  # True if primary→secondary, False if secondary→primary
    success: bool


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
```

### 2.4 Key Design Decisions

**Why `submit_` prefix instead of `load`/`store`?**
- Makes it explicit that the operation is asynchronous and non-blocking
- Distinguishes the submission step from the completion step (`get_finished()`)
- Consistent with the pattern used in [`OffloadingWorker.transfer_async()`](vllm/v1/kv_offload/worker/worker.py)

**Why pass `job_id` into `submit_store()` / `submit_load()`?**
- Provides a unique identifier for tracking async jobs
- Secondary tier returns this `job_id` in `get_finished()` along with the block_hashes
- Enables correlation between job submission and completion
- Mirrors the pattern in [`OffloadingConnectorWorker`](vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:558)

**Why does `submit_store()` receive `primary_load_spec`?**
- The spec is obtained by calling `primary.prepare_load()`, which also increments `ref_cnt`
- Passing it in makes the contract explicit: the caller is responsible for pinning the blocks before submitting the store

**Why are secondary tiers responsible for their own evictions?**
- Each secondary tier has its own capacity and eviction policy
- The primary tier does not need to know about secondary tier capacity
- Simplifies the coordination logic in `TieredOffloadingManager`

---

## 3. TieredOffloadingManager Architecture

### 3.1 Class Structure

```python
from collections.abc import Iterable
from enum import Enum
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    OffloadingManager,
    LoadStoreSpec,
    PrepareStoreOutput,
    OffloadingEvent,
    SecondaryTierManager,
)

JobId = int


class TieredOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a primary tier (with GPU access, currently
    CPU-based) and zero or more secondary tiers (Storage, Network, etc.) to
    provide hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion via get_finished()
      - Secondary tiers return CompletedJob objects containing all necessary information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: OffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False
    ):
        self.primary_tier = primary_tier
        self.secondary_tiers = secondary_tiers or []
        
        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def _next_job_id(self) -> JobId:
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id
```

### 3.2 `prepare_store()` and the `get_finished()` Call

A critical design point: **`prepare_store()` must call `get_finished()` on all secondary tiers before calling `primary.prepare_store()`**. This ensures that:

1. Any previously completed async transfers have their `ref_cnt` decremented (via `primary.complete_load()`)
2. Blocks that have been successfully cascaded to secondary tiers are marked as `BOTH`
3. The primary tier has the most up-to-date view of which blocks are pinned, enabling accurate eviction decisions

```python
def prepare_store(
    self, block_hashes: Iterable[BlockHash]
) -> PrepareStoreOutput | None:
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
```

### 3.3 `complete_store()` and Secondary Tier Cascading

`complete_store()` is called by the connector when the GPU→Primary transfer finishes. At this point, the blocks are available in the primary tier and ready to be cascaded to secondary tiers.

**This is where secondary tier cascading happens** — after blocks are confirmed to be in the primary tier.

```python
def complete_store(
    self, block_hashes: Iterable[BlockHash], success: bool = True
):
    # Step 1: Complete store in primary tier (makes blocks loadable from primary)
    self.primary_tier.complete_store(block_hashes, success)
    
    if not success:
        # If GPU→Primary transfer failed, don't cascade to secondary tiers
        return
    
    # Step 2: Cascade to ALL secondary tiers
    # For each secondary tier, call primary.prepare_load() to get the
    # LoadStoreSpec AND to increment ref_cnt (protecting blocks from eviction
    # during the async transfer). One prepare_load() call per secondary tier.
    for tier_idx, tier in enumerate(self.secondary_tiers):
        primary_load_spec = self.primary_tier.prepare_load(block_hashes)
        job_id = self._next_job_id()
        result = tier.submit_store(
            job_id,
            block_hashes,
            primary_load_spec
        )
    
    # Note: The async transfers are now in flight.
    # Their completion is tracked via get_finished() / _process_finished_jobs().
```

### 3.4 `_process_finished_jobs()` — The Completion Handler

This method polls all secondary tiers for completed jobs and updates state accordingly:

```python
def _process_finished_jobs(self):
    for tier_idx, tier in enumerate(self.secondary_tiers):
        for completed in tier.get_finished():
            if completed.is_store:
                # primary→secondary transfer completed.
                # Decrement ref_cnt on primary blocks.
                self.primary_tier.complete_load(completed.block_hashes)
            else:
                # secondary→primary transfer (promotion) completed.
                # Make blocks available in primary tier.
                self.primary_tier.complete_store(completed.block_hashes, completed.success)
```

---

## 4. Tier Coordination and Routing Logic

### 4.1 Lookup Flow

**Algorithm:**

1. **Primary Tier Check**
   ```python
   primary_hits = self.primary_tier.lookup(block_hashes)
   if primary_hits == len(block_hashes):
       return primary_hits  # All blocks in primary, done
   ```

2. **Transfer Check**
   ```python
   # Check if any remaining blocks are in-flight
   # Note: We rely on the secondary tier's lookup() returning None for in-flight blocks
   # This avoids the need to track in-flight state in TieredOffloadingManager
   ```

3. **Secondary Tier Check**
   ```python
   secondary_hits = self._lookup_secondary_tiers(remaining_blocks)
   if secondary_hits > 0:
       self._initiate_promotion(remaining_blocks[:secondary_hits])
       return None  # Promotion initiated, retry later
   ```

4. **Return Result**
   ```python
   return primary_hits  # No more blocks found
   ```

### 4.2 Store Flow (Cascade to ALL Tiers)

```
Scheduler calls prepare_store(block_hashes)
    │
    ├─ 1. _process_finished_jobs()          ← poll secondary tiers first
    │       └─ complete_load() on primary   ← decrement ref_cnt
    │
    ├─ 2. primary.prepare_store()           ← allocate primary tier space, evict if needed
    │
    └─ 3. For EACH secondary tier:
            ├─ primary.prepare_load()       ← get LoadStoreSpec + increment ref_cnt
            └─ tier.submit_store(job_id, ..., primary_load_spec)
                    └─ async: primary → secondary
    
    Worker executes GPU → primary transfer (using primary store_spec)

Scheduler calls complete_store(block_hashes)
    └─ primary.complete_store()             ← blocks now loadable from primary

Later: secondary tier completes async transfer
    └─ get_finished() → _process_finished_jobs()
            └─ primary.complete_load()      ← decrement ref_cnt
```

### 4.3 Load Flow (Promotion from Secondary to Primary)

```
Scheduler calls lookup(block_hashes)
    └─ blocks found in secondary tier
            ├─ primary.prepare_store()      ← allocate primary tier space for promotion
            └─ tier.submit_load(job_id, block_hashes, primary_store_spec)
                    └─ async: secondary → primary

lookup() returns None (retry later)

Later: secondary tier completes async transfer
    └─ get_finished() → _process_finished_jobs()
            └─ primary.complete_store()     ← blocks now loadable from primary

Next lookup() call:
    └─ primary.lookup() returns hits        ← blocks now in primary
```

### 4.4 Primary Tier's `prepare_load()` Dual Role

[`primary.prepare_load()`](vllm/v1/kv_offload/abstract.py:178) serves two distinct purposes in the tiered design:

| Caller | Purpose | Effect on `ref_cnt` |
|--------|---------|---------------------|
| `TieredOffloadingManager.prepare_load()` | Get spec for primary→GPU transfer | Incremented; decremented by `complete_load()` after GPU load |
| `TieredOffloadingManager.complete_store()` (cascade) | Get spec for primary→secondary transfer | Incremented; decremented by `_process_finished_jobs()` after async store completes |

When there are N secondary tiers, `primary.prepare_load()` is called N times for the same set of blocks (in `complete_store()`), incrementing `ref_cnt` by N. Each completed secondary-tier store decrements it by 1.

---

## 5. Architecture Diagrams

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TieredOffloadingManager                  │
│                  Implements OffloadingManager               │
│  Minimal state: just tracks job_id counter                  │
│  Secondary tiers report completion via CompletedJob         │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Primary Tier   │ │ Secondary      │ │ Secondary      │
│ (CPU impl)     │ │ Tier 1         │ │ Tier 2         │
│ LRU/ARC        │ │ Storage        │ │ Network        │
│ Manager        │ │ Manager        │ │ Manager        │
│                │ │                │ │                │
│ ref_cnt tracks │ │ submit_store() │ │ submit_store() │
│ pinned blocks  │ │ submit_load()  │ │ submit_load()  │
│                │ │ get_finished() │ │ get_finished() │
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ CPUBackend     │ │ StorageBackend │ │ NetworkBackend │
└────────────────┘ └────────────────┘ └────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ CPU Memory     │ │ Disk Storage   │ │ Remote Storage │
└────────────────┘ └────────────────┘ └────────────────┘
```

### 5.2 Data Flow

```
GPU
 ▲ │
 │ ▼  Direct access only via primary tier
primary tier (CPU implementation)
 ▲ │  ref_cnt protects blocks during async transfers
 │ ▼
Storage (secondary tier 1)   ← submit_store / submit_load
 │
 ▼
Network (secondary tier 2)   ← submit_store / submit_load

Store (offload):  GPU → primary → Storage, primary → Network  (all tiers)
Load (restore):   Storage → primary → GPU  (staged promotion)
```

### 5.3 Sequence Diagram: Store with Cascade

```
Scheduler          TieredManager       Primary          Secondary Tier
    │                    │                │                    │
    │ prepare_store()    │                │                    │
    │───────────────────>│                │                    │
    │                    │ get_finished() │                    │
    │                    │────────────────────────────────────>│
    │                    │<─ CompletedJob (decrement ref_cnt)  │
    │                    │ complete_load()│                    │
    │                    │───────────────>│                    │
    │                    │ prepare_store()│                    │
    │                    │───────────────>│                    │
    │                    │<── store_spec  │                    │
    │<── PrepareStoreOutput               │                    │
    │                    │                │                    │
    │ [Worker: GPU→primary transfer]      │                    │
    │                    │                │                    │
    │ complete_store()   │                │                    │
    │───────────────────>│                │                    │
    │                    │ complete_store()│                   │
    │                    │───────────────>│ (blocks ready)     │
    │                    │                │                    │
    │                    │ *** CASCADE TO SECONDARY TIERS ***  │
    │                    │ prepare_load() │                    │
    │                    │───────────────>│ (ref_cnt++)        │
    │                    │<── load_spec   │                    │
    │                    │ submit_store(job_id, load_spec)     │
    │                    │────────────────────────────────────>│
    │                    │                │  [async: primary→secondary]
    │                    │                │                    │
    │ [later] prepare_store() or lookup() │                    │
    │───────────────────>│                │                    │
    │                    │ get_finished() │                    │
    │                    │────────────────────────────────────>│
    │                    │<─ CompletedJob(success=True)        │
    │                    │ complete_load()│                    │
    │                    │───────────────>│ (ref_cnt--)        │
```

---

## 6. Key Design Decisions Summary

| Aspect | Design Choice | Rationale |
|--------|--------------|-----------|
| Secondary tier store method | `submit_store(job_id, ...)` — async, non-blocking | Keeps Scheduler process responsive; actual transfers happen asynchronously |
| Secondary tier load method | `submit_load(job_id, ...)` — async, non-blocking | Consistent with store; enables parallel transfers |
| Completion tracking | `get_finished()` polls for completed jobs | Decouples job submission from completion; supports multiple in-flight transfers |
| `job_id` parameter | Required in `submit_store()` / `submit_load()` | Unique identifier returned by secondary tier in `CompletedJob` |
| Cascade timing | Happens in `complete_store()` after GPU→Primary completes | Ensures blocks are ready in primary before cascading to secondary tiers |
| `prepare_store()` ordering | Call `get_finished()` first, then primary | Decrements ref_cnt before eviction decisions, enabling accurate capacity assessment |
| `primary.prepare_load()` in cascade | Called once per secondary tier in `complete_store()` | Gets transfer spec AND increments `ref_cnt` to protect blocks during async transfer |
| `ref_cnt` management | Explicitly managed via `prepare_load()` / `complete_load()` | Protects blocks from eviction during async transfers; one increment per secondary tier |
| Secondary tier evictions | Each tier manages its own eviction policy | Decentralized design; tiers are autonomous |
| Offload to all tiers | ALL secondary tiers receive every block stored to primary | Maximizes data availability across the tier hierarchy |

---

## 7. Migration Strategy

### 7.1 Backward Compatibility

**Existing Code (unchanged):**
```python
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend

backend = CPUBackend(block_size=16, num_blocks=1000)
manager = LRUOffloadingManager(backend)
```

**New Code (Multi-Tier):**
```python
from vllm.v1.kv_offload.tiered_manager import TieredOffloadingManager
from vllm.v1.kv_offload.secondary_tiers.storage import StorageSecondaryTier

# Create primary tier (same as before) - CPU-based implementation
cpu_backend = CPUBackend(block_size=16, num_blocks=1000)
primary_tier = LRUOffloadingManager(cpu_backend)

# Create secondary tier(s)
storage_tier = StorageSecondaryTier(
    block_size=16,
    storage_path="/tmp/kv_cache",
    max_blocks=10000
)

# Wrap in tiered manager
manager = TieredOffloadingManager(
    primary_tier=primary_tier,
    secondary_tiers=[storage_tier]
)
```

### 7.2 Implementation Phases

**Phase 1: Core Infrastructure (Weeks 1-2)**
- Update [`SecondaryTierManager`](vllm/v1/kv_offload/abstract.py:69) abstract class with `submit_store()`, `submit_load()`, `get_finished()` API returning `CompletedJob`
- Implement `TieredOffloadingManager` with `_process_finished_jobs()`
- Ensure `prepare_store()` calls `get_finished()` before `primary.prepare_store()`
- Ensure cascade calls `primary.prepare_load()` once per secondary tier

**Phase 2: Dummy Secondary Tier (Week 3)**
- Implement `DummySecondaryTier` for testing
- Add unit tests for `ref_cnt` management, job tracking, and cascade behavior

**Phase 3: Storage Backend (Weeks 4-5)**
- Implement `StorageSecondaryTier` with file-based storage
- Implement async transfer mechanisms using background threads/processes

**Phase 4: Integration (Week 6)**
- Integrate with vLLM scheduler via [`OffloadingConnectorScheduler`](vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:244)
- Performance tuning
- Documentation

---

## 8. Summary

This design provides a comprehensive architecture for multi-tier KV cache offloading that:

1. ✅ Maintains the existing [`OffloadingManager`](vllm/v1/kv_offload/abstract.py:159) API contract
2. ✅ Introduces `SecondaryTierManager` with async `submit_store()` / `submit_load()` / `get_finished()` API returning `CompletedJob`
3. ✅ Implements `TieredOffloadingManager` with minimal state tracking
4. ✅ Supports staged promotion (Secondary → Primary → GPU)
5. ✅ Enables cascade offloading to ALL secondary tiers (GPU → Primary → All Secondaries)
6. ✅ Correctly manages `ref_cnt` via `primary.prepare_load()` / `primary.complete_load()`
7. ✅ Calls `get_finished()` before `primary.prepare_store()` to release pinned blocks
8. ✅ Delegates eviction responsibility to each secondary tier
9. ✅ Maintains backward compatibility
10. ✅ All Scheduler-side methods are lightweight and non-blocking

**Next Steps:**
1. Review and approve this design
2. Update [`SecondaryTierManager`](vllm/v1/kv_offload/abstract.py:69) in code to match `submit_store()` / `submit_load()` / `get_finished()` API with `CompletedJob` return type
3. Update [`TieredOffloadingManager`](vllm/v1/kv_offload/tiered_manager.py:49) to implement `_process_finished_jobs()` using `CompletedJob`
4. Create dummy secondary tier for testing
5. Implement storage backend