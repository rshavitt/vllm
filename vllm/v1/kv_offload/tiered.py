# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TierOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieredOffloadingManager with a CPU-based primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: "lru" or
    "arc" (default: "lru")
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "dummy", "storage", "network")
      - tier_name: (optional) Name for this tier (default: based on type)
      - max_blocks: (optional) Maximum blocks for this tier
      - simulate_async: (optional) For dummy tier: simulate async behavior
        (default: False)
      - ... (other tier-specific config)

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "dummy",
            "tier_name": "TestStorage",
            "max_blocks": 10000,
            "simulate_async": False
        }
    ]
}
"""

from collections.abc import Iterator

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.dummy_secondary_tier import DummySecondaryTier
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.tiered_manager import TieredOffloadingManager
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class TierOffloadingSpec(OffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieredOffloadingManager with:
    - Primary tier: CPU-based (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The primary tier has direct GPU access and serves as the gateway for all
    GPU↔offload operations. Secondary tiers cannot directly access GPU memory
    and must coordinate with the primary tier for data transfers.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # Validate required configuration
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config "
                "for TierOffloadingSpec"
            )

        # Calculate kv_bytes_per_offloaded_block (same as CPUOffloadingSpec)
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_cpu_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # Primary tier eviction policy
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        if self.eviction_policy not in ["lru", "arc"]:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. "
                f"Supported policies: lru, arc"
            )

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        # Scheduler-side
        self._manager: TieredOffloadingManager | None = None

        # Worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

    def _create_secondary_tier(self, tier_config: dict):
        """
        Create a secondary tier from configuration.

        Args:
            tier_config: Dictionary with tier configuration

        Returns:
            SecondaryTierManager instance

        Raises:
            ValueError: If tier type is unknown or configuration is invalid
        """
        tier_type = tier_config.get("type")
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")

        if tier_type == "dummy":
            # DummySecondaryTier for testing
            tier_name = tier_config.get("tier_name", "DummyStorage")
            max_blocks = tier_config.get("max_blocks", 10000)
            simulate_async = tier_config.get("simulate_async", False)

            return DummySecondaryTier(
                tier_name=tier_name,
                max_blocks=max_blocks,
                simulate_async=simulate_async,
            )
        else:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type}. Supported types: dummy"
            )

    def get_manager(self) -> OffloadingManager:
        """
        Get the TieredOffloadingManager.

        Creates a TieredOffloadingManager with:
        - Primary tier: CPU-based (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieredOffloadingManager instance
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # Create primary tier (CPU-based)
            cpu_backend = CPUBackend(
                block_size=self.offloaded_block_size, num_blocks=self.num_cpu_blocks
            )

            if self.eviction_policy == "lru":
                primary_tier = LRUOffloadingManager(
                    backend=cpu_backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                primary_tier = ARCOffloadingManager(
                    backend=cpu_backend, enable_events=enable_events
                )
            else:
                raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

            # Create secondary tiers
            secondary_tiers = []
            for tier_config in self.secondary_tier_configs:
                try:
                    tier = self._create_secondary_tier(tier_config)
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier: %s (type: %s)",
                        tier.get_tier_name(),
                        tier_config.get("type"),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create secondary tier from config %s: %s",
                        tier_config,
                        e,
                    )
                    raise

            # Create tiered manager
            self._manager = TieredOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
                enable_events=enable_events,
            )

            logger.info(
                "Created TieredOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_cpu_blocks,
                len(secondary_tiers),
            )

        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        Get offloading handlers for GPU↔CPU transfers.

        Note: Secondary tier transfers are handled internally by the
        TieredOffloadingManager and do not require separate handlers here.
        The handlers returned are for GPU↔primary tier (CPU) transfers only.

        Args:
            kv_caches: Dictionary of layer_name -> gpu_kv_cache tensor
            attn_backends: Dictionary of layer_name -> AttentionBackend

        Yields:
            Tuples of (src_type, dst_type, offloading_handler) for GPU↔CPU
        """
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "TierOffloadingSpec is currently only supported on CUDA-alike GPUs"
                )

            # Create handlers for GPU↔CPU transfers
            # (same as CPUOffloadingSpec since primary tier is CPU-based)
            self._handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


# Made with Bob
