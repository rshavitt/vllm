# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Adaptive prefill_ratio controller for queue balancing.

A request routed to the prefill queue finishes after
``prefill_len / R_prefill``; one routed to the load queue after
``load_len / R_load``. Both finish together when

    prefill_len / load_len == R_prefill / R_load == prefill_ratio

so the ratio is the quotient of the two queues' *capacities*: the rate each
drains while backlogged. Realized throughput is not a capacity (it equals
demand when a queue is idle, and demand is set by the ratio itself), so each
rate is sampled only while its queue is busy, and an estimate that gets no
busy time is held rather than decayed.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

_MIN_RATIO = 0.05
_MAX_RATIO = 100.0
# A gap between steps longer than this means the engine idled, not computed.
_MAX_STEP_GAP_S = 1.0


class _RateEstimator:
    """EMA of blocks/s, fed only with samples taken while the queue is busy."""

    def __init__(self, alpha: float, min_busy_s: float) -> None:
        self._alpha = alpha
        self._min_busy_s = min_busy_s
        self._blocks = 0.0
        self._busy_s = 0.0
        self.rate: float | None = None

    def add(self, blocks: float, busy_s: float) -> bool:
        """Accumulate a busy sample; return True if the estimate updated."""
        self._blocks += blocks
        self._busy_s += busy_s
        if self._busy_s < self._min_busy_s:
            return False
        sample = self._blocks / self._busy_s
        self._blocks = 0.0
        self._busy_s = 0.0
        self.rate = (
            sample
            if self.rate is None
            else self._alpha * sample + (1 - self._alpha) * self.rate
        )
        return True


class CapacityRatioController:
    def __init__(
        self,
        tokens_per_block: int,
        ema_alpha: float = 0.5,
        min_busy_s: float = 0.5,
        init_ratio: float = 1.0,
    ) -> None:
        self.ratio: float = init_ratio
        self._tokens_per_block = tokens_per_block
        self._prefill = _RateEstimator(ema_alpha, min_busy_s)
        self._load = _RateEstimator(ema_alpha, min_busy_s)
        self._prev_step_time: float | None = None
        self._prev_compute_bound = False

    def on_schedule_end(
        self,
        now: float,
        num_prefill_tokens: int,
        compute_bound: bool,
        load_blocks: int,
        load_busy_s: float,
    ) -> None:
        """Feed one scheduler step.

        Args:
            now: Monotonic time of this step.
            num_prefill_tokens: Prompt tokens scheduled this step.
            compute_bound: Whether the step filled the token budget.
            load_blocks: Load blocks completed while the load queue was
                saturated since the previous call.
            load_busy_s: Seconds the load queue was saturated since the
                previous call.
        """
        updated = False
        if load_busy_s > 0:
            updated |= self._load.add(load_blocks, load_busy_s)

        # Step duration is the gap since the previous step; it is only the
        # prefill drain time when both steps were compute-bound.
        prev = self._prev_step_time
        if (
            prev is not None
            and compute_bound
            and self._prev_compute_bound
            and num_prefill_tokens > 0
            and now - prev < _MAX_STEP_GAP_S
        ):
            updated |= self._prefill.add(
                num_prefill_tokens / self._tokens_per_block, now - prev
            )
        self._prev_step_time = now
        self._prev_compute_bound = compute_bound

        if not updated:
            return
        if self._prefill.rate is None or self._load.rate is None:
            return
        old = self.ratio
        self.ratio = min(
            max(self._prefill.rate / self._load.rate, _MIN_RATIO), _MAX_RATIO
        )
        logger.info(
            "Adaptive ratio: %.2f -> %.2f (prefill_cap=%.1f blk/s, "
            "load_cap=%.1f blk/s)",
            old,
            self.ratio,
            self._prefill.rate,
            self._load.rate,
        )
