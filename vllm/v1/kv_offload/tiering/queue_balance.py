# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Adaptive prefill_ratio controller for queue balancing.

Every request whose lookup found stored blocks is routed either to the load
path (fetch from storage) or to the forced-prefill path (recompute on GPU).
For each path we measure the per-block wait: time from the routing decision
until the request's prefill completes, divided by the request's lookup
blocks. Both paths are measured to the same end point so the residual
prefill of new tokens is counted on both sides.

The controller is a multiplicative integral controller that drives the two
waits to equality:

    ratio <- ratio * clamp((W_load / W_prefill) ** gain, 1/max_step, max_step)

Loads slower than prefill -> ratio grows -> more requests forced to prefill,
and vice versa. Unlike a throughput ratio, these waits are not determined by
the ratio itself, so the only fixed point is the balanced one.
"""

import threading
import time
from collections.abc import Collection

from vllm.logger import init_logger

logger = init_logger(__name__)

_MIN_RATIO = 0.05
_MAX_RATIO = 100.0


class _PathStats:
    def __init__(self) -> None:
        # req_id -> (decision time, num blocks)
        self.inflight: dict[str, tuple[float, int]] = {}
        self.wait_sum: float = 0.0
        self.blocks: int = 0
        self.ema: float | None = None

    def complete(self, req_id: str, now: float) -> bool:
        entry = self.inflight.pop(req_id, None)
        if entry is None:
            return False
        start, n = entry
        self.wait_sum += now - start
        self.blocks += n
        return True

    def window_wait(self, now: float) -> float | None:
        """Per-block wait over the window, or None if nothing to measure.

        Without completions, fall back to the oldest in-flight request's age
        per block — a lower bound that keeps the signal alive when a path is
        so congested that nothing completes within a window.
        """
        if self.blocks:
            return self.wait_sum / self.blocks
        if self.inflight:
            return max((now - t) / max(n, 1) for t, n in self.inflight.values())
        return None

    def reset_window(self) -> None:
        self.wait_sum = 0.0
        self.blocks = 0


class WaitBalanceController:
    def __init__(
        self,
        update_interval_s: float = 1.0,
        ema_alpha: float = 0.5,
        init_ratio: float = 1.0,
        gain: float = 0.5,
        max_step: float = 1.5,
    ) -> None:
        self.ratio: float = init_ratio
        self._update_interval_s = update_interval_s
        self._ema_alpha = ema_alpha
        self._gain = gain
        self._max_step = max_step
        self._lock = threading.Lock()
        self._load = _PathStats()
        self._prefill = _PathStats()
        # Requests whose final prefill chunk was scheduled in the previous
        # step; timestamped at the next step, once that chunk has run.
        self._prefill_done_pending: Collection[str] = ()
        self._window_start: float = time.monotonic()

    def on_routed(self, req_id: str, num_blocks: int, forced_prefill: bool) -> None:
        """Record a routing decision. Called from the lookup thread."""
        path = self._prefill if forced_prefill else self._load
        with self._lock:
            if req_id in self._load.inflight or req_id in self._prefill.inflight:
                return
            path.inflight[req_id] = (time.monotonic(), num_blocks)

    def on_schedule_end(self, prefill_done_req_ids: Collection[str]) -> None:
        now = time.monotonic()
        with self._lock:
            for req_id in self._prefill_done_pending:
                if not self._load.complete(req_id, now):
                    self._prefill.complete(req_id, now)
            self._prefill_done_pending = prefill_done_req_ids
            self._maybe_update(now)

    def on_request_finished(self, req_id: str) -> None:
        with self._lock:
            self._load.inflight.pop(req_id, None)
            self._prefill.inflight.pop(req_id, None)

    def _maybe_update(self, now: float) -> None:
        if now - self._window_start < self._update_interval_s:
            return
        self._window_start = now

        load_w = self._load.window_wait(now)
        prefill_w = self._prefill.window_wait(now)
        self._load.reset_window()
        self._prefill.reset_window()

        if load_w is None:
            # No storage-hit traffic: nothing to balance.
            return

        alpha = self._ema_alpha
        for stats, w in ((self._load, load_w), (self._prefill, prefill_w)):
            if w is not None:
                stats.ema = (
                    w if stats.ema is None else alpha * w + (1 - alpha) * (stats.ema)
                )

        old = self.ratio
        if prefill_w is None:
            # Nothing routed to prefill: the path is idle, so probe upward.
            factor = self._max_step
        else:
            assert self._load.ema is not None and self._prefill.ema is not None
            raw = (self._load.ema / max(self._prefill.ema, 1e-9)) ** self._gain
            factor = min(max(raw, 1.0 / self._max_step), self._max_step)
        self.ratio = min(max(old * factor, _MIN_RATIO), _MAX_RATIO)

        logger.info(
            "Adaptive ratio: %.2f -> %.2f (load_wait=%.2f ms/blk, "
            "prefill_wait=%s ms/blk, load_inflight=%d, prefill_inflight=%d)",
            old,
            self.ratio,
            1000.0 * self._load.ema if self._load.ema is not None else 0.0,
            (
                "%.2f" % (1000.0 * self._prefill.ema)
                if prefill_w is not None and self._prefill.ema is not None
                else "n/a"
            ),
            len(self._load.inflight),
            len(self._prefill.inflight),
        )
