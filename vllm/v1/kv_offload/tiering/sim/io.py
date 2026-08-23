# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

from vllm.v1.kv_offload.base import OffloadKey


class BandwidthLimiter:
    """Simulates a shared bandwidth-limited pipe.

    Threads acquire time slots proportional to the bytes they transfer.
    Scheduling is serialized (lock), but waiting happens outside the lock
    so threads sleep in parallel at their assigned slot times.
    """

    def __init__(self, bytes_per_sec: float):
        self._lock = threading.Lock()
        self._bytes_per_sec = bytes_per_sec
        self._next_available = time.monotonic()

    def acquire(self, nbytes: int) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_available - now)
            self._next_available = max(now, self._next_available) + (
                nbytes / self._bytes_per_sec
            )
        if wait > 0:
            time.sleep(wait)

    def reserve_overhead(self, seconds: float) -> None:
        """Push all subsequent acquire() calls back by `seconds`.

        Call this before enqueueing block tasks to model a per-request
        setup cost (seek, connection, metadata lookup) that must complete
        before any data transfer begins.
        """
        with self._lock:
            self._next_available = max(time.monotonic(), self._next_available) + seconds


def _store_block(
    key: OffloadKey,
    buffer: memoryview,
    offset: int,
    block_size: int,
    bandwidth_limiter: BandwidthLimiter,
    stored_keys: set,
    stored_keys_lock: threading.Lock,
) -> None:
    if key in stored_keys:
        return
    bandwidth_limiter.acquire(block_size)
    with stored_keys_lock:
        stored_keys.add(key)


def _load_block(
    key: OffloadKey,
    view: memoryview,
    offset: int,
    block_size: int,
    bandwidth_limiter: BandwidthLimiter,
    stored_keys: set,
    stored_keys_lock: threading.Lock,
) -> None:
    bandwidth_limiter.acquire(block_size)
    with stored_keys_lock:
        stored_keys.add(key)
