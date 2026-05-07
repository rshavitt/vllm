# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(ok) when it finishes.  The scheduler thread
    polls is_done to detect completion.
    """
    __slots__ = ("_total", "_completed", "_success", "_errors", "_lock")

    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0
        self._success = True
        self._errors: list[Exception] = []
        self._lock = threading.Lock()

    def task_done(self, ok: bool, exc: Exception | None = None) -> None:
        with self._lock:
            if not ok:
                self._success = False
                if exc is not None:
                    self._errors.append(exc)
            self._completed += 1

    @property
    def is_done(self) -> bool:
        # _completed is monotonically increasing; reading without a lock is
        # safe for a poll: we may see it one cycle late at worst.
        return self._completed >= self._total

    @property
    def success(self) -> bool:
        return self._success

    @property
    def errors(self) -> list[Exception]:
        return list(self._errors)
