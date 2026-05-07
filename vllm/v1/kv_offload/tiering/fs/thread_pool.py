# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import collections
import threading
"""
Thread pool:
    Two queues (read, write) and two sets of threads:
      - Read-priority threads: drain the read queue first, then the write queue.
      - Write-priority threads: drain the write queue first, then the read queue.
    Load jobs are enqueued to the read queue; store jobs to the write queue.
"""


class DualQueueThreadPool:
    """
    Thread pool with two task queues (read and write) and two thread groups.

    Read-priority threads drain the read queue first, then fall back to the
    write queue.  Write-priority threads do the reverse.  Both queues share
    a single condition variable (same as the C++ DualQueueThreadPool).
    """

    def __init__(
        self,
        n_read_prio: int,
        n_write_prio: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._read_q: collections.deque = collections.deque()
        self._write_q: collections.deque = collections.deque()
        self._cv = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []

        for i in range(n_read_prio):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_r{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_prio):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_w{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_read(self, fn) -> None:
        """Enqueue a read task (high-priority for read-priority threads)."""
        with self._cv:
            self._read_q.append(fn)
            self._cv.notify()

    def enqueue_write(self, fn) -> None:
        """Enqueue a write task (high-priority for write-priority threads)."""
        with self._cv:
            self._write_q.append(fn)
            self._cv.notify()

    def shutdown(self, wait: bool = True) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, read_priority: bool) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: self._stop or self._read_q or self._write_q
                )
                if self._stop and not self._read_q and not self._write_q:
                    return
                primary   = self._read_q  if read_priority else self._write_q
                secondary = self._write_q if read_priority else self._read_q
                task = primary.popleft() if primary else secondary.popleft()
            task()
