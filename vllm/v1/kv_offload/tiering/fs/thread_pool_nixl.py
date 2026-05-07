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

class NixlDualQueuePool:
    """
    Thread pool with two task queues (read and write) and two thread groups.

    Read-priority threads drain the read queue first, then fall back to the
    write queue.  Write-priority threads do the reverse.  Both queues share
    a single condition variable (same as the C++ DualQueueThreadPool).

    Each thread is assigned a dedicated NIXL agent at creation time and passes
    it to every task it runs (NIXL agents are not thread-safe).

    Tasks submitted to this pool must accept two positional arguments:
        task(agent, dram_handle) -> None
    where dram_handle is the pre-registered DRAM memory handle for that agent.
    """

    def __init__(
        self,
        n_read_prio: int,
        n_write_prio: int,
        make_agent,  # callable(thread_index: int) -> nixl_agent
        thread_name_prefix: str = "vllm_nixl_dq",
    ) -> None:
        self._read_q: collections.deque = collections.deque()
        self._write_q: collections.deque = collections.deque()
        self._cv = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        
        # Store agents and their DRAM handles (set later via register_dram_buffer)
        self._agents: list = []
        self._dram_handles: list = []

        # Create all agents upfront so each thread gets its own.
        agents = [make_agent(i) for i in range(n_read_prio + n_write_prio)]
        self._agents = agents

        for i in range(n_read_prio):
            t = threading.Thread(
                target=self._worker,
                args=(True, i),
                name=f"{thread_name_prefix}_r{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_prio):
            t = threading.Thread(
                target=self._worker,
                args=(False, n_read_prio + i),
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

    def register_dram_buffer(self, view_addr: int, view_size: int) -> None:
        """
        Register the DRAM buffer with all agents. Must be called after pool
        creation but before any tasks are enqueued.
        """
        self._dram_handles = []
        for agent in self._agents:
            dram_handle = agent.register_memory(
                [(view_addr, view_size, 0, "")],
                mem_type="DRAM",
                backends=["POSIX"]
            )
            self._dram_handles.append(dram_handle)

    def shutdown(self, wait: bool = True) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if wait:
            for t in self._threads:
                t.join()
        
        # Deregister DRAM handles on shutdown
        for agent, dram_handle in zip(self._agents, self._dram_handles):
            if dram_handle is not None:
                try:
                    agent.deregister_memory(dram_handle)
                except Exception:
                    pass

    def _worker(self, read_priority: bool, agent_idx: int) -> None:
        agent = self._agents[agent_idx]
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
            
            # Pass both agent and its pre-registered DRAM handle to the task
            dram_handle = self._dram_handles[agent_idx] if self._dram_handles else None
            task(agent, dram_handle)
