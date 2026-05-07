

import os
from state import JobState
try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    raise ImportError(
        "FileSystemTierManagerNixl requires the 'nixl' package. "
        "Install it with: pip install nixl"
    ) from e

def _run_nixl_store(
    agent,
    dram_handle,
    block_size: int,
    view_addr: int,
    block_idx: int,
    dest_path: str,
    state: JobState,
) -> None:
    """
    Store callback: write one KV block via NIXL WRITE transfer.
    
    Writes to a temp file then atomically replaces the destination.
    Checks if block exists first to avoid redundant writes.
    Uses pre-registered DRAM handle to avoid repeated registration overhead.
    """
    tmp_fname = dest_path.replace('.bin', '.tmp')
    fd = -1
    xfer_handle = None
    file_handle = None
    try:
        # Check if block already exists to avoid redundant writes
        if os.path.exists(dest_path):
            state.task_done(True)
            return
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        fd = os.open(
            tmp_fname,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )

        block_addr = view_addr + block_idx * block_size
        local_descs = [(block_addr, block_size, 0)]
        remote_descs = [(0, block_size, fd, "")]

        # Use pre-registered DRAM handle instead of registering again
        local_dl = agent.get_xfer_descs(local_descs, mem_type="DRAM")
        file_handle = agent.register_memory(
            remote_descs, mem_type="FILE", backends=["POSIX"]
        )
        xfer_handle = agent.initialize_xfer(
            operation="WRITE",
            local_descs=local_dl,
            remote_descs=file_handle.trim(),
            remote_agent=agent.name,
            backends=["POSIX"],
        )
        agent.transfer(xfer_handle)

        while True:
            xfer_state = agent.check_xfer_state(xfer_handle)
            if xfer_state == "DONE":
                break
            if xfer_state == "ERR":
                raise RuntimeError(
                    f"NIXL WRITE transfer failed for block {block_idx} -> {dest_path}"
                )

        os.close(fd)
        fd = -1
        os.replace(tmp_fname, dest_path)
        state.task_done(True)

    except Exception as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        state.task_done(False, exc)
    finally:
        if xfer_handle is not None:
            agent.release_xfer_handle(xfer_handle)
        if file_handle is not None:
            agent.deregister_memory(file_handle)


def _run_nixl_load(
    agent,
    dram_handle,
    block_size: int,
    view_addr: int,
    block_idx: int,
    source_path: str,
    state: JobState,
) -> None:
    """
    Load callback: read one KV block via NIXL READ transfer.
    Uses pre-registered DRAM handle to avoid repeated registration overhead.
    """
    fd = -1
    xfer_handle = None
    file_handle = None
    try:
        fd = os.open(source_path, os.O_RDONLY | os.O_DIRECT)

        block_addr = view_addr + block_idx * block_size
        local_descs = [(block_addr, block_size, 0)]
        remote_descs = [(0, block_size, fd, "")]

        # Use pre-registered DRAM handle instead of registering again
        local_dl = agent.get_xfer_descs(local_descs, mem_type="DRAM")
        file_handle = agent.register_memory(
            remote_descs, mem_type="FILE", backends=["POSIX"]
        )
        xfer_handle = agent.initialize_xfer(
            operation="READ",
            local_descs=local_dl,
            remote_descs=file_handle.trim(),
            remote_agent=agent.name,
            backends=["POSIX"],
        )
        agent.transfer(xfer_handle)

        while True:
            xfer_state = agent.check_xfer_state(xfer_handle)
            if xfer_state == "DONE":
                break
            if xfer_state == "ERR":
                raise RuntimeError(
                    f"NIXL READ transfer failed for block {block_idx} <- {source_path}"
                )

        os.close(fd)
        fd = -1
        state.task_done(True)

    except Exception as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        state.task_done(False, exc)
    finally:
        if xfer_handle is not None:
            agent.release_xfer_handle(xfer_handle)
        if file_handle is not None:
            agent.deregister_memory(file_handle)
