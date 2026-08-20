# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging
import mmap
import os
import random
import threading
import time

from vllm.v1.kv_offload.base import OffloadKey

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# Thread-local storage for unique temporary file suffixes
_thread_local = threading.local()


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def probe_o_direct(directory: str) -> bool:
    """Return whether ``O_DIRECT`` I/O works in *directory*.

    ``O_DIRECT`` is unsupported on some filesystems (e.g. the overlayfs backing
    a container ``/tmp``, older tmpfs, or some NFS mounts), where opening or
    writing a file with it fails with ``EINVAL``. Probe once with an aligned
    single-page write so callers can fall back to buffered I/O instead of
    failing on every block.
    """
    if not O_DIRECT:
        return False
    path = os.path.join(directory, f".o_direct_probe{_get_tmp_suffix()}")
    page = mmap.mmap(-1, mmap.PAGESIZE)
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | O_DIRECT, 0o644)
        try:
            os.write(fd, page)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False
    finally:
        page.close()
        with contextlib.suppress(OSError):
            os.remove(path)


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _validate_offsets(view: memoryview, offsets: list[int], block_size: int) -> None:
    """Raise if any block would read/write past the bounds of `view`.

    Without this, an out-of-range offset silently clips to a shorter (or
    empty) slice instead of failing, since memoryview slicing follows
    Python's slice-clamping semantics rather than raising.
    """
    total_len = len(view.cast("B"))
    for offset in offsets:
        if offset < 0 or offset + block_size > total_len:
            raise ValueError(
                f"block offset {offset} (block_size {block_size}) is out of "
                f"bounds for a buffer of size {total_len}"
            )


def _store_block(
    key: OffloadKey,
    buffer: memoryview,
    offset: int,
    block_size: int,
    block_write_delay_s: float,
    stored_keys,
    stored_keys_lock,
    write_overhead_s: float = 0.0,
) -> None:
    if key in stored_keys:
        return
    time.sleep(block_write_delay_s + write_overhead_s)
    with stored_keys_lock:
        stored_keys.add(key)


def _load_block(
    key: OffloadKey,
    view: memoryview,
    offset: int,
    block_size: int,
    block_read_delay_s: float,
    stored_keys,
    stored_keys_lock,
    read_overhead_s: float = 0.0,
) -> None:
    time.sleep(block_read_delay_s + read_overhead_s)
    with stored_keys_lock:
        stored_keys.add(key)


def batch_store_block(
    keys: list[OffloadKey],
    view: memoryview,
    offsets: list[int],
    block_size: int,
    stored_keys,
    stored_keys_lock,
) -> None:
    _validate_offsets(view, offsets, block_size)
    for key, offset in zip(keys, offsets):
        _store_block(key, view, offset, block_size, 0.0, stored_keys, stored_keys_lock)


def batch_load_block(
    keys: list[OffloadKey],
    view: memoryview,
    offsets: list[int],
    block_size: int,
    stored_keys,
    stored_keys_lock,
) -> None:
    _validate_offsets(view, offsets, block_size)
    for key, offset in zip(keys, offsets):
        _load_block(key, view, offset, block_size, 0.0, stored_keys, stored_keys_lock)
