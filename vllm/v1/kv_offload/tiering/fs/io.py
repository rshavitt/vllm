# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from state import JobState

def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    state: "JobState",
) -> None:
    """
    Store callback: write one KV block atomically, optionally checking if it exists first.
    
    Writes to a temp file then atomically replaces the destination.
    Uses O_DIRECT with page-aligned bounce buffer for kernel bypass.
    """
    tmp_path = dest_path.replace('.bin', '.tmp')
    try:
        # Check if block already exists to avoid redundant writes
        if os.path.exists(dest_path):
            state.task_done(True)
            return
        
        # Write block atomically
        _ensure_dirs(dest_path)
        slice = buffer[offset: offset + block_size]
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT,
            0o644,
        )
        try:
            os.write(fd, slice)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
        state.task_done(True)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        state.task_done(False, exc)


def _load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    state: "JobState",
) -> None:
    """
    Load callback: read one KV block from disk.
    """
    try:
        fd = os.open(source_path, os.O_RDONLY | os.O_DIRECT)
        slice = view[offset: offset + block_size]
        try:
            os.readv(fd, [slice])
        finally:
            os.close(fd)
        state.task_done(True)
    except Exception as exc:
        state.task_done(False, exc)