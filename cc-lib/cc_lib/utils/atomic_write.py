"""Atomic file writing for claude-workspace tools."""

from __future__ import annotations

__all__ = [
    'atomic_write',
]

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write(
    target: Path,
    content: bytes,
    *,
    mode: int | None = None,
    reference: Path | None = None,
) -> None:
    """Write content to target atomically via temp file + rename.

    Prevents partial-write corruption: readers never see an incomplete file.
    The temp file is created in the same directory as ``target`` to guarantee
    ``os.replace()`` doesn't cross filesystem boundaries.

    Args:
        target: Destination path. Parent directory is created if needed.
        content: Raw bytes to write.
        mode: Explicit permissions (e.g., ``0o755``). Applied via ``fchmod``
            before the rename so the file is never world-readable by default.
            Mutually exclusive with ``reference``.
        reference: Copy permissions and timestamps from this file via
            ``shutil.copystat``. Useful when replacing an existing file
            in-place (pass ``reference=target``). Mutually exclusive with
            ``mode``.

    Raises:
        ValueError: If both ``mode`` and ``reference`` are provided.
    """
    if mode is not None and reference is not None:
        raise ValueError('mode and reference are mutually exclusive')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=target.parent,
            prefix=f'.{target.name}.',
            suffix='.tmp',
        )
        os.write(fd, content)
        if mode is not None:
            os.fchmod(fd, mode)
        os.close(fd)
        fd = None
        if reference is not None:
            shutil.copystat(reference, tmp_path)
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:  # noqa: SIM105 — explicit try/except preserves cleanup intent in finally block
                os.unlink(tmp_path)
            except OSError:
                pass
