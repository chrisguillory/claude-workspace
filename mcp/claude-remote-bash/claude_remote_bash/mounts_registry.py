"""Active-mount registry persisted in ``mounts.json``.

CRUD operations are filelock-guarded so concurrent ``crb mount``/``crb umount``
invocations don't corrupt the file. Each entry records the supervisor PID
so ``crb umount`` can SIGTERM the right process.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from cc_lib.schemas import ClosedModel
from cc_lib.types import JsonDatetime
from cc_lib.utils.atomic_write import atomic_write
from filelock import FileLock

from claude_remote_bash.paths import DATA_DIR, MOUNTS_REGISTRY, MOUNTS_REGISTRY_LOCK

__all__ = [
    'MountEntry',
    'MountsRegistry',
    'add_entry',
    'find_by_mountpoint',
    'find_by_supervisor_pid',
    'read_registry',
    'remove_entry',
]


class MountEntry(ClosedModel):
    """One active mount supervised by a detached process."""

    mount_id: str
    """Server-assigned identifier — same value the daemon returned in MountResponse."""

    peer_alias: str
    """mDNS alias of the peer daemon serving the mount."""

    remote_path: str
    """Path as the user typed it (unexpanded ``~`` is preserved for display)."""

    mountpoint: str
    """Absolute local mountpoint path."""

    supervisor_pid: int
    """PID of the detached supervisor that owns this mount."""

    readonly: bool
    """Mount was requested read-only."""

    established_at: JsonDatetime
    """Wall-clock time the supervisor reported ``READY``."""


class MountsRegistry(ClosedModel):
    """All currently-active mounts."""

    mounts: Sequence[MountEntry] = ()


def read_registry() -> MountsRegistry:
    """Read the registry; return an empty registry if the file is absent."""
    with FileLock(MOUNTS_REGISTRY_LOCK):
        if not MOUNTS_REGISTRY.exists():
            return MountsRegistry()
        return MountsRegistry.model_validate_json(MOUNTS_REGISTRY.read_text())


def add_entry(entry: MountEntry) -> None:
    """Append ``entry`` to the registry. Rejects duplicate mountpoints."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_DIR, 0o700)
    with FileLock(MOUNTS_REGISTRY_LOCK):
        current = _read_unlocked()
        if any(m.mountpoint == entry.mountpoint for m in current.mounts):
            raise ValueError(f'mountpoint already in registry: {entry.mountpoint}')
        new = MountsRegistry(mounts=[*current.mounts, entry])
        _write_unlocked(new)


def remove_entry(mountpoint: Path) -> bool:
    """Remove the entry matching ``mountpoint``. Returns True iff an entry was removed."""
    target = str(mountpoint)
    with FileLock(MOUNTS_REGISTRY_LOCK):
        current = _read_unlocked()
        remaining = [m for m in current.mounts if m.mountpoint != target]
        if len(remaining) == len(current.mounts):
            return False
        _write_unlocked(MountsRegistry(mounts=remaining))
        return True


def find_by_mountpoint(mountpoint: Path) -> MountEntry | None:
    """Look up an entry by its mountpoint, or ``None`` if not present."""
    target = str(mountpoint)
    for entry in read_registry().mounts:
        if entry.mountpoint == target:
            return entry
    return None


def find_by_supervisor_pid(pid: int) -> MountEntry | None:
    """Look up an entry by its supervisor PID, or ``None`` if not present."""
    for entry in read_registry().mounts:
        if entry.supervisor_pid == pid:
            return entry
    return None


def _read_unlocked() -> MountsRegistry:
    if not MOUNTS_REGISTRY.exists():
        return MountsRegistry()
    return MountsRegistry.model_validate_json(MOUNTS_REGISTRY.read_text())


def _write_unlocked(registry: MountsRegistry) -> None:
    atomic_write(MOUNTS_REGISTRY, registry.model_dump_json().encode(), mode=0o600)
