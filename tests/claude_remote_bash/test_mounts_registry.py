"""Empirical tests for the ``mounts_registry`` CRUD surface.

Each test redirects ``MOUNTS_REGISTRY`` and ``MOUNTS_REGISTRY_LOCK`` to a
temp path so the user's real registry isn't touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from claude_remote_bash import mounts_registry as registry_module
from claude_remote_bash import paths as paths_module
from claude_remote_bash.mounts_registry import (
    MountEntry,
    add_entry,
    find_by_mountpoint,
    find_by_supervisor_pid,
    read_registry,
    remove_entry,
)


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the registry to a temp file. Returns its path."""
    registry_path = tmp_path / 'mounts.json'
    lock_path = tmp_path / 'mounts.lock'
    monkeypatch.setattr(paths_module, 'MOUNTS_REGISTRY', registry_path)
    monkeypatch.setattr(paths_module, 'MOUNTS_REGISTRY_LOCK', lock_path)
    monkeypatch.setattr(paths_module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(registry_module, 'MOUNTS_REGISTRY', registry_path)
    monkeypatch.setattr(registry_module, 'MOUNTS_REGISTRY_LOCK', lock_path)
    monkeypatch.setattr(registry_module, 'DATA_DIR', tmp_path)
    return registry_path


def test_read_registry_returns_empty_when_file_missing(isolated_registry: Path) -> None:
    _ = isolated_registry  # monkeypatch is applied; no file should exist yet
    assert len(read_registry().mounts) == 0


def test_add_entry_persists_and_round_trips(isolated_registry: Path) -> None:
    _ = isolated_registry
    entry = _entry()
    add_entry(entry)

    reg = read_registry()
    assert len(reg.mounts) == 1
    assert reg.mounts[0].mount_id == entry.mount_id
    assert reg.mounts[0].mountpoint == entry.mountpoint
    assert reg.mounts[0].supervisor_pid == entry.supervisor_pid


def test_add_entry_rejects_duplicate_mountpoint(isolated_registry: Path) -> None:
    _ = isolated_registry
    add_entry(_entry(mountpoint='/m/a', mount_id='mid-1'))
    with pytest.raises(ValueError, match='mountpoint already in registry'):
        add_entry(_entry(mountpoint='/m/a', mount_id='mid-2'))


def test_add_entry_appends_without_losing_prior_entries(isolated_registry: Path) -> None:
    """Atomic write must not truncate-on-write — second add keeps the first."""
    _ = isolated_registry
    add_entry(_entry(mountpoint='/m/a', mount_id='mid-1', supervisor_pid=111))
    add_entry(_entry(mountpoint='/m/b', mount_id='mid-2', supervisor_pid=222))

    reg = read_registry()
    paths = sorted(m.mountpoint for m in reg.mounts)
    assert paths == ['/m/a', '/m/b']


def test_remove_entry_returns_true_for_existing(isolated_registry: Path) -> None:
    _ = isolated_registry
    add_entry(_entry(mountpoint='/m/x'))
    assert remove_entry(Path('/m/x')) is True
    assert len(read_registry().mounts) == 0


def test_remove_entry_returns_false_for_missing(isolated_registry: Path) -> None:
    _ = isolated_registry
    assert remove_entry(Path('/m/nope')) is False


def test_remove_entry_preserves_siblings(isolated_registry: Path) -> None:
    """Removing one entry shouldn't disturb the others."""
    _ = isolated_registry
    add_entry(_entry(mountpoint='/m/a', mount_id='mid-1'))
    add_entry(_entry(mountpoint='/m/b', mount_id='mid-2'))
    add_entry(_entry(mountpoint='/m/c', mount_id='mid-3'))

    assert remove_entry(Path('/m/b')) is True

    paths = sorted(m.mountpoint for m in read_registry().mounts)
    assert paths == ['/m/a', '/m/c']


def test_find_by_mountpoint(isolated_registry: Path) -> None:
    _ = isolated_registry
    entry = _entry(mountpoint='/m/x', mount_id='mid-find')
    add_entry(entry)

    found = find_by_mountpoint(Path('/m/x'))
    assert found is not None
    assert found.mount_id == 'mid-find'

    assert find_by_mountpoint(Path('/m/nope')) is None


def test_find_by_supervisor_pid(isolated_registry: Path) -> None:
    _ = isolated_registry
    add_entry(_entry(mountpoint='/m/a', supervisor_pid=4321))

    found = find_by_supervisor_pid(4321)
    assert found is not None
    assert found.mountpoint == '/m/a'

    assert find_by_supervisor_pid(9999) is None


# -- Test helpers --------------------------------------------------------------


def _entry(
    mountpoint: str = '/Users/chris/.crb/host/m2/foo',
    *,
    mount_id: str = 'mid-00000001',
    peer_alias: str = 'm2',
    remote_path: str = '~/projects/foo',
    supervisor_pid: int = 12345,
    readonly: bool = False,
) -> MountEntry:
    return MountEntry(
        mount_id=mount_id,
        peer_alias=peer_alias,
        remote_path=remote_path,
        mountpoint=mountpoint,
        supervisor_pid=supervisor_pid,
        readonly=readonly,
        established_at=datetime.now(UTC),
    )
