#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../cc-lib/", editable = true }
# ///

"""Patch Claude Code statusline wrap:"truncate" → wrap:"wrap".

Ink's Text component truncates multi-line statusline output on narrow terminals,
dropping lines 2+ entirely. This same-length binary patch restores natural wrapping
by replacing the 2 statusline-specific instances while leaving the other ~84 intact.

Re-signs with ad-hoc codesign preserving original entitlements after patching.

Regression introduced in 2.1.51, patch verified working through 2.1.62+.

References:
    https://github.com/anthropics/claude-code/issues/28750
    https://github.com/anthropics/claude-code/issues/22115

Usage:
    claude-patch-statusline apply [PATH]       # patch the binary
    claude-patch-statusline check [PATH]       # dry run
    claude-patch-statusline restore [PATH]     # restore from backup
    claude-patch-statusline install            # install to PATH with completions
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import typer
from cc_lib.cli import add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.utils.atomic_write import atomic_write

app = create_app(help='Patch Claude Code statusline wrap:"truncate" → wrap:"wrap".')
error_boundary = ErrorBoundary(exit_code=1)


@app.command()
@error_boundary
def apply(
    path: Path | None = typer.Argument(None, help='Explicit binary path (default: auto-detect)'),
) -> None:
    """Auto-detect and patch the Claude binary."""
    patcher = StatuslinePatcher(path) if path else StatuslinePatcher.detect()
    print(f'Target: {patcher.path} (version {patcher.version})')
    print(f'Size: {patcher.size_mb:.1f} MB\n')
    patcher.apply()


@app.command()
@error_boundary
def check(
    path: Path | None = typer.Argument(None, help='Explicit binary path (default: auto-detect)'),
) -> None:
    """Dry run — report patch sites without modifying."""
    patcher = StatuslinePatcher(path) if path else StatuslinePatcher.detect()
    print(f'Target: {patcher.path} (version {patcher.version})')
    print(f'Size: {patcher.size_mb:.1f} MB\n')
    patcher.check()


@app.command()
@error_boundary
def restore(
    path: Path | None = typer.Argument(None, help='Explicit binary path (default: auto-detect)'),
) -> None:
    """Restore original binary from backup."""
    StatuslinePatcher.restore(path)


add_install_command(app, script_path=__file__)


class StatuslinePatcher:
    """Patches Claude Code's Ink wrap:"truncate" → wrap:"wrap" for multi-line statuslines.

    The bundled Bun binary contains 2 identical copies of the statusline rendering
    component in the __BUN segment. Each copy has the anchor ``statusLine?.padding``
    (a settings property name, stable across versions) within 200 bytes of the target
    ``wrap:"truncate"`` prop.

    Same-length replacement (space-padded to 15 bytes) preserves all binary offsets,
    alignment, and structure. After patching, ad-hoc codesign restores executable
    validity with the original entitlements.
    """

    _ANCHOR = b'statusLine?.padding'
    _OLD = b'wrap:"truncate"'
    _NEW = b'wrap:"wrap"    '  # space-padded to preserve binary size
    _SEARCH_WINDOW = 200
    _DEFAULT_SYMLINK = Path.home() / '.local' / 'bin' / 'claude'

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        try:
            self._data = self._path.read_bytes()
        except FileNotFoundError:
            raise BinaryNotFoundError(self._path) from None

    @classmethod
    def resolve_path(cls, explicit: Path | None = None) -> Path:
        """Resolve binary path without loading it."""
        if explicit is not None:
            return explicit.resolve()
        if not cls._DEFAULT_SYMLINK.exists():
            raise BinaryNotFoundError(cls._DEFAULT_SYMLINK)
        return cls._DEFAULT_SYMLINK.resolve()

    @classmethod
    def detect(cls) -> StatuslinePatcher:
        """Auto-detect binary from the default symlink."""
        return cls(cls.resolve_path())

    @classmethod
    def restore(cls, explicit: Path | None = None) -> None:
        """Restore original binary from backup."""
        path = cls.resolve_path(explicit)
        backup = Path(f'{path}.bak')
        if not backup.exists():
            raise PatchError(f'No backup found at {backup}')
        tmp = path.with_name(path.name + '.tmp')
        shutil.copy2(backup, tmp)
        tmp.replace(path)
        print(f'Restored {path.name} from backup.')

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version(self) -> str:
        return self._path.name

    @property
    def size_mb(self) -> float:
        return len(self._data) / 1e6

    def check(self) -> None:
        """Dry run — report patch sites without modifying anything."""
        sites = self._find_sites()
        if not sites:
            self._report_no_sites()
            return
        self._report_sites(sites)
        print('\nDry run — no changes made.')

    def apply(self) -> None:
        """Patch the binary, back up the original, and re-sign."""
        sites = self._find_sites()
        if not sites:
            self._report_no_sites()
            return
        self._report_sites(sites)
        backup = self._write_patched(sites)
        try:
            self._resign()
        except CodesignError:
            shutil.copy2(backup, self._path)
            print(f'Restored original from {backup}', file=sys.stderr)
            raise
        print('\nDone. Restart Claude Code to pick up the change.')

    def _find_sites(self) -> Sequence[int]:
        """Byte offsets of statusline-specific wrap:"truncate" instances."""
        sites: list[int] = []
        pos = 0
        while (pos := self._data.find(self._ANCHOR, pos)) != -1:
            target = self._data.find(self._OLD, pos, pos + self._SEARCH_WINDOW)
            if target != -1:
                sites.append(target)
            pos += len(self._ANCHOR)
        return tuple(sites)

    def _report_no_sites(self) -> None:
        if self._data.find(self._ANCHOR) != -1:
            print('Already patched (anchor found but no wrap:"truncate" nearby).')
        else:
            raise AnchorNotFoundError(self._path)

    def _report_sites(self, sites: Sequence[int]) -> None:
        print(f'Found {len(sites)} statusline patch site(s):')
        for i, offset in enumerate(sites):
            # 10 bytes pre-context + the 15-byte target
            ctx = self._data[offset - 10 : offset + len(self._OLD)]
            print(f'  [{i + 1}] 0x{offset:x}: ...{ctx!r}...')

    def _write_patched(self, sites: Sequence[int]) -> Path:
        """Patch and write binary. Returns backup path for rollback."""
        result = bytearray(self._data)
        for offset in sites:
            result[offset : offset + len(self._OLD)] = self._NEW

        backup = Path(f'{self._path}.bak')
        if not backup.exists():
            shutil.copy2(self._path, backup)
            print(f'\nBackup saved to {backup}')
        else:
            print(f'\nBackup already exists at {backup}')

        atomic_write(self._path, bytes(result), reference=self._path)
        print(f'Patched {len(sites)} occurrence(s).')
        return backup

    def _resign(self) -> None:
        """Ad-hoc codesign preserving key original signing properties.

        Uses ``--preserve-metadata`` to carry forward the identifier,
        entitlements, flags, and runtime version from the (now-invalid)
        signature already embedded in the modified binary.

        ``requirements`` is deliberately excluded: the original designated
        requirement references Anthropic's Developer ID certificate chain,
        which an ad-hoc signature cannot satisfy.  Omitting it lets codesign
        auto-generate a cdhash-based requirement the new signature passes.

        ``TeamIdentifier`` is inherently lost with ad-hoc signing (no
        certificate means no team).  Does not affect local CLI execution.
        """
        try:
            subprocess.run(
                [
                    'codesign',
                    '--sign',
                    '-',
                    '--force',
                    '--preserve-metadata=identifier,entitlements,flags,runtime',
                    str(self._path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print('Re-signed with ad-hoc signature (preserved entitlements + flags).')
        except subprocess.CalledProcessError as e:
            raise CodesignError(e.stderr) from e


# Exceptions


class PatchError(Exception):
    """Base for patching errors."""


class BinaryNotFoundError(PatchError):
    def __init__(self, path: Path) -> None:
        super().__init__(f'Claude binary not found at {path}')


class AnchorNotFoundError(PatchError):
    def __init__(self, path: Path) -> None:
        super().__init__(f'Anchor pattern not found in {path.name} — binary structure may have changed')


class CodesignError(PatchError):
    def __init__(self, stderr: str) -> None:
        super().__init__(f'codesign failed: {stderr}')


# Boundary handlers — dispatched when main() raises


@error_boundary.handler(PatchError)
def _handle_patch_error(exc: PatchError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_crash(exc: Exception) -> None:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    for frame in traceback.format_tb(exc.__traceback__)[-2:]:
        print(frame.rstrip(), file=sys.stderr)


if __name__ == '__main__':
    run_app(app)
