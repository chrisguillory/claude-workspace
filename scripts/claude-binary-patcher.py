#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

"""Claude Code binary patcher.

Applies same-length byte replacements to the Claude Code Mach-O binary.
Patch definitions and scan logic live in cc_lib.claude_binary_patching.
This script handles file I/O, codesign, and the CLI interface.

Originals (pre-patch binaries) are stored in our workspace directory:
    ~/.claude-workspace/binary-patcher/originals/{version}

Usage:
    claude-binary-patcher apply               Apply fix patches (safe default)
    claude-binary-patcher apply --all         Apply all patches (fixes + features)
    claude-binary-patcher apply --features    Apply feature patches only
    claude-binary-patcher apply NAME...       Apply specific patches by name
    claude-binary-patcher check --all         Dry run — show all patch status
    claude-binary-patcher list                Show available patches by kind
    claude-binary-patcher restore             Restore original binary from backup
    claude-binary-patcher install             Install to PATH with completions

References:
    https://github.com/anthropics/claude-code/issues/28750  (statusline)
    https://github.com/anthropics/claude-code/issues/41361  (mcp-tool-results)
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer
from cc_lib.claude_binary_patching import (
    ORIGINALS_DIR,
    PATCHES,
    PATCHES_BY_KIND,
    PATCHES_BY_NAME,
    PatchDef,
    PatchKind,
    PatchScanResult,
    scan_binary,
)
from cc_lib.claude_context import ClaudeContext
from cc_lib.claude_process import kill_and_copy_resume
from cc_lib.cli import add_help_command, add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.exceptions import ClaudeContextError
from cc_lib.types import CCVersion
from cc_lib.utils.atomic_write import atomic_write


class PatchError(Exception):
    """Base for patching errors."""


class BinaryNotFoundError(PatchError):
    def __init__(self, path: Path) -> None:
        super().__init__(f'Claude binary not found at {path}')


class CodesignError(PatchError):
    def __init__(self, stderr: str) -> None:
        super().__init__(f'codesign failed: {stderr}')


class RequiredSettingsUnsatisfiedError(PatchError):
    """One or more selected patches' required_setting isn't satisfied — refuse to apply.

    Raised by the apply command's pre-flight check. Carries per-patch reasons
    so the boundary handler can render a single loud error block.
    """

    def __init__(self, unsatisfied: Mapping[str, Sequence[str]]) -> None:
        super().__init__('required settings not satisfied for selected patches')
        self.unsatisfied = unsatisfied


app = create_app(help='Claude Code binary patcher.')
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


def resolve_patches(
    names: Sequence[str] | None,
    *,
    fixes: bool = False,
    features: bool = False,
    tweaks: bool = False,
    all_: bool = False,
) -> Sequence[PatchDef]:
    """Resolve CLI names/flags to PatchDefs.

    Priority: explicit names > kind flags > default (fixes only).
    """
    if names:
        unknown = [n for n in names if n not in PATCHES_BY_NAME]
        if unknown:
            available = ', '.join(PATCHES_BY_NAME)
            raise PatchError(f'Unknown patch(es): {", ".join(unknown)}\nAvailable: {available}')
        seen: dict[str, None] = {}
        for n in names:
            seen[n] = None
        return tuple(PATCHES_BY_NAME[n] for n in seen)

    if all_:
        return PATCHES

    kinds: list[PatchKind] = []
    if fixes:
        kinds.append(PatchKind.FIX)
    if features:
        kinds.append(PatchKind.FEATURE)
    if tweaks:
        kinds.append(PatchKind.TWEAK)

    if not kinds:
        kinds = [PatchKind.FIX]  # safe default

    return tuple(p for p in PATCHES if p.kind in kinds)


@app.command()
@error_boundary
def apply(
    names: list[str] | None = typer.Argument(  # strict_typing_linter.py: mutable-type — typer requires list
        None,
        help='Patch names to apply',
        autocompletion=lambda incomplete: _complete_patch_names(incomplete),
    ),
    path: Path | None = typer.Option(None, '--path', help='Explicit binary path (default: auto-detect)'),
    fixes: bool = typer.Option(False, '--fixes', help='Apply fix patches'),
    features: bool = typer.Option(False, '--features', help='Apply feature patches'),
    tweaks: bool = typer.Option(False, '--tweaks', help='Apply tweak patches'),
    all_: bool = typer.Option(False, '--all', help='Apply all patches'),
    restart: bool = typer.Option(False, '--restart', help='Kill Claude Code and copy resume command to clipboard'),
) -> None:
    """Apply patches. Default (no names/flags): fixes only.

    \b
    Examples:
        apply                Apply all fix patches (safe default)
        apply --all          Apply everything (fixes + features + tweaks)
        apply --features     Apply feature patches only
        apply statusline     Apply a specific patch by name
        apply --restart      Apply fixes and restart Claude Code
    """
    patches = resolve_patches(names, fixes=fixes, features=features, tweaks=tweaks, all_=all_)
    _preflight_required_settings(patches)
    patcher = BinaryPatcher(path) if path else BinaryPatcher.detect()
    print(f'Target: {patcher.path} (version {patcher.version})')
    print(f'Size: {patcher.size_mb:.1f} MB')
    patcher.apply(patches)

    if restart:
        try:
            resume_cmd = kill_and_copy_resume(ClaudeContext.from_env())
            print()
            print(f'Resume command copied: {resume_cmd}')
            print('Paste Cmd+V + Enter after Claude exits.')
        except ClaudeContextError as e:
            print(f'\nNote: {e}', file=sys.stderr)
            print('Restart Claude Code manually to pick up the changes.')


@app.command()
@error_boundary
def check(
    names: list[str] | None = typer.Argument(  # strict_typing_linter.py: mutable-type — typer requires list
        None,
        help='Patch names to check',
        autocompletion=lambda incomplete: _complete_patch_names(incomplete),
    ),
    path: Path | None = typer.Option(None, '--path', help='Explicit binary path (default: auto-detect)'),
    fixes: bool = typer.Option(False, '--fixes', help='Check fix patches only'),
    features: bool = typer.Option(False, '--features', help='Check feature patches only'),
    tweaks: bool = typer.Option(False, '--tweaks', help='Check tweak patches only'),
    all_: bool = typer.Option(False, '--all', help='Check all patches'),
) -> None:
    """Dry run — show per-patch status. Default: fixes only.

    \b
    Examples:
        check                Check fix patches (safe default)
        check --all          Check all patches
        check --features     Check feature patches only
    """
    patches = resolve_patches(names, fixes=fixes, features=features, tweaks=tweaks, all_=all_)
    patcher = BinaryPatcher(path) if path else BinaryPatcher.detect()
    print(f'Target: {patcher.path} (version {patcher.version})')
    print(f'Size: {patcher.size_mb:.1f} MB\n')
    results = patcher.scan(patches)
    _print_results_by_kind(results)
    print('\nDry run — no changes made.')


def _print_results_by_kind(results: Mapping[str, PatchScanResult]) -> None:
    """Print scan results grouped by patch kind."""
    by_kind: dict[PatchKind, list[tuple[str, PatchScanResult]]] = {}
    for name, result in results.items():
        by_kind.setdefault(result.patch.kind, []).append((name, result))

    for kind in PatchKind:
        entries = by_kind.get(kind)
        if not entries:
            continue
        print(f'  {kind.value.title()}:')
        for name, result in entries:
            print(f'    {name:<20} {_format_status(result)}')


def _format_status(result: PatchScanResult) -> str:
    if result.status == 'unpatched':
        return f'unpatched  ({len(result.sites)} sites)'
    if result.status == 'applied':
        reasons = _unsatisfied_reasons(result.patch)
        if reasons:
            return f'applied    (INERT — {"; ".join(reasons)})'
        return 'applied'
    if result.status == 'changed':
        return 'changed    (anchor found, code different)'
    if result.status == 'out_of_range':
        bound = (
            f'max_version={result.patch.max_version}'
            if result.patch.max_version is not None
            else f'min_version={result.patch.min_version}'
        )
        return f'skipped    (out of range, {bound})'
    return 'missing    (anchor not found)'


def _unsatisfied_reasons(patch: PatchDef) -> Sequence[str]:
    """Return human-readable reasons each required-setting isn't met (empty if all met)."""
    return [reason for rs in patch.required_setting if (reason := rs.unsatisfied_reason()) is not None]


def _preflight_required_settings(patches: Sequence[PatchDef]) -> None:
    """Raise ``RequiredSettingsUnsatisfiedError`` if any selected patch is inert.

    Runs BEFORE any byte modification so the binary stays untouched on
    failure. Fail-fast surface for the user: fix settings, then re-run.
    """
    unsatisfied: dict[str, Sequence[str]] = {}
    for patch in patches:
        reasons = _unsatisfied_reasons(patch)
        if reasons:
            unsatisfied[patch.name] = reasons
    if unsatisfied:
        raise RequiredSettingsUnsatisfiedError(unsatisfied)


@app.command(name='list')
@error_boundary
def list_patches() -> None:
    """Show all available patches grouped by kind."""
    for kind in PatchKind:
        patches = PATCHES_BY_KIND.get(kind, ())
        if not patches:
            continue
        print(f'{kind.value.title()}:')
        for patch in patches:
            print(f'  {patch.name:<20} {patch.description}')
        print()


@app.command()
@error_boundary
def restore(
    path: Path | None = typer.Option(None, '--path', help='Explicit binary path (default: auto-detect)'),
    restart: bool = typer.Option(False, '--restart', help='Kill Claude Code and copy resume command to clipboard'),
) -> None:
    """Restore original binary from backup."""
    BinaryPatcher.restore(path)

    if restart:
        try:
            resume_cmd = kill_and_copy_resume(ClaudeContext.from_env())
            print()
            print(f'Resume command copied: {resume_cmd}')
            print('Paste Cmd+V + Enter after Claude exits.')
        except ClaudeContextError as e:
            print(f'\nNote: {e}', file=sys.stderr)
            print('Restart Claude Code manually to pick up the changes.')


add_install_command(app, script_path=__file__)


@error_boundary.handler(RequiredSettingsUnsatisfiedError)
def _handle_required_settings_unsatisfied(exc: RequiredSettingsUnsatisfiedError) -> None:
    bar = '═' * 70
    print(f'\n{bar}', file=sys.stderr)
    print('  REFUSING TO APPLY — required settings not satisfied', file=sys.stderr)
    print(f'{bar}\n', file=sys.stderr)
    for name, reasons in exc.unsatisfied.items():
        print(f'  {name}', file=sys.stderr)
        for reason in reasons:
            print(f'    • {reason}', file=sys.stderr)
        print(file=sys.stderr)
    print('  To fix: edit ~/.claude/settings.json and set the required keys,', file=sys.stderr)
    print('  or unset the disabling env vars, then re-run.', file=sys.stderr)
    print(f'{bar}', file=sys.stderr)


@error_boundary.handler(PatchError)
def _handle_patch_error(exc: PatchError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_crash(exc: Exception) -> None:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    for frame in traceback.format_tb(exc.__traceback__)[-2:]:
        print(frame.rstrip(), file=sys.stderr)


class BinaryPatcher:
    """Same-length binary patcher for Claude Code.

    Reads the full binary into memory, delegates scan logic to
    cc_lib.claude_binary_patching.scan_binary(), handles file I/O,
    atomic writes, codesign, and originals management.
    """

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
    def detect(cls) -> BinaryPatcher:
        """Auto-detect binary from the default symlink."""
        return cls(cls.resolve_path())

    @classmethod
    def restore(cls, explicit: Path | None = None) -> None:
        """Restore original binary from backup.

        Checks the originals directory first, falls back to legacy .bak
        location for backward compatibility.
        """
        path = cls.resolve_path(explicit)
        version = path.name
        original = ORIGINALS_DIR / version
        legacy_bak = Path(f'{path}.bak')

        if original.exists():
            backup_source = original
        elif legacy_bak.exists():
            backup_source = legacy_bak
        else:
            raise PatchError(f'No original found at {original} or {legacy_bak}')

        atomic_write(path, backup_source.read_bytes(), reference=backup_source)
        print(f'Restored {path.name} from {backup_source}.')

    @property
    def path(self) -> Path:
        return self._path

    @property
    def version(self) -> CCVersion:
        return CCVersion(self._path.name)

    @property
    def size_mb(self) -> float:
        return len(self._data) / 1e6

    def scan(self, patches: Sequence[PatchDef]) -> Mapping[str, PatchScanResult]:
        """Scan binary for all patches."""
        return scan_binary(self._data, patches, current_version=self.version)

    def apply(self, patches: Sequence[PatchDef]) -> None:
        """Apply patches in a single read-modify-write-sign cycle."""
        scan_results = self.scan(patches)

        pending_sites: dict[str, Sequence[int]] = {}
        applied_names: list[str] = []
        missing_names: list[str] = []
        unknown_names: list[str] = []
        skipped_names: list[str] = []

        for name, result in scan_results.items():
            if result.status == 'out_of_range':
                skipped_names.append(name)
                continue
            if result.status == 'unpatched':
                pending_sites[name] = result.sites
                self._report_sites(result.patch, result.sites)
            elif result.status == 'applied':
                applied_names.append(name)
                print(f'{name}:  already applied')
            elif result.status == 'changed':
                unknown_names.append(name)
                print(f'{name}:  UNKNOWN — anchor found but code changed (investigate)')
            elif result.status == 'missing':
                missing_names.append(name)
                print(f'{name}:  ANCHOR NOT FOUND — binary structure may have changed')

        if not pending_sites:
            if applied_names:
                suffix = (
                    f'\nSkipped {len(skipped_names)} out-of-range patch(es): {", ".join(skipped_names)}'
                    if skipped_names
                    else ''
                )
                print(f'\nAll requested patches already applied.{suffix}')
                return
            raise PatchError('No patches could be applied (all anchors missing or code changed)')

        total_sites = sum(len(s) for s in pending_sites.values())

        original_path = self._save_original()
        self._write_patched(pending_sites, scan_results)
        try:
            self._resign()
        except CodesignError:
            try:
                atomic_write(self._path, original_path.read_bytes(), reference=original_path)
                print(f'Restored original from {original_path}', file=sys.stderr)
            except Exception as restore_err:
                print(f'CRITICAL: Codesign failed AND restore failed: {restore_err}', file=sys.stderr)
                print(f'Manual recovery: cp "{original_path}" "{self._path}"', file=sys.stderr)
            raise

        suffix = ''
        if applied_names:
            suffix += f' ({", ".join(applied_names)} already applied)'
        if missing_names:
            suffix += f'\nWARNING: anchor not found for: {", ".join(missing_names)}'
        if unknown_names:
            suffix += f'\nWARNING: code changed for: {", ".join(unknown_names)}'
        if skipped_names:
            suffix += f'\nSkipped {len(skipped_names)} out-of-range patch(es): {", ".join(skipped_names)}'

        print(f'\nPatched {total_sites} site(s) across {len(pending_sites)} patch(es).{suffix}')

        print('\nDone. Restart Claude Code to pick up the changes.')

    def _report_sites(self, patch: PatchDef, sites: Sequence[int]) -> None:
        print(f'\n{patch.name}:')
        print(f'  Found {len(sites)} patch site(s)')
        for i, offset in enumerate(sites):
            ctx = self._data[max(0, offset - 10) : offset + len(patch.old)]
            print(f'  [{i + 1}] 0x{offset:x}: ...{ctx!r}...')

    def _save_original(self) -> Path:
        """Save the original binary to our workspace directory."""
        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        original_path = ORIGINALS_DIR / str(self.version)
        if not original_path.exists():
            atomic_write(original_path, self._data, reference=self._path)
            print(f'\nOriginal saved to {original_path}')
        else:
            print(f'\nOriginal already exists at {original_path}')
        return original_path

    def _write_patched(
        self,
        all_sites: Mapping[str, Sequence[int]],
        scan_results: Mapping[str, PatchScanResult],
    ) -> None:
        """Apply byte replacements and write the patched binary atomically."""
        result = bytearray(self._data)
        for name, sites in all_sites.items():
            patch = scan_results[name].patch
            for offset in sites:
                result[offset : offset + len(patch.old)] = patch.new
        atomic_write(self._path, bytes(result), reference=self._path)

    def _resign(self) -> None:
        """Ad-hoc codesign preserving key original signing properties."""
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


def _complete_patch_names(incomplete: str) -> Sequence[tuple[str, str]]:
    return [(p.name, p.description) for p in PATCHES if p.name.startswith(incomplete)]


if __name__ == '__main__':
    run_app(app)
