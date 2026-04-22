#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
#     "httpx",
#     "pydantic>=2.0",
#     "typer>=0.9.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

"""Claude Code version manager.

Fetch, list, clean, and run specific Claude Code versions side-by-side.
Versions are installed alongside the active binary and run by path —
the tool never touches ~/.local/bin/claude (CC's auto-updater owns that).

Patch detection:
    Mach-O header   CS_ADHOC bit — definitive modified vs original (~0.5ms)
    Patch manifest  JSON from claude-binary-patcher — which patches applied (~0ms)

Infrastructure:
    CDN:   Native binaries with SHA-256 verification (~9 month retention, 1.0.37+)
    npm:   @anthropic-ai/claude-code (all 313+ versions, used for enumeration only)
    Local: ~/.local/share/claude/versions/{VERSION}

Usage:
    claude-version-manager list                        # local versions with patch status
    claude-version-manager list --remote               # all published versions via npm
    claude-version-manager fetch 2.1.74               # download with SHA-256 verification
    claude-version-manager run 2.1.74                  # fetch if needed, then launch
    claude-version-manager clean                       # remove .bak files and 0-byte ghosts
    claude-version-manager info                        # active version details
    claude-version-manager install                     # install to PATH with completions
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import pydantic
import typer
from cc_lib.claude_binary_patching import ORIGINALS_DIR, scan_binary
from cc_lib.cli import add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import ClosedModel, OpenModel
from cc_lib.types import CCVersion

CDN_BASE = 'https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases'
NPM_PACKAGE = '@anthropic-ai/claude-code'
VERSIONS_DIR = Path.home() / '.local' / 'share' / 'claude' / 'versions'
SYMLINK_PATH = Path.home() / '.local' / 'bin' / 'claude'
PLATFORM = 'darwin-arm64'


class PlatformInfo(OpenModel):
    """Single platform entry from CDN manifest.json."""

    binary: str
    checksum: str
    size: int


class Manifest(OpenModel):
    """CDN manifest.json for a specific version."""

    version: CCVersion
    build_date: str = pydantic.Field(alias='buildDate')
    platforms: Mapping[str, PlatformInfo]


class LocalVersion(ClosedModel):
    """A version discovered on the local filesystem."""

    version: CCVersion
    path: Path
    size: int
    is_active: bool
    is_bak: bool
    is_ghost: bool
    is_adhoc: bool | None  # True = patched (ad-hoc signed), False = Anthropic-signed, None = unknown
    patches: Sequence[str]  # applied patch names from binary scan


app = create_app(help='Claude Code version manager.')
error_boundary = ErrorBoundary(exit_code=1)


@app.command('list')
@error_boundary
def cli_list(
    remote: bool = typer.Option(False, '--remote', '-r', help='Query npm for all available versions'),
    last: int = typer.Option(20, '--last', '-n', help='Show last N versions (--remote only)'),
) -> None:
    """List installed versions. Use --remote to see all available.

    Local mode shows binaries in ~/.local/share/claude/versions/ with
    signature and patch status from Mach-O header + manifest detection.
    Remote mode queries npm for all 313+ published versions.

    \b
    Output markers:
        *           active version (~/.local/bin/claude symlink target)
        anthropic   original Anthropic-signed binary
        patched     ad-hoc signed (modified by binary patcher)
        legacy-bak  .bak file from old patcher location (use clean to remove)
        ghost       0-byte file from a failed install
    """
    store = VersionStore()
    local = store.scan()

    if not remote:
        if not local:
            print('No versions installed.')
            return

        active = store.active_version()
        print(f'Active: {active or "unknown"}\n')
        for v in local:
            marker = ' *' if v.is_active else ''
            if v.is_ghost:
                label = '0 bytes — ghost'
            elif v.is_bak:
                label = f'{v.size / 1_000_000:.0f} MB, legacy-bak'
            elif v.is_adhoc and v.patches:
                label = f'{v.size / 1_000_000:.0f} MB, patched: {", ".join(v.patches)}'
            elif v.is_adhoc:
                label = f'{v.size / 1_000_000:.0f} MB, patched'
            else:
                label = f'{v.size / 1_000_000:.0f} MB, anthropic'
            print(f'  {v.version}{marker}  ({label})')
        return

    npm = NpmRegistry()
    all_versions = npm.versions()
    recent = list(reversed(all_versions[-last:]))

    installed = {v.version for v in local if not v.is_bak}
    active = store.active_version()

    print(f'Last {last} versions (of {len(all_versions)} total):\n')
    for version in recent:
        markers: list[str] = []
        if version in installed:
            markers.append('installed')
        if version == active:
            markers.append('active')
        suffix = f'  ({", ".join(markers)})' if markers else ''
        print(f'  {version}{suffix}')


@app.command('fetch')
@error_boundary
def cli_fetch(version: str = typer.Argument(..., help='Version number, "latest", or "stable"')) -> None:
    """Fetch a version from CDN (with SHA-256 verification)."""
    store = VersionStore()
    cdn = CDNClient()

    if version in ('latest', 'stable'):
        resolved = cdn.resolve_channel(version)
        print(f'{version} → {resolved}')
        version = resolved

    if store.is_installed(version):
        print(f'Already installed: {store.path_for(version)}')
        return

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = cdn.download(version, VERSIONS_DIR)
    print(f'Installed: {path}')


@app.command('clean')
@error_boundary
def cli_clean(dry_run: bool = typer.Option(False, '--dry-run', help='Show what would be removed')) -> None:
    """Remove .bak files and 0-byte ghost files."""
    store = VersionStore()
    removed = store.clean(dry_run=dry_run)
    if not removed:
        print('Nothing to clean.')
    elif dry_run:
        total = sum(p.stat().st_size for p in removed if p.exists())
        print(f'\n{len(removed)} files, {total / 1_000_000:.0f} MB (dry run — pass without --dry-run to remove)')
    else:
        print(f'\nRemoved {len(removed)} files.')


@app.command('info')
@error_boundary
def cli_info(
    version: str = typer.Argument(
        None, help='Version (default: active)', autocompletion=lambda incomplete: _complete_local_version(incomplete)
    ),
) -> None:
    """Show detailed version info (local + remote)."""
    store = VersionStore()

    if version is None:
        version = store.active_version()
        if version is None:
            print('No active version detected. Pass a version number.', file=sys.stderr)
            raise SystemExit(1)
        print(f'Active version: {version}\n')

    path = store.path_for(version)
    if path.exists():
        st = path.stat()
        adhoc = MachOSignature.is_adhoc(path)
        sig_label = 'ad-hoc (modified)' if adhoc else 'Anthropic' if adhoc is False else 'unknown'

        print('Local:')
        print(f'  Path: {path}')
        print(f'  Size: {st.st_size / 1_000_000:.0f} MB')
        print(f'  Signature: {sig_label}')
        print(f'  Active: {store.active_version() == version}')

        if adhoc:
            data = path.read_bytes()
            results = scan_binary(data)
            applied = [n for n, r in results.items() if r.status == 'applied']
            unpatched = [n for n, r in results.items() if r.status == 'unpatched']
            changed = [n for n, r in results.items() if r.status == 'changed']
            if applied:
                print(f'  Patches applied: {", ".join(applied)}')
            if unpatched:
                print(f'  Patches available: {", ".join(unpatched)}')
            if changed:
                print(f'  Patches changed: {", ".join(changed)} (code differs from expected)')
            if not applied and not unpatched and not changed:
                print('  Modified (ad-hoc signed, no recognized patches)')

        original = ORIGINALS_DIR / version
        if original.exists():
            print(f'  Original: {original} ({original.stat().st_size / 1_000_000:.0f} MB)')
    else:
        print('Not installed locally.')

    cdn = CDNClient()
    try:
        cdn_manifest = cdn.fetch_manifest(version)
        platform = cdn_manifest.platforms.get(PLATFORM)
        print('\nCDN:')
        print(f'  Build date: {cdn_manifest.build_date}')
        if platform:
            print(f'  Size: {platform.size / 1_000_000:.0f} MB')
            print(f'  Checksum: {platform.checksum[:32]}...')
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:  # noqa: PLR2004 — HTTP status code
            print('\nNot available on CDN (retention ~9 months, 1.0.37+).')
        else:
            raise


@app.command('run')
def cli_run(
    version: str = typer.Argument(
        ..., help='Version to run', autocompletion=lambda incomplete: _complete_local_version(incomplete)
    ),
    args: list[str] = typer.Argument(
        None, help='Arguments to pass to claude'
    ),  # strict_typing_linter.py: mutable-type — typer requires list
) -> None:
    """Run a specific Claude Code version (fetches if needed).

    Replaces the current process via execvp — the launched Claude Code
    inherits the terminal directly for interactive use with the statusline.

    \b
    Examples:
        claude-version-manager run 2.1.74
        claude-version-manager run 2.1.74 -- --resume abc123
        claude-version-manager run stable
    """
    store = VersionStore()

    if not store.is_installed(version):
        cli_fetch(version)

    path = store.path_for(version)
    if not path.exists():
        print(f'ERROR: Binary not found after fetch: {path}', file=sys.stderr)
        raise SystemExit(1)

    os.execvp(str(path), [str(path), *(args or [])])


add_install_command(app, script_path=__file__)


class MachOSignature:
    """Detect ad-hoc vs Anthropic-signed from Mach-O CodeDirectory flags.

    Reads under 200 bytes from the file header — no full binary load needed.
    The CS_ADHOC bit (0x2) in CodeDirectory flags is set by our patcher's
    ad-hoc codesign and absent on Anthropic's original Developer ID signature.
    """

    MH_MAGIC_64 = 0xFEEDFACF
    LC_CODE_SIGNATURE = 0x1D
    CS_ADHOC = 0x2
    CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
    CSMAGIC_CODEDIRECTORY = 0xFADE0C02

    @classmethod
    def is_adhoc(cls, path: Path) -> bool | None:
        """Check if a Mach-O binary has an ad-hoc code signature.

        Returns True (ad-hoc/modified), False (real signature), or None
        (not a valid Mach-O or no code signature found).
        """
        try:
            with path.open('rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != cls.MH_MAGIC_64:
                    return None

                f.seek(16)
                ncmds = struct.unpack('<I', f.read(4))[0]
                f.seek(32)

                for _ in range(ncmds):
                    pos = f.tell()
                    cmd, cmdsize = struct.unpack('<II', f.read(8))
                    if cmd == cls.LC_CODE_SIGNATURE:
                        dataoff = struct.unpack('<I', f.read(4))[0]
                        f.seek(dataoff)
                        cs_magic = struct.unpack('>I', f.read(4))[0]
                        if cs_magic != cls.CSMAGIC_EMBEDDED_SIGNATURE:
                            return None
                        _cs_length, cs_count = struct.unpack('>II', f.read(8))
                        for _ in range(cs_count):
                            blob_type, blob_offset = struct.unpack('>II', f.read(8))
                            if blob_type == 0:  # CSSLOT_CODEDIRECTORY
                                f.seek(dataoff + blob_offset)
                                cd_magic = struct.unpack('>I', f.read(4))[0]
                                if cd_magic != cls.CSMAGIC_CODEDIRECTORY:
                                    return None
                                _cd_length, _cd_version, cd_flags = struct.unpack('>III', f.read(12))
                                return bool(cd_flags & cls.CS_ADHOC)
                        return None
                    f.seek(pos + cmdsize)
        except (OSError, struct.error):
            return None
        return None


class VersionStore:
    """Local version inventory and cleanup.

    Scans ~/.local/share/claude/versions/ for installed binaries,
    resolves the active version from the claude symlink, checks
    Mach-O signature and binary patch status for each version, and
    removes detritus (.bak backups, 0-byte ghost files).
    """

    def __init__(self, versions_dir: Path = VERSIONS_DIR) -> None:
        self._dir = versions_dir

    def scan(self) -> Sequence[LocalVersion]:
        """Scan versions directory with binary inspection.

        For each binary: Mach-O header (~0.5ms) for signature type.
        For ad-hoc binaries: full binary scan (~90ms) for patch status.
        Sorted descending by version.
        """
        if not self._dir.exists():
            return []

        active = self.active_version()
        versions: list[LocalVersion] = []

        for entry in self._dir.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            version = name.removesuffix('.bak')
            is_bak = name.endswith('.bak')
            st = entry.stat()
            is_ghost = st.st_size == 0

            adhoc: bool | None = None
            patches: Sequence[str] = ()
            if not is_bak and not is_ghost:
                adhoc = MachOSignature.is_adhoc(entry)
                if adhoc:
                    data = entry.read_bytes()
                    results = scan_binary(data)
                    patches = [patch_name for patch_name, r in results.items() if r.status == 'applied']

            versions.append(
                LocalVersion(
                    version=version,
                    path=entry,
                    size=st.st_size,
                    is_active=(not is_bak and active is not None and version == active),
                    is_bak=is_bak,
                    is_ghost=is_ghost,
                    is_adhoc=adhoc,
                    patches=patches,
                ),
            )

        return sorted(versions, key=lambda v: v.version, reverse=True)

    def active_version(self) -> str | None:
        """Read the claude symlink target, extract version string."""
        try:
            target = SYMLINK_PATH.resolve()
            if target.parent == self._dir.resolve():
                return target.name
        except OSError:
            pass
        return None

    def path_for(self, version: str) -> Path:
        """Return expected path for a version binary."""
        return self._dir / version

    def is_installed(self, version: str) -> bool:
        """Check if version exists locally and is valid (non-zero, executable)."""
        path = self.path_for(version)
        if not path.exists():
            return False
        st = path.stat()
        return st.st_size > 0 and bool(st.st_mode & stat.S_IXUSR)

    def clean(self, *, dry_run: bool) -> Sequence[Path]:
        """Remove .bak files and 0-byte ghosts. Returns removed paths."""
        removed: list[Path] = []
        for v in self.scan():
            if v.is_bak or v.is_ghost:
                if dry_run:
                    label = 'ghost (0 bytes)' if v.is_ghost else 'legacy-bak (old patcher artifact)'
                    print(f'  Would remove: {v.path.name} ({label})')
                else:
                    v.path.unlink()
                    print(f'  Removed: {v.path.name}')
                removed.append(v.path)
        return removed


class CDNClient:
    """Claude Code CDN — binary download with checksum verification.

    Downloads native binaries from the GCS-hosted CDN with streaming
    SHA-256 verification and atomic file placement. The CDN retains
    approximately 9 months of versions (1.0.37+ as of March 2026).
    """

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(30.0, read=300.0))

    def fetch_manifest(self, version: str) -> Manifest:
        """Fetch manifest.json for a version."""
        resp = self._client.get(f'{CDN_BASE}/{version}/manifest.json')
        resp.raise_for_status()
        return Manifest.model_validate(resp.json())

    def resolve_channel(self, channel: str) -> str:
        """Resolve 'latest' or 'stable' to a version string."""
        resp = self._client.get(f'{CDN_BASE}/{channel}')
        resp.raise_for_status()
        return resp.text.strip()

    def download(self, version: str, dest_dir: Path) -> Path:
        """Download binary with streaming SHA-256 verification."""
        manifest = self.fetch_manifest(version)
        platform = manifest.platforms.get(PLATFORM)
        if platform is None:
            msg = f'Platform {PLATFORM} not in manifest for {version}'
            raise KeyError(msg)

        binary_url = f'{CDN_BASE}/{version}/{PLATFORM}/{platform.binary}'
        dest = dest_dir / version
        tmp = dest_dir / f'.{version}.tmp'

        print(f'  Downloading {version} ({platform.size / 1_000_000:.0f} MB)...')

        hasher = hashlib.sha256()
        with self._client.stream('GET', binary_url) as resp:
            resp.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    hasher.update(chunk)

        actual = hasher.hexdigest()
        if actual != platform.checksum:
            tmp.unlink(missing_ok=True)
            msg = f'Checksum mismatch for {version}: expected {platform.checksum[:16]}..., got {actual[:16]}...'
            raise ValueError(msg)

        os.replace(tmp, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        print(f'  Verified: SHA-256 {actual[:16]}...')
        return dest

    def is_available(self, version: str) -> bool:
        """HEAD check on manifest.json."""
        try:
            resp = self._client.head(f'{CDN_BASE}/{version}/manifest.json')
            return resp.status_code == 200  # noqa: PLR2004 — HTTP status code
        except httpx.HTTPError:
            return False


class NpmRegistry:
    """npm version enumeration — the CDN has no listing endpoint.

    All 313+ published versions are available via npm back to 0.2.9.
    Used strictly for version discovery and publish date lookup;
    binaries always come from the CDN.
    """

    def versions(self) -> Sequence[str]:
        """All published versions from npm registry."""
        result = subprocess.run(
            ['npm', 'view', NPM_PACKAGE, 'versions', '--json'],
            capture_output=True,
            text=True,
            check=True,
        )
        versions: Sequence[str] = json.loads(result.stdout)
        return versions

    def timestamps(self) -> Mapping[str, str]:
        """Publish timestamps from npm registry."""
        result = subprocess.run(
            ['npm', 'view', NPM_PACKAGE, 'time', '--json'],
            capture_output=True,
            text=True,
            check=True,
        )
        timestamps: Mapping[str, str] = json.loads(result.stdout)
        return timestamps


@error_boundary.handler(httpx.HTTPStatusError)
def _handle_http_error(exc: httpx.HTTPStatusError) -> None:
    if exc.response.status_code == 404:  # noqa: PLR2004 — HTTP status code
        print('ERROR: Version not found on CDN. Retention is ~9 months (1.0.37+).', file=sys.stderr)
    else:
        print(f'ERROR: CDN request failed: {exc.response.status_code} {exc.request.url}', file=sys.stderr)


@error_boundary.handler(FileNotFoundError)
def _handle_file_not_found(exc: FileNotFoundError) -> None:
    print(f'ERROR: {exc}', file=sys.stderr)


@error_boundary.handler(subprocess.CalledProcessError)
def _handle_subprocess(exc: subprocess.CalledProcessError) -> None:
    print(f'ERROR: npm failed (exit {exc.returncode}): {exc.stderr or ""}', file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'ERROR: {exc!r}', file=sys.stderr)


def _complete_local_version(incomplete: str) -> Sequence[tuple[str, str]]:
    """Complete version strings from locally installed versions."""
    store = VersionStore()
    return [
        (v.version, f'{v.size / 1_000_000:.0f} MB')
        for v in store.scan()
        if not v.is_bak and not v.is_ghost and v.version.startswith(incomplete)
    ]


if __name__ == '__main__':
    run_app(app)
