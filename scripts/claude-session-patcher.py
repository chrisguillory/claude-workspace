#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
"""Claude Code session patcher -- diagnose and repair session JSONL corruption.

Detects and applies patches for known corruption in Claude Code's session
JSONL files that cause --resume to fail, show empty history, or enter error
loops.

Backups are stored in:
    ~/.claude-workspace/scripts/claude-session-patcher/backups/

Session corruption falls into two categories:

    Chain corruption targets the parentUuid linked-list that --resume walks
    to reconstruct conversation history. Orphan pointers, duplicate UUIDs,
    and stale parent references break this walk. Fixes rewire parentUuid
    fields on affected records.

    Protocol corruption targets the API message structure. Message
    structure violations trigger HTTP 400 rejection loops that grow
    unboundedly as each retry appends another synthetic error record.
    Fixes truncate the corrupt tail back to the last clean exchange.

Patches:
    duplicate-uuid      saved_hook_context entries share a UUID derived from
                        toolUseID "SessionStart". Creates cycles or dangling
                        refs in the UUID index. Session shows empty on
                        resume. Fix: exclude duplicates, rewire affected
                        records.
                        https://github.com/anthropics/claude-code/issues/22178
                        https://github.com/anthropics/claude-code/issues/22042

    orphan-sidechain    SubAgent/hook progress entries write parentUuid
                        pointing into sidechain files. Tip selector picks
                        the orphan branch over the real conversation. Session
                        shows empty on resume. Fix: rewire to nearest valid
                        predecessor.
                        https://github.com/anthropics/claude-code/issues/33651
                        https://github.com/anthropics/claude-code/issues/35024
                        https://github.com/anthropics/claude-code/issues/37437
                        https://github.com/anthropics/claude-code/issues/24304

    oversized-image     An image in the session exceeds Anthropic's 2000px
                        many-image dimension cap (triggered when a request
                        contains >20 images). Each resume replays the full
                        history; API rejects with 400; a <synthetic> error
                        record is appended, creating a growing error tail.
                        Fix: redact the oversized image block in place
                        (replace with a text placeholder preserving the
                        tool_result structure) AND drop the dim-error
                        synthetic record and everything after it. The
                        pending user message that triggered the rejection
                        is preserved, since after redaction the same
                        request will succeed on replay.
                        https://github.com/anthropics/claude-code/issues/34025
                        https://github.com/anthropics/claude-code/issues/16173
                        https://github.com/anthropics/claude-code/issues/13480
                        https://github.com/anthropics/claude-code/issues/9375

    stale-parent        (detect-only) After resume, new user record's
                        parentUuid latches onto old compaction segment,
                        skipping newer compact_boundary records. Session
                        works but missing recent context. Auto-fix deferred
                        until empirically proven.
                        https://github.com/anthropics/claude-code/issues/43941
                        https://github.com/anthropics/claude-code/issues/39856
                        https://github.com/anthropics/claude-code/issues/40319
                        https://github.com/anthropics/claude-code/issues/43044

Possible Patches (not yet implemented):
    compaction-orphan-pair  Compaction drops one side of a tool_use/tool_result
                        pair, leaving an orphan the API rejects on replay.
                        https://github.com/anthropics/claude-code/issues/40305

    empty-text-block    Extended thinking streaming emits empty text content
                        blocks. API rejects on resume.
                        https://github.com/anthropics/claude-code/issues/41992

    retry-accumulation  Streaming disconnects cause retry message accumulation,
                        growing payload until permanently unrecoverable.
                        https://github.com/anthropics/claude-code/issues/40316

    string-content-format  Voice-dictated messages store content as string
                        instead of array. Compaction can't merge, eventually
                        producing API-rejected message sequences.
                        https://github.com/anthropics/claude-code/issues/37452

Usage:
    claude-session-patcher scan                     Scan all sessions
    claude-session-patcher scan --project <path>    Scope to one project
    claude-session-patcher check <id>               Detailed diagnosis
    claude-session-patcher check <id> --json        Machine-readable output
    claude-session-patcher fix <id>                 Fix with backup + verify
    claude-session-patcher fix --all                Fix all fixable sessions
    claude-session-patcher fix <id> --dry-run       Show what would change
    claude-session-patcher restore <id>             Restore from backup
    claude-session-patcher restore --list           Show available backups
    claude-session-patcher install                  Install to PATH

Exit codes:
    0 -- Healthy / success
    1 -- Issues detected, fixable
    2 -- Issues detected, unfixable
    3 -- Error
"""

from __future__ import annotations

import base64
import binascii
import json
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import typer
from cc_lib import session_tracker
from cc_lib.cli import add_help_command, add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.os_process import ProcessHandle
from cc_lib.picklable import PickleByInitArgs
from cc_lib.types import CCVersion
from cc_lib.utils import (
    encode_project_path,
    get_active_cc_version,
    get_claude_config_home_dir,
    get_claude_workspace_config_home_dir,
    version_in_range,
)
from cc_lib.utils.atomic_write import atomic_write

# -- Constants -----------------------------------------------------------------

BACKUP_DIR = get_claude_workspace_config_home_dir() / 'scripts' / 'claude-session-patcher' / 'backups'
LEGACY_BACKUP_DIR = get_claude_workspace_config_home_dir() / 'claude-session' / 'chain-backups'

EXIT_HEALTHY = 0
EXIT_FIXABLE = 1
EXIT_UNFIXABLE = 2
EXIT_ERROR = 3


# -- App + ErrorBoundary -------------------------------------------------------

app = create_app(help='Claude Code session patcher -- diagnose and repair session corruption.')
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=EXIT_ERROR)


# -- CLI commands (public interface) -------------------------------------------


@app.command()
@error_boundary
def scan(
    project: Path | None = typer.Option(
        None,
        '--project',
        '-p',
        help='Scope to sessions under this project path',
    ),
    json_output: bool = typer.Option(False, '--json', help='Machine-readable JSON output'),
) -> None:
    """Scan all sessions and print summary.

    \b
    Examples:
        scan                              Check all sessions
        scan --project /path/to/project   Scope to one project
        scan --json                       Machine-readable output
    """
    sessions = SessionFile.find_all(project=project)
    if not sessions:
        print('No session files found under ~/.claude/projects/')
        raise SystemExit(EXIT_ERROR)

    print(f'Scanning {len(sessions)} session files...\n')

    current_version = get_active_cc_version()
    categories: dict[str, list[tuple[SessionFile, Sequence[SessionScanResult]]]] = {
        'healthy': [],
        'fixable': [],
        'unfixable': [],
        'error': [],
    }

    for session in sessions:
        try:
            analyzer = SessionAnalyzer(session.read_lines(), sidechain_resolver=session.sidechain_resolver())
            results = analyzer.scan(current_version=current_version)

            detected = [r for r in results if r.status == 'detected']
            unfixable = [r for r in results if r.status == 'unfixable']

            if detected and not unfixable:
                categories['fixable'].append((session, results))
            elif detected or unfixable:
                categories['unfixable'].append((session, results))
            else:
                categories['healthy'].append((session, results))
        except json.JSONDecodeError as e:
            categories['error'].append((session, ()))
            if not json_output:
                print(f'  [ERROR] {session.session_id}: {e}', file=sys.stderr)

    if json_output:
        _print_scan_json(categories)
    else:
        _print_scan_summary(categories)

    if categories['unfixable']:
        raise UnfixableFound(f'{len(categories["unfixable"])} unfixable session(s)')
    if categories['fixable']:
        raise FixableFound(f'{len(categories["fixable"])} fixable session(s)')
    if categories['error']:
        raise SystemExit(EXIT_ERROR)


@app.command()
@error_boundary
def check(
    session_id: str | None = typer.Argument(None, help='Session ID or prefix'),
    file: Path | None = typer.Option(None, '--file', '-f', help='Direct path to session JSONL'),
    json_output: bool = typer.Option(False, '--json', help='Machine-readable JSON output'),
) -> None:
    """Detailed diagnosis with patch documentation.

    \b
    Shows per-patch status, affected lines, and fix approach
    for the diagnosed session.
    """
    if file:
        session = SessionFile(file)
    elif session_id:
        session = SessionFile.find_by_id(session_id)
    else:
        raise SessionPatchError('Provide a session ID or use --file')

    analyzer = SessionAnalyzer(session.read_lines(), sidechain_resolver=session.sidechain_resolver())
    results = analyzer.scan(current_version=get_active_cc_version())

    if json_output:
        _print_check_json(results, session)
    else:
        _print_check(results, session)

    detected = [r for r in results if r.status == 'detected']
    unfixable = [r for r in results if r.status == 'unfixable']

    if unfixable:
        raise UnfixableFound(f'{len(unfixable)} unfixable issue(s)')
    if detected:
        raise FixableFound(f'{len(detected)} fixable patch(es)')


@app.command()
@error_boundary
def fix(
    session_id: str | None = typer.Argument(None, help='Session ID or prefix'),
    all_: bool = typer.Option(False, '--all', help='Fix all fixable sessions'),
    dry_run: bool = typer.Option(False, '--dry-run', '-n', help='Show what would change'),
) -> None:
    """Apply patches with backup, verify, rollback on failure.

    \b
    Rewires are applied first (modify parentUuid in place), then
    truncation (remove records from tail). A backup is created before
    any modification. After patching, the session is re-scanned to
    verify. If verification fails, the backup is automatically restored.
    """
    if all_:
        _fix_all(dry_run=dry_run)
        return

    if not session_id:
        raise SessionPatchError('Provide a session ID or use --all')

    session = SessionFile.find_by_id(session_id)
    if session.is_active():
        raise ActiveSessionError(session.session_id)

    current_version = get_active_cc_version()
    analyzer = SessionAnalyzer(session.read_lines(), sidechain_resolver=session.sidechain_resolver())
    preview_results = analyzer.scan(current_version=current_version)
    detected = [r for r in preview_results if r.status == 'detected']
    pre_unfixable = [r for r in preview_results if r.status == 'unfixable']

    if not detected:
        if pre_unfixable:
            print(f'Session {session.session_id} has unfixable issues:')
            for r in pre_unfixable:
                print(f'  {r.patch.name}: {r.details}')
            raise UnfixableFound(f'{len(pre_unfixable)} unfixable issue(s)')
        print(f'Session {session.session_id} is healthy, nothing to fix.')
        return

    if dry_run:
        _print_check(preview_results, session)
        print('\n(dry run — no changes made)')
        raise FixableFound(f'{len(detected)} fixable patch(es)')

    backup_mgr = BackupManager()
    backup_path = backup_mgr.create(session)
    print(f'Backup: {backup_path}')

    runner = PatchRunner(session)
    try:
        run_result = runner.run()
    except Exception:
        backup_mgr.rollback(session)
        raise

    # Verify: nothing detected or unfixable on the post-run file. Anything that
    # remains is either a regression or an unfixable that the runner couldn't
    # clear; rollback in either case.
    verify_results = SessionAnalyzer(
        session.read_lines(),
        sidechain_resolver=session.sidechain_resolver(),
    ).scan(current_version=current_version)
    verify_bad = [r for r in verify_results if r.status in ('detected', 'unfixable')]

    if verify_bad:
        print('Verification FAILED — restoring from backup', file=sys.stderr)
        for r in verify_bad:
            print(f'  {r.patch.name}: {r.status} — {r.details}', file=sys.stderr)
        backup_mgr.rollback(session)
        raise SystemExit(EXIT_ERROR)

    backup_mgr.record_applied(session, run_result.applied, run_result.iterations)
    print(f'Fixed {session.session_id}: {", ".join(run_result.applied)}')


@app.command()
@error_boundary
def restore(
    session_id: str | None = typer.Argument(None, help='Session ID or prefix'),
    list_: bool = typer.Option(False, '--list', '-l', help='List available backups'),
    force: bool = typer.Option(
        False,
        '--force',
        '-f',
        help='Discard appended content; required when the live session has been modified since backup',
    ),
) -> None:
    """Restore a session from backup.

    \b
    Searches both patcher-backups and legacy chain-backups directories.
    Refuses to overwrite a session with a live Claude process attached, or
    one that has been appended to since backup (use --force to override the
    latter).
    """
    backup_mgr = BackupManager()

    if list_:
        metas = backup_mgr.list_backups()
        if not metas:
            print('No backups found.')
            return
        print('Available backups:\n')
        for meta in metas:
            print(f'  {meta["session_id"]}')
            print(f'    Fixed:    {meta.get("fixed_at", "?")}')
            print(f'    Original: {meta.get("original_path", "?")}')
            print()
        return

    if not session_id:
        raise SessionPatchError('Provide a session ID or use --list')

    original_path = backup_mgr.restore(session_id, force=force)
    print(f'Restored {session_id}')
    print(f'  -> {original_path}')


add_install_command(app, script_path=__file__)


# ──── Patch Definitions & Analysis (pure, no I/O) ────────────────────────────


class PatchKind(str, Enum):
    """What the patch does."""

    FIX = 'fix'  # Corrective — repairs corruption so the session works again


@dataclass(frozen=True, slots=True)
class SessionPatchDef:
    """A known session patch with detection and fix logic.

    `deps` declares which other patches must complete before this one runs.
    The runner topologically orders patches by deps and uses alphabetical
    name as deterministic tiebreak among equally-runnable patches.
    """

    name: str
    description: str
    kind: PatchKind
    detector: Callable[[SessionAnalyzer, SessionPatchDef], SessionScanResult]
    deps: Sequence[str] = ()
    min_version: CCVersion | None = None
    max_version: CCVersion | None = None


class FixDataType:
    """Namespace for per-patch fix data shapes."""

    @dataclass(frozen=True, slots=True)
    class Rewire:
        """Patches that rewire parentUuid pointers."""

        rewire_map: Mapping[int, str]

    @dataclass(frozen=True, slots=True)
    class DuplicateUuid(Rewire):
        """duplicate-uuid patch."""

        duplicate_uuids: Mapping[str, int]

    @dataclass(frozen=True, slots=True)
    class OrphanSidechain(Rewire):
        """orphan-sidechain patch."""

        attributions: Mapping[str, str]

    @dataclass(frozen=True, slots=True)
    class ImageRedaction:
        """One image content block to be replaced with a text placeholder."""

        record_index: int  # 0-based index into analyzer._records
        json_path: Sequence[str | int]  # path from the record root to the image block
        placeholder_text: str
        original_dimensions: tuple[int, int]  # (width, height)
        original_format: str  # 'PNG', 'JPEG', 'GIF', 'WebP'

    @dataclass(frozen=True, slots=True)
    class RedactImageAndTruncate:
        """oversized-image patch: redact image blocks AND truncate error tail."""

        redactions: Sequence[FixDataType.ImageRedaction]
        truncate_to: int | None
        error_loop_start: int | None
        records_to_remove: int


type FixData = FixDataType.DuplicateUuid | FixDataType.OrphanSidechain | FixDataType.RedactImageAndTruncate


@dataclass(frozen=True, slots=True)
class SessionScanResult:
    """Per-patch scan outcome for one session."""

    patch: SessionPatchDef
    status: Literal['clean', 'detected', 'unfixable', 'out_of_range']
    # clean        = detector ran, nothing to fix
    # detected     = detector ran, fixable corruption found
    # unfixable    = detector ran, corruption found but cannot be auto-fixed
    # out_of_range = patch's [min_version, max_version] excludes the active CC
    #                version (detector skipped, no scanning performed)
    details: str
    affected_lines: Sequence[int] = ()
    fix_data: FixData | None = None


# -- Session analyzer ----------------------------------------------------------

# Callback type for sidechain UUID attribution (I/O injected from CLI layer)
type SidechainResolver = Callable[[Set[str]], Mapping[str, str]]


class SessionAnalyzer:
    """Analyzes session JSONL for patch-addressable corruption.

    Encapsulates index building, chain walking, and all detector logic.
    Pure — no I/O. File reading is the caller's responsibility.
    """

    def __init__(
        self,
        lines: Sequence[str],
        *,
        sidechain_resolver: SidechainResolver | None = None,
    ) -> None:
        self._records: list[dict[str, Any] | None] = []
        self._uuid_index: dict[str, int] = {}
        self._uuid_counts: dict[str, int] = {}
        self._duplicate_uuids: dict[str, int] = {}
        self._sidechain_resolver = sidechain_resolver
        self._build_index(lines)

    @property
    def total_records(self) -> int:
        return sum(1 for r in self._records if r is not None)

    @property
    def total_uuids(self) -> int:
        return len(self._uuid_index)

    @property
    def tail_uuid(self) -> str | None:
        for i in range(len(self._records) - 1, -1, -1):
            rec = self._records[i]
            if rec is not None and rec.get('uuid'):
                return str(rec['uuid'])
        return None

    @property
    def tail_line(self) -> int | None:
        for i in range(len(self._records) - 1, -1, -1):
            rec = self._records[i]
            if rec is not None and rec.get('uuid'):
                return i + 1
        return None

    def scan(
        self,
        *,
        patches: Sequence[SessionPatchDef] | None = None,
        current_version: CCVersion | None = None,
    ) -> Sequence[SessionScanResult]:
        """Run all detectors and return results.

        When ``current_version`` is provided, patches whose declared
        ``[min_version, max_version]`` range excludes it short-circuit to
        ``status='out_of_range'`` without invoking the detector. ``None``
        (default) preserves prior behavior — every detector runs.
        """
        if patches is None:
            patches = PATCHES

        results: list[SessionScanResult] = []
        for patch in sorted(patches, key=lambda p: p.name):
            if current_version is not None and not version_in_range(
                current_version, patch.min_version, patch.max_version
            ):
                bound = (
                    f'max_version={patch.max_version}'
                    if patch.max_version is not None
                    else f'min_version={patch.min_version}'
                )
                results.append(
                    SessionScanResult(
                        patch=patch,
                        status='out_of_range',
                        details=f'Skipped: out of range ({bound}; active version: {current_version})',
                    ),
                )
                continue
            results.append(patch.detector(self, patch))
        return results

    def walk_chain(self, start_uuid: str) -> tuple[Set[str], str | None, int]:
        """Walk parentUuid chain from start.

        Returns (visited_uuids, break_uuid_or_None, step_count).
        break_uuid is None if the chain reaches root cleanly.
        """
        current: str | None = start_uuid
        steps = 0
        visited: set[str] = set()

        while current:
            if current in visited:
                return visited, current, steps
            visited.add(current)
            if current in self._uuid_index:
                rec = self._records[self._uuid_index[current]]
                if rec is None:
                    return visited, current, steps
                current = rec.get('parentUuid') or None
            else:
                return visited, current, steps
            steps += 1

        return visited, None, steps

    def find_rewire_target(self, line_idx: int) -> tuple[str, int, str] | None:
        """Find nearest preceding record with a valid UUID in the index.

        Returns (uuid, 1-based line number, type label) or None.
        """
        for j in range(line_idx - 1, -1, -1):
            prev_rec = self._records[j]
            if prev_rec is None:
                continue
            prev_uuid = prev_rec.get('uuid')
            if prev_uuid and prev_uuid in self._uuid_index:
                pt = prev_rec.get('type', '?')
                ps = prev_rec.get('subtype', '')
                label = pt + (f'/{ps}' if ps else '')
                return prev_uuid, j + 1, label
        return None

    def _build_index(self, lines: Sequence[str]) -> None:
        """Parse JSONL lines and build the UUID index, excluding duplicates.

        Records with duplicate UUIDs are kept in `_records` (data preserved)
        but excluded from `_uuid_index`, so `find_rewire_target` and
        `walk_chain` cannot recommend or traverse ambiguous UUIDs. Duplicate
        metadata lives in `_duplicate_uuids` for the duplicate-uuid detector.
        """
        for line in lines:
            if not line.strip():
                self._records.append(None)
                continue
            rec = json.loads(line)
            self._records.append(rec)
            uuid = rec.get('uuid')
            if uuid:
                self._uuid_counts[uuid] = self._uuid_counts.get(uuid, 0) + 1

        self._duplicate_uuids = {u: c for u, c in self._uuid_counts.items() if c > 1}

        for i, rec in enumerate(self._records):
            if rec is None:
                continue
            uuid = rec.get('uuid')
            if uuid and uuid not in self._duplicate_uuids:
                self._uuid_index[uuid] = i


# -- Detector functions --------------------------------------------------------
# Each detector takes (analyzer, patch, **kwargs) and returns SessionScanResult.
# Execution order at apply-time is determined by `SessionPatchDef.deps` (DAG)
# resolved by PatchRunner. scan() iterates alphabetically since it's read-only.


def _detect_duplicate_uuid(
    analyzer: SessionAnalyzer,
    patch: SessionPatchDef,
) -> SessionScanResult:
    """Detect and handle duplicate UUIDs from saved_hook_context.

    Reads `_duplicate_uuids` to find parent pointers targeting duplicate
    UUIDs and computes a rewire map redirecting them to valid predecessors.
    """
    if not analyzer._duplicate_uuids:
        return SessionScanResult(patch=patch, status='clean', details='No duplicate UUIDs')

    # Find records orphaned by the exclusion
    rewire_map: dict[int, str] = {}
    affected: list[int] = []

    for i, rec in enumerate(analyzer._records):
        if rec is None:
            continue
        parent = rec.get('parentUuid')
        if parent and parent in analyzer._duplicate_uuids:
            target = analyzer.find_rewire_target(i)
            if target:
                uuid, _line, _label = target
                rewire_map[i + 1] = uuid
            affected.append(i + 1)

    total_dups = sum(analyzer._duplicate_uuids.values())
    if not affected:
        return SessionScanResult(
            patch=patch,
            status='clean',
            details=f'{len(analyzer._duplicate_uuids)} duplicate UUID(s) but none on active chain',
        )

    return SessionScanResult(
        patch=patch,
        status='detected',
        details=f'{len(analyzer._duplicate_uuids)} duplicate UUID(s) ({total_dups} total occurrences), {len(affected)} orphaned records',
        affected_lines=tuple(affected),
        fix_data=FixDataType.DuplicateUuid(rewire_map=rewire_map, duplicate_uuids=dict(analyzer._duplicate_uuids)),
    )


def _detect_orphan_sidechain(
    analyzer: SessionAnalyzer,
    patch: SessionPatchDef,
) -> SessionScanResult:
    """Detect parentUuid orphans pointing into sidechain agent files.

    Finds records whose parentUuid is not in the uuid_index (after duplicate
    exclusion). For each, finds the nearest valid predecessor as a rewire target.
    """
    orphans: list[tuple[int, str, str | None]] = []  # (line_1based, old_parent, rewire_uuid)
    rewire_map: dict[int, str] = {}

    for i, rec in enumerate(analyzer._records):
        if rec is None:
            continue
        parent = rec.get('parentUuid')
        if not parent or parent in analyzer._uuid_index:
            continue

        target = analyzer.find_rewire_target(i)
        rewire_uuid = target[0] if target else None
        orphans.append((i + 1, parent, rewire_uuid))
        if rewire_uuid:
            rewire_map[i + 1] = rewire_uuid

    if not orphans:
        return SessionScanResult(patch=patch, status='clean', details='No orphan pointers')

    # Check if orphans affect the active chain
    tail = analyzer.tail_uuid
    on_active_chain = False
    if tail:
        _visited, break_uuid, _steps = analyzer.walk_chain(tail)
        on_active_chain = break_uuid is not None

    # Attribute orphans to sidechain files
    attributions: dict[str, str] = {}
    if analyzer._sidechain_resolver:
        orphan_uuids = {parent for _, parent, _ in orphans}
        attributions = dict(analyzer._sidechain_resolver(orphan_uuids))

    unfixable_count = sum(1 for _, _, rw in orphans if rw is None)

    if not on_active_chain:
        return SessionScanResult(
            patch=patch,
            status='clean',
            details=f'{len(orphans)} orphans on dead branches (not affecting active chain)',
        )

    if unfixable_count:
        return SessionScanResult(
            patch=patch,
            status='unfixable',
            details=f'{len(orphans)} orphans, {unfixable_count} have no rewire candidate',
            affected_lines=tuple(line for line, _, _ in orphans),
        )

    # No per-detector simulation — the fix command's verify-then-rollback loop
    # handles correctness holistically across all patches. Per-detector simulation
    # would only see this detector's rewires, missing rewires from duplicate-uuid
    # that may also be needed for a clean chain walk.
    return SessionScanResult(
        patch=patch,
        status='detected',
        details=f'{len(orphans)} orphans on active chain, all rewirable',
        affected_lines=tuple(line for line, _, _ in orphans),
        fix_data=FixDataType.OrphanSidechain(rewire_map=rewire_map, attributions=attributions),
    )


def _detect_stale_parent(
    analyzer: SessionAnalyzer,
    patch: SessionPatchDef,
) -> SessionScanResult:
    """Detect stale parent after resume — chain intact but skips newer segments.

    Detect-only: no auto-fix (rewire target is ambiguous).
    """
    tail = analyzer.tail_uuid
    if not tail:
        return SessionScanResult(patch=patch, status='clean', details='No tail UUID')

    visited, break_uuid, steps = analyzer.walk_chain(tail)

    if break_uuid is not None:
        return SessionScanResult(patch=patch, status='clean', details='Chain broken (handled by other patches)')

    # Chain reaches root — check for unreachable compact_boundary segments
    chain_lines = {analyzer._uuid_index[u] for u in visited if u in analyzer._uuid_index}
    chain_min_line = min(chain_lines) if chain_lines else 0

    all_boundaries = [
        (i, rec)
        for i, rec in enumerate(analyzer._records)
        if rec is not None
        and rec.get('type') == 'system'
        and rec.get('subtype') == 'compact_boundary'
        and rec.get('uuid')
    ]

    stale_segments: list[dict[str, Any]] = []
    for bi, (i, rec) in enumerate(all_boundaries):
        if rec['uuid'] in visited or i <= chain_min_line:
            continue
        # Upper bound of this segment (next boundary start, or end of file)
        deepest = all_boundaries[bi + 1][0] if bi + 1 < len(all_boundaries) else len(analyzer._records)
        stale_segments.append({'root_line': i + 1, 'root_uuid': rec['uuid'], 'deepest_line': deepest})

    if not stale_segments:
        return SessionScanResult(patch=patch, status='clean', details=f'Chain healthy ({steps} steps to root)')

    segment_details = '; '.join(
        f'L{seg["root_line"]} (uuid={seg["root_uuid"][:16]}...) to L{seg["deepest_line"]}' for seg in stale_segments
    )
    return SessionScanResult(
        patch=patch,
        status='unfixable',
        details=f'Chain reaches root ({steps} steps) but skipped {len(stale_segments)} newer segment(s): {segment_details}. Manual rewiring required.',
        affected_lines=tuple(seg['root_line'] for seg in stale_segments),
    )


def _detect_oversized_image(
    analyzer: SessionAnalyzer,
    patch: SessionPatchDef,
) -> SessionScanResult:
    """Detect images exceeding Anthropic's 2000px many-image dimension cap.

    Two conditions required to flag a session:
    1. At least one base64 image block has width or height > 2000px.
    2. At least one <synthetic> assistant record contains the API error
       text 'dimension limit for many-image requests' (confirms the image
       is actively blocking, not merely historical).

    Fix is composite: redact each oversized image (replace the image block
    with a text placeholder in place, preserving tool_result structure) AND
    truncate the synthetic-error tail back to the last clean exchange.
    """
    # Phase 1: find oversized base64 image blocks
    redactions: list[FixDataType.ImageRedaction] = []
    for rec_idx, rec in enumerate(analyzer._records):
        if rec is None:
            continue
        msg = rec.get('message')
        if not isinstance(msg, dict):
            continue
        for path, block in _walk_image_blocks(msg):
            src = block.get('source') or {}
            if src.get('type') != 'base64':
                continue
            dims = _sample_image_dimensions(src.get('data') or '')
            if dims is None:
                continue
            fmt, w, h = dims
            if w > IMAGE_DIMENSION_THRESHOLD or h > IMAGE_DIMENSION_THRESHOLD:
                placeholder = (
                    f'[image redacted by claude-session-patcher -- '
                    f"original {fmt} was {w}x{h} pixels, exceeded Anthropic's "
                    f'{IMAGE_DIMENSION_THRESHOLD}px many-image dimension limit]'
                )
                redactions.append(
                    FixDataType.ImageRedaction(
                        record_index=rec_idx,
                        json_path=('message', *path),
                        placeholder_text=placeholder,
                        original_dimensions=(w, h),
                        original_format=fmt,
                    )
                )

    # Phase 2: find synthetic dimension-limit error records
    dim_error_lines: list[int] = []
    for i, rec in enumerate(analyzer._records):
        if rec is None:
            continue
        if rec.get('type') != 'assistant':
            continue
        msg = rec.get('message') or {}
        if msg.get('model') != '<synthetic>':
            continue
        content = msg.get('content') or []
        if not isinstance(content, list):
            continue
        text_blob = ' '.join(b.get('text', '') for b in content if isinstance(b, dict))
        if 'dimension limit for many-image requests' in text_blob:
            dim_error_lines.append(i + 1)

    # Phase 3: decide status
    if not redactions and not dim_error_lines:
        return SessionScanResult(patch=patch, status='clean', details='No oversized images or dimension errors')

    if not redactions and dim_error_lines:
        return SessionScanResult(
            patch=patch,
            status='unfixable',
            details=(
                f'Dimension error loop at L{dim_error_lines[0]} but no oversized '
                f'base64 image found; may be a URL image or unsupported format'
            ),
            affected_lines=tuple(dim_error_lines),
        )

    if redactions and not dim_error_lines:
        first = redactions[0]
        return SessionScanResult(
            patch=patch,
            status='clean',
            details=(
                f'{len(redactions)} oversized image(s) present but no active error loop '
                f'(primary: L{first.record_index + 1} {first.original_format} '
                f'{first.original_dimensions[0]}x{first.original_dimensions[1]})'
            ),
            affected_lines=tuple(r.record_index + 1 for r in redactions),
        )

    # Both conditions met — compute truncation point.
    #
    # The dim-error fires synchronously when the API rejects a request
    # containing oversized images. After redaction, those images are text
    # placeholders and the same request would succeed on replay, so the
    # synthetic error record no longer represents a real condition and
    # must be removed. But the user message that triggered the rejection
    # is real input the user expects to see honored.
    #
    # Surgical rule: truncate to (first dim-error synthetic - 1). This
    # drops every dim-error synthetic and everything after them (heartbeat
    # system records, additional retry-loop iterations) while preserving
    # the user message(s) that came before. Walking back to the last clean
    # tool_result/assistant — the previous behavior — discarded any pending
    # user message authored between the last assistant turn and the error,
    # which on a typical session means losing the message that made the
    # user run the patcher in the first place.
    error_loop_start = dim_error_lines[0]
    truncate_to = error_loop_start - 1

    if truncate_to <= 0:
        return SessionScanResult(
            patch=patch,
            status='unfixable',
            details=(
                f'{len(redactions)} oversized image(s) and error loop at L{error_loop_start}, '
                f'but no records precede the error loop'
            ),
            affected_lines=tuple(r.record_index + 1 for r in redactions),
        )

    records_to_remove = len(analyzer._records) - truncate_to
    primary = redactions[0]
    details = (
        f'{len(redactions)} oversized image(s) '
        f'(primary: L{primary.record_index + 1} {primary.original_format} '
        f'{primary.original_dimensions[0]}x{primary.original_dimensions[1]}), '
        f'error loop at L{error_loop_start}; redact + truncate to L{truncate_to} '
        f'(remove {records_to_remove} tail records)'
    )
    return SessionScanResult(
        patch=patch,
        status='detected',
        details=details,
        affected_lines=tuple(r.record_index + 1 for r in redactions) + tuple(dim_error_lines),
        fix_data=FixDataType.RedactImageAndTruncate(
            redactions=tuple(redactions),
            truncate_to=truncate_to,
            error_loop_start=error_loop_start,
            records_to_remove=records_to_remove,
        ),
    )


PATCHES: Sequence[SessionPatchDef] = (
    SessionPatchDef(
        name='duplicate-uuid',
        description='saved_hook_context entries share UUIDs, creating cycles in UUID index. Session shows empty on resume.',
        kind=PatchKind.FIX,
        detector=_detect_duplicate_uuid,
        deps=(),
        min_version=CCVersion('2.1.27'),
        max_version=CCVersion('2.1.27'),
    ),
    SessionPatchDef(
        name='orphan-sidechain',
        description='SubAgent/hook progress entries create parentUuid orphans pointing into sidechain files. Session shows empty on resume.',
        kind=PatchKind.FIX,
        detector=_detect_orphan_sidechain,
        # Both upstream patches mutate parent chains; orphan-sidechain must see the post-fix file.
        deps=('duplicate-uuid', 'oversized-image'),
        min_version=CCVersion('2.1.24'),
    ),
    SessionPatchDef(
        name='oversized-image',
        description=(
            'An image >2000px triggers an API 400 loop on sessions with >20 images. '
            'Fix: redact the image in place + truncate the synthetic error tail.'
        ),
        kind=PatchKind.FIX,
        detector=_detect_oversized_image,
        deps=(),
    ),
    SessionPatchDef(
        name='stale-parent',
        description='Resume latches onto old compaction segment, skipping newer compact_boundary records. Session works but missing recent context. Detect-only.',
        kind=PatchKind.FIX,
        detector=_detect_stale_parent,
        # Walks the chain; needs post-fix file from chain-mutating patches.
        deps=('duplicate-uuid', 'oversized-image'),
        min_version=CCVersion('2.1.85'),
    ),
)

PATCHES_BY_NAME: Mapping[str, SessionPatchDef] = {p.name: p for p in PATCHES}


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a PatchRunner.run() invocation."""

    iterations: int
    applied: Sequence[str]  # patch names whose fix was applied, in apply order
    unfixable: Sequence[str]  # patch names that reported 'unfixable'


@dataclass(frozen=True, slots=True)
class _PassResult:
    """Outcome of one DAG sweep within PatchRunner.run()."""

    applied: Sequence[str]
    unfixable: Sequence[str]


class PatchRunner:
    """DAG-ordered patch application against a session file.

    Each iteration is a topological sweep. A patch is runnable when all its
    `deps` have completed in this iteration; alphabetical name order is the
    deterministic tiebreak among equally-runnable patches. For each runnable
    patch the runner re-reads the file, builds a fresh `SessionAnalyzer`, runs
    the detector, and applies the fix to the file if `status == 'detected'`.

    The loop terminates when an iteration applies zero fixes (convergence) or
    when MAX_ITERATIONS is reached (signals a non-idempotent patch — failure).
    """

    MAX_ITERATIONS = 10

    def __init__(self, session: SessionFile, patches: Sequence[SessionPatchDef] = PATCHES) -> None:
        self._session = session
        self._patches = patches

    def run(self) -> RunResult:
        applied: list[str] = []
        unfixable_seen: list[str] = []
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            pass_result = self._run_one_pass()
            applied.extend(pass_result.applied)
            for name in pass_result.unfixable:
                if name not in unfixable_seen:
                    unfixable_seen.append(name)
            if not pass_result.applied:
                return RunResult(iterations=iteration, applied=applied, unfixable=unfixable_seen)
        raise SessionPatchError(
            f'Patcher did not converge after {self.MAX_ITERATIONS} iterations '
            f'(applied: {applied}); a patch may be non-idempotent.',
        )

    def _run_one_pass(self) -> _PassResult:
        completed: set[str] = set()
        applied: list[str] = []
        unfixable: list[str] = []
        while True:
            patch = self._next_runnable(completed)
            if patch is None:
                return _PassResult(applied=applied, unfixable=unfixable)
            completed.add(patch.name)
            outcome = self._run_patch(patch)
            if outcome == 'applied':
                applied.append(patch.name)
            elif outcome == 'unfixable':
                unfixable.append(patch.name)

    def _next_runnable(self, completed: Set[str]) -> SessionPatchDef | None:
        candidates = [p for p in self._patches if p.name not in completed and all(dep in completed for dep in p.deps)]
        return min(candidates, key=lambda p: p.name) if candidates else None

    def _run_patch(self, patch: SessionPatchDef) -> str:
        """Run patch's detector against a fresh analyzer; apply fix if detected.

        Returns one of: 'applied', 'clean', 'unfixable'.
        """
        analyzer = SessionAnalyzer(
            self._session.read_lines(),
            sidechain_resolver=self._session.sidechain_resolver(),
        )
        result = patch.detector(analyzer, patch)
        if result.status == 'unfixable':
            return 'unfixable'
        if result.status != 'detected' or result.fix_data is None:
            return 'clean'
        _apply_single_patch(self._session, result.fix_data)
        return 'applied'


def _apply_single_patch(session: SessionFile, fix_data: FixData) -> None:
    """Apply one patch's fix data to the session file."""
    lines = list(session.read_lines())

    match fix_data:
        case FixDataType.RedactImageAndTruncate(
            redactions=redactions,
            truncate_to=truncate_to,
        ):
            redactions_by_line: dict[int, list[FixDataType.ImageRedaction]] = {}
            for r in redactions:
                redactions_by_line.setdefault(r.record_index, []).append(r)
            for idx, reds in redactions_by_line.items():
                rec = json.loads(lines[idx])
                for r in reds:
                    _apply_image_redaction(rec, r)
                lines[idx] = json.dumps(rec) + '\n'
            if truncate_to is not None:
                lines = lines[:truncate_to]
        case FixDataType.Rewire(rewire_map=rewire_map):
            # Matches DuplicateUuid and OrphanSidechain subclasses too.
            for line_num, new_parent in rewire_map.items():
                idx = line_num - 1
                rec = json.loads(lines[idx])
                rec['parentUuid'] = new_parent
                lines[idx] = json.dumps(rec) + '\n'

    session.write_lines(lines)


# -- Domain classes ------------------------------------------------------------


class SessionFile:
    """Session JSONL file handle with discovery methods."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> str:
        return self._path.stem

    @property
    def project_dir(self) -> Path:
        return self._path.parent

    def read_lines(self) -> Sequence[str]:
        return self._path.read_text().splitlines(keepends=True)

    def write_lines(self, lines: Sequence[str]) -> None:
        atomic_write(self._path, ''.join(lines).encode(), reference=self._path)

    def is_active(self) -> bool:
        """Check if a Claude process is currently running on this session.

        Looks up the session in `cc_lib.session_tracker`'s database and verifies
        the tracked claude_pid via ``ProcessHandle.is_alive`` (recycle-defense via
        the ``process_created_at`` anchor). Catches sessions whether they're
        actively writing or sitting idle — both states matter for safety.
        """
        db = session_tracker.load_sessions(str(self._path.parent))
        for session in db.sessions:
            if session.session_id == self.session_id and session.state == 'active':
                return ProcessHandle(session.metadata.claude_pid, session.metadata.process_created_at).is_alive()
        return False

    def sidechain_resolver(self) -> SidechainResolver:
        """Return a resolver callback for sidechain UUID attribution."""
        path = self._path

        def resolve(orphan_uuids: Set[str]) -> Mapping[str, str]:
            return _attribute_orphans(path, orphan_uuids)

        return resolve

    @classmethod
    def find_all(cls, *, project: Path | None = None) -> Sequence[SessionFile]:
        """Find all session JSONL files, optionally scoped to a project."""
        projects_dir = get_claude_config_home_dir() / 'projects'
        if not projects_dir.exists():
            return []
        if project:
            encoded = encode_project_path(project)
            target_dir = projects_dir / encoded
            if not target_dir.is_dir():
                return []
            dirs = [target_dir]
        else:
            dirs = [d for d in projects_dir.iterdir() if d.is_dir()]
        results: list[SessionFile] = []
        for d in dirs:
            results.extend(cls(f) for f in d.iterdir() if f.suffix == '.jsonl' and not f.name.startswith('agent-'))
        return sorted(results, key=lambda s: s.path)

    @classmethod
    def find_by_id(cls, session_id: str) -> SessionFile:
        """Find by ID or prefix. Raises SessionNotFoundError on no/ambiguous match."""
        all_sessions = cls.find_all()
        matches = [s for s in all_sessions if s.session_id.startswith(session_id)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact = [s for s in matches if s.session_id == session_id]
            if len(exact) == 1:
                return exact[0]
            raise SessionPatchError(
                f"Ambiguous prefix '{session_id}', matches {len(matches)} files:\n"
                + '\n'.join(f'  {s.session_id}' for s in matches),
            )
        raise SessionNotFoundError(session_id)


class BackupManager:
    """Manages session backups with metadata sidecars."""

    def __init__(self, backup_dir: Path = BACKUP_DIR) -> None:
        self._dir = backup_dir

    def create(self, session: SessionFile) -> Path:
        """Create backup + skeleton meta. Returns backup path.

        Meta is finalized via `record_applied` after the run completes.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        backup_path = self._dir / f'{session.session_id}.jsonl'
        meta_path = self._dir / f'{session.session_id}.meta.json'
        if backup_path.exists():
            raise BackupExistsError(backup_path, session.session_id)
        atomic_write(backup_path, session.path.read_bytes(), reference=session.path)
        meta = {
            'session_id': session.session_id,
            'original_path': str(session.path),
            'backup_path': str(backup_path),
            'created_at': datetime.now(UTC).isoformat(),
            'tool_version': 'claude-session-patcher v1',
            'patches_applied': [],
        }
        atomic_write(meta_path, (json.dumps(meta, indent=2) + '\n').encode())
        return backup_path

    def record_applied(self, session: SessionFile, applied: Sequence[str], iterations: int) -> None:
        """Update meta after a successful run with applied patches and iteration count."""
        _backup_path, meta_path, meta = self._find_backup(session.session_id)
        updated = dict(meta)
        updated['fixed_at'] = datetime.now(UTC).isoformat()
        updated['patches_applied'] = list(applied)
        updated['iterations'] = iterations
        atomic_write(meta_path, (json.dumps(updated, indent=2) + '\n').encode())

    def rollback(self, session: SessionFile) -> None:
        """Restore from backup and clean up. Used on fix verification failure."""
        backup_path, meta_path, _meta = self._find_backup(session.session_id)
        atomic_write(session.path, backup_path.read_bytes(), reference=backup_path)
        backup_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    def restore(self, session_id: str, *, force: bool = False) -> Path:
        """Restore and clean up. Returns original path.

        Refuses if a Claude process is currently running on the target session.
        Refuses (with a hazard message) if the live file has been modified after
        the backup was taken — set `force=True` to discard the appended content.
        """
        backup_path, meta_path, meta = self._find_backup(session_id)
        original_path = Path(meta['original_path'])

        if original_path.exists():
            candidate = SessionFile(original_path)
            if candidate.is_active():
                raise ActiveSessionError(candidate.session_id)
            # File size differs => content differs. mtime is unreliable here
            # because atomic_write preserves it via reference=, so size is the
            # cheap, robust signal that the live file diverged from the backup.
            if not force and original_path.stat().st_size != backup_path.stat().st_size:
                live_lines = sum(1 for _ in original_path.open())
                backup_lines = sum(1 for _ in backup_path.open())
                raise RestoreOverwriteError(candidate.session_id, live_lines, backup_lines)

        atomic_write(original_path, backup_path.read_bytes(), reference=backup_path)
        if not original_path.exists() or original_path.stat().st_size != backup_path.stat().st_size:
            raise SessionPatchError(f'Restore verification failed, backup preserved at {backup_path}')
        backup_path.unlink()
        meta_path.unlink()
        return original_path

    def list_backups(self) -> Sequence[Mapping[str, Any]]:  # strict_typing_linter.py: loose-typing — meta schema varies
        """List all available backups with metadata from both directories."""
        metas: list[Mapping[str, Any]] = []
        for dir_ in (self._dir, LEGACY_BACKUP_DIR):
            if dir_.exists():
                metas.extend(json.loads(p.read_text()) for p in sorted(dir_.glob('*.meta.json')))
        return metas

    def _find_backup(self, session_id: str) -> tuple[Path, Path, Mapping[str, Any]]:
        """Find backup by session ID prefix. Searches patcher-backups then legacy."""
        for dir_ in (self._dir, LEGACY_BACKUP_DIR):
            if not dir_.exists():
                continue
            matches = list(dir_.glob(f'{session_id}*.meta.json'))
            if len(matches) == 1:
                meta = json.loads(matches[0].read_text())
                sid = meta['session_id']
                backup_path = dir_ / f'{sid}.jsonl'
                if not backup_path.exists():
                    raise SessionPatchError(f'Backup file missing: {backup_path}')
                return backup_path, matches[0], meta
            if len(matches) > 1:
                raise SessionPatchError(
                    f"Ambiguous prefix '{session_id}', matches:\n"
                    + '\n'.join(f'  {m.stem.removesuffix(".meta")}' for m in matches),
                )
        raise SessionPatchError(f"No backup found for '{session_id}'")


# -- Private helpers -----------------------------------------------------------


IMAGE_DIMENSION_THRESHOLD = 2000  # Anthropic's many-image per-request cap
IMAGE_HEADER_B64_SAMPLE = 524288  # ~384KB of image data, plenty for any header


def _image_dimensions(data: bytes) -> tuple[str, int, int] | None:
    """Parse image header; return (format, width, height), or None if unknown."""
    if len(data) < 24:
        return None
    # PNG: 89 50 4E 47 0D 0A 1A 0A + IHDR chunk (width @ 16, height @ 20, big-endian)
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return ('PNG', int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big'))
    # JPEG: SOI FF D8, then scan for SOFn marker
    if data[:2] == b'\xff\xd8':
        return _parse_jpeg_dimensions(data)
    # GIF87a / GIF89a: width @ 6 (LE), height @ 8 (LE)
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return ('GIF', int.from_bytes(data[6:8], 'little'), int.from_bytes(data[8:10], 'little'))
    # WebP: RIFF....WEBP + VP8/VP8L/VP8X chunk
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return _parse_webp_dimensions(data)
    return None


def _parse_jpeg_dimensions(data: bytes) -> tuple[str, int, int] | None:
    """Scan JPEG markers for SOFn and read dimensions."""
    i = 2  # skip SOI (FF D8)
    n = len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        # SOF markers (non-arithmetic, non-differential frames): C0-C3, C5-C7, C9-CB, CD-CF.
        # Exclude C4 (DHT), C8 (reserved), CC (DAC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = int.from_bytes(data[i + 5 : i + 7], 'big')
            w = int.from_bytes(data[i + 7 : i + 9], 'big')
            return ('JPEG', w, h)
        seg_len = int.from_bytes(data[i + 2 : i + 4], 'big')
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def _parse_webp_dimensions(data: bytes) -> tuple[str, int, int] | None:
    """Parse VP8/VP8L/VP8X chunk dimensions."""
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b'VP8 ':
        # Lossy VP8: width/height packed at offset 26 as 14-bit LE
        w = int.from_bytes(data[26:28], 'little') & 0x3FFF
        h = int.from_bytes(data[28:30], 'little') & 0x3FFF
        return ('WebP', w, h)
    if chunk == b'VP8L':
        # Lossless VP8L: (width-1) + (height-1), 14 bits each, packed from byte 21
        b = data[21:25]
        w = 1 + (b[0] | ((b[1] & 0x3F) << 8))
        h = 1 + ((b[1] >> 6) | (b[2] << 2) | ((b[3] & 0x0F) << 10))
        return ('WebP', w, h)
    if chunk == b'VP8X':
        # Extended VP8X: (width-1) + (height-1), 24-bit LE at offset 24, 27
        w = 1 + int.from_bytes(data[24:27], 'little')
        h = 1 + int.from_bytes(data[27:30], 'little')
        return ('WebP', w, h)
    return None


def _walk_image_blocks(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> Sequence[tuple[tuple[str | int, ...], Mapping[str, Any]]]:
    """Collect (json_path, image_block) for every image content block in `value`.

    Images can live at varying depths inside a session record: directly in
    message.content, nested inside tool_result content arrays, or deeper.
    """
    found: list[tuple[tuple[str | int, ...], Mapping[str, Any]]] = []
    if isinstance(value, dict):
        if value.get('type') == 'image':
            # An image block is a leaf for this search: its `source` dict is a
            # descriptor, not a content container for further image blocks.
            found.append((path, value))
            return found
        for k, v in value.items():
            found.extend(_walk_image_blocks(v, (*path, k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(_walk_image_blocks(v, (*path, i)))
    return found


def _sample_image_dimensions(b64_data: Any) -> tuple[str, int, int] | None:
    """Decode a sample of base64 and parse image header dimensions.

    Typed `Any` because the caller reads from potentially-malformed session
    records — `source.data` should be a string but can be anything in broken
    sessions. Malformed inputs silently return None (skip the image) rather
    than crashing the scan.
    """
    if not isinstance(b64_data, str) or not b64_data:
        return None
    try:
        sample = base64.b64decode(b64_data[:IMAGE_HEADER_B64_SAMPLE], validate=False)
    except (binascii.Error, ValueError, TypeError):
        return None
    return _image_dimensions(sample)


def _apply_image_redaction(
    rec: dict[str, Any],  # strict_typing_linter.py: mutable-type — rec is mutated in place
    redaction: FixDataType.ImageRedaction,
) -> None:
    """Navigate rec via redaction.json_path and replace the image block with text."""
    path = redaction.json_path
    if not path:
        return
    parent: Any = rec
    for key in path[:-1]:
        parent = parent[key]
    last_key = path[-1]
    parent[last_key] = {'type': 'text', 'text': redaction.placeholder_text}


def _attribute_orphans(file_path: Path, orphan_uuids: Set[str]) -> Mapping[str, str]:
    """Search sidechain agent files for orphan UUIDs. Returns uuid -> agent filename."""
    if not orphan_uuids:
        return {}

    session_id = file_path.stem
    project_dir = file_path.parent
    attribution: dict[str, str] = {}

    agent_files: list[Path] = []
    subagents_dir = project_dir / session_id / 'subagents'
    if subagents_dir.exists():
        agent_files.extend(subagents_dir.glob('agent-*.jsonl'))
    agent_files.extend(project_dir.glob(f'agent-*-clone-{session_id[:8]}*.jsonl'))
    agent_files.extend(project_dir.glob('agent-*.jsonl'))

    remaining = set(orphan_uuids)
    for af in agent_files:
        if not remaining:
            break
        with open(af) as f:
            for line in f:
                rec = json.loads(line)
                uuid = rec.get('uuid', '')
                if uuid in remaining:
                    attribution[uuid] = af.name
                    remaining.discard(uuid)
                    if not remaining:
                        break

    return attribution


def _fix_all(*, dry_run: bool) -> None:
    """Fix all fixable sessions."""
    sessions = SessionFile.find_all()
    targets: list[tuple[SessionFile, Sequence[SessionScanResult]]] = []
    current_version = get_active_cc_version()

    for session in sessions:
        try:
            analyzer = SessionAnalyzer(session.read_lines(), sidechain_resolver=session.sidechain_resolver())
            results = analyzer.scan(current_version=current_version)
            detected = [r for r in results if r.status == 'detected']
            unfixable = [r for r in results if r.status == 'unfixable']
            if detected and not unfixable:
                targets.append((session, results))
        except json.JSONDecodeError:
            continue

    if not targets:
        print('No fixable sessions found.')
        return

    if dry_run:
        print(f'{len(targets)} fixable sessions:\n')
        for session, results in targets:
            names = ', '.join(r.patch.name for r in results if r.status == 'detected')
            print(f'  {session.session_id}  [{names}]')
        print('\n(dry run — no changes made)')
        raise FixableFound(f'{len(targets)} fixable session(s)')

    print(f'Fixing {len(targets)} sessions...\n')
    backup_mgr = BackupManager()
    failed = 0

    for session, _results in targets:
        try:
            if session.is_active():
                print(f'  [SKIP] {session.session_id}  (in active use)', file=sys.stderr)
                failed += 1
                continue

            backup_mgr.create(session)
            try:
                run_result = PatchRunner(session).run()

                verify_results = SessionAnalyzer(
                    session.read_lines(),
                    sidechain_resolver=session.sidechain_resolver(),
                ).scan(current_version=current_version)
                verify_bad = [r for r in verify_results if r.status in ('detected', 'unfixable')]

                if verify_bad:
                    backup_mgr.rollback(session)
                    print(f'  [FAILED] {session.session_id}  verification failed, rolled back', file=sys.stderr)
                    failed += 1
                else:
                    backup_mgr.record_applied(session, run_result.applied, run_result.iterations)
                    print(f'  [OK] {session.session_id}  [{", ".join(run_result.applied)}]')
            except Exception:
                backup_mgr.rollback(session)
                raise
        except BackupExistsError:
            print(f'  [SKIP] {session.session_id}  (backup exists)', file=sys.stderr)
            failed += 1

    success = len(targets) - failed
    if success:
        print(f'\n{success} sessions fixed. Backups in {BACKUP_DIR}/')
    if failed:
        print(f'{failed} sessions could not be fixed.', file=sys.stderr)
        raise SystemExit(EXIT_ERROR)


# -- Display helpers -----------------------------------------------------------


def _print_check(
    results: Sequence[SessionScanResult],
    session: SessionFile,
) -> None:
    """Print detailed check output."""
    print(f'Session: {session.session_id}')
    print(f'File:    {session.path}')
    print()

    detected = [r for r in results if r.status == 'detected']
    unfixable = [r for r in results if r.status == 'unfixable']

    for r in results:
        status_str = 'SKIPPED' if r.status == 'out_of_range' else r.status.upper()
        tag = f'[{r.patch.kind.value}]'
        print(f'  {r.patch.name:<24s} {tag:<8s} {status_str}')
        if r.status == 'out_of_range':
            print(f'    {r.details}')
        if r.status in ('detected', 'unfixable'):
            print(f'    {r.details}')
            if isinstance(r.fix_data, FixDataType.RedactImageAndTruncate):
                print(f'    Patch: redact {len(r.fix_data.redactions)} oversized image block(s)')
                for red in r.fix_data.redactions:
                    w, h = red.original_dimensions
                    print(f'      - L{red.record_index + 1} {red.original_format} {w}x{h}')
                if r.fix_data.truncate_to is not None:
                    print(
                        f'    Then: truncate to L{r.fix_data.truncate_to} '
                        f'(remove {r.fix_data.records_to_remove} tail records)'
                    )
            elif isinstance(r.fix_data, FixDataType.OrphanSidechain):
                print(f'    Patch: rewire {len(r.fix_data.rewire_map)} parentUuid pointer(s)')
                for uuid, agent in r.fix_data.attributions.items():
                    print(f'    Source: {uuid[:16]}... from {agent}')
            elif isinstance(r.fix_data, FixDataType.Rewire):
                print(f'    Patch: rewire {len(r.fix_data.rewire_map)} parentUuid pointer(s)')
        print()

    if detected and unfixable:
        print(f'Status: PARTIALLY FIXABLE ({len(detected)} fixable, {len(unfixable)} unfixable)')
    elif detected:
        print(f'Status: FIXABLE ({len(detected)} patch{"es" if len(detected) != 1 else ""} to apply)')
    elif unfixable:
        print(f'Status: UNFIXABLE ({len(unfixable)} issue{"s" if len(unfixable) != 1 else ""})')
    else:
        print('Status: HEALTHY')


def _print_check_json(
    results: Sequence[SessionScanResult],
    session: SessionFile,
) -> None:
    """Print machine-readable JSON check output."""
    detected = [r for r in results if r.status == 'detected']
    unfixable = [r for r in results if r.status == 'unfixable']
    output = {
        'session_id': session.session_id,
        'file': str(session.path),
        'fixable': bool(detected) and not unfixable,
        'results': [
            {
                'patch': r.patch.name,
                'kind': r.patch.kind.value,
                'status': r.status,
                'details': r.details,
                'affected_lines': list(r.affected_lines),
            }
            for r in results
        ],
    }
    print(json.dumps(output, indent=2))


def _print_scan_summary(
    categories: Mapping[str, Sequence[tuple[SessionFile, Sequence[SessionScanResult]]]],
) -> None:
    """Print human-readable scan summary."""
    print(f'Healthy:    {len(categories["healthy"]):>4d}')
    print(f'Fixable:    {len(categories["fixable"]):>4d}')
    print(f'Unfixable:  {len(categories["unfixable"]):>4d}')
    print(f'Errors:     {len(categories["error"]):>4d}')

    if categories['fixable']:
        print('\nFixable sessions:')
        for session, results in categories['fixable']:
            names = ', '.join(r.patch.name for r in results if r.status == 'detected')
            proj = session.project_dir.name
            print(f'  {session.session_id}  [{names}]  ({proj})')
        print(f"\nRun 'claude-session-patcher fix --all' to fix all {len(categories['fixable'])} sessions.")

    if categories['unfixable']:
        print('\nUnfixable sessions:')
        for session, results in categories['unfixable']:
            names = ', '.join(r.patch.name for r in results if r.status == 'unfixable')
            proj = session.project_dir.name
            print(f'  {session.session_id}  [{names}]  ({proj})')


def _print_scan_json(
    categories: Mapping[str, Sequence[tuple[SessionFile, Sequence[SessionScanResult]]]],
) -> None:
    """Print machine-readable scan summary."""
    output = {
        'summary': {cat: len(items) for cat, items in categories.items()},
        'sessions': {
            cat: [
                {
                    'session_id': session.session_id,
                    'file': str(session.path),
                    'patches': [r.patch.name for r in results if r.status != 'clean'],
                }
                for session, results in items
            ]
            for cat, items in categories.items()
            if items
        },
    }
    print(json.dumps(output, indent=2))


# -- Exceptions & error handlers -----------------------------------------------


class SessionPatchError(Exception):
    """Base for session patcher errors."""


class SessionNotFoundError(PickleByInitArgs, SessionPatchError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found")


class BackupExistsError(PickleByInitArgs, SessionPatchError):
    def __init__(self, backup_path: Path, session_id: str) -> None:
        self.backup_path = backup_path
        self.session_id = session_id
        super().__init__(
            f'Backup already exists: {backup_path}\n'
            f"Run 'claude-session-patcher restore {session_id}' first, "
            f'or manually remove the backup.',
        )


class ActiveSessionError(PickleByInitArgs, SessionPatchError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f'Session {session_id} appears to be in active use by Claude. Exit Claude Code first, then retry.',
        )


class RestoreOverwriteError(PickleByInitArgs, SessionPatchError):
    """Raised when the live file has been modified since the backup was taken."""

    def __init__(self, session_id: str, live_lines: int, backup_lines: int) -> None:
        self.session_id = session_id
        self.live_lines = live_lines
        self.backup_lines = backup_lines
        delta = live_lines - backup_lines
        if delta > 0:
            change = f'{delta} record(s) appended'
        elif delta < 0:
            change = f'{-delta} record(s) removed'
        else:
            change = 'records modified in place'
        super().__init__(
            f'Restoring {session_id} would discard live changes ({change}; '
            f'live: {live_lines}, backup: {backup_lines}). Use --force to overwrite anyway.',
        )


class FixableFound(SessionPatchError):
    """Raised when fixable patches are detected. Exits with EXIT_FIXABLE."""


class UnfixableFound(SessionPatchError):
    """Raised when only unfixable/detect-only patches are found. Exits with EXIT_UNFIXABLE."""


@error_boundary.handler(FixableFound)
def _handle_fixable(exc: FixableFound) -> None:
    sys.exit(EXIT_FIXABLE)


@error_boundary.handler(UnfixableFound)
def _handle_unfixable(exc: UnfixableFound) -> None:
    sys.exit(EXIT_UNFIXABLE)


@error_boundary.handler(SessionPatchError)
def _handle_patch_error(exc: SessionPatchError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_crash(exc: Exception) -> None:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    for frame in traceback.format_tb(exc.__traceback__)[-2:]:
        print(frame.rstrip(), file=sys.stderr)


# -- Entry point ---------------------------------------------------------------

if __name__ == '__main__':
    run_app(app)
