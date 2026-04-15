#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "pydantic>=2.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
"""Transparent proxy for ``claude`` with tab completions and environment fixes.

Provides zsh tab completion for all Claude Code flags (51 documented + 23
hidden, from v2.1.92 binary analysis), subcommands with descriptions,
model aliases, and project-scoped session completion with custom titles
for ``--resume``.

Also fixes environment issues that the bare ``claude`` binary doesn't handle:
    - Activates ``$PWD/.venv/`` and cleans stale/IDE-injected PATH entries
    - Injects ``--effort`` from settings.json env block
    - Resolves ``--resume <title>`` to session UUIDs

Usage::

    claude-exec [claude args...]      # all args pass through to claude
    claude-exec ext install           # install launcher + completions to PATH
    claude-exec ext uninstall         # remove launcher + completions

All arguments pass through to the real binary via ``os.execv`` — the wrapper
process is replaced entirely, the terminal talks directly to Claude Code.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import mmap
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import pydantic
import rich.console
import typer
from cc_lib.cli import LauncherInstaller, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.utils import encode_project_path, get_claude_config_home_dir, get_claude_workspace_config_home_dir

boundary = ErrorBoundary(exit_code=1)
ext_app = create_app(help='Claude exec management commands.')

COMPLETION_DIR = Path.home() / '.config' / 'zsh' / 'completions'


@boundary
def main() -> None:
    """Entry point: intercept ``ext`` subcommand or activate venv + exec claude."""
    if len(sys.argv) > 1 and sys.argv[1] == 'ext':
        sys.argv = [sys.argv[0], *sys.argv[2:]]  # strip 'ext' so typer sees subcommands
        run_app(ext_app)
        return

    args = _inject_effort_flag(sys.argv[1:])
    args = _resolve_resume_arg(args)

    # Worktree venv takes priority over CWD venv.
    # Must run after _resolve_resume_arg so title→UUID is resolved.
    venv_bin = Path.cwd() / '.venv' / 'bin'
    worktree_venv = _resolve_worktree_venv(args)
    if worktree_venv:
        _activate_venv(worktree_venv)
    elif venv_bin.is_dir():
        _activate_venv(venv_bin.parent)

    binary = _resolve_binary()
    # Intentional: show the full exec line for development visibility.
    # This is a single-user development tool, not a shared deployment.
    print(f'claude-exec: claude {" ".join(str(a) for a in args)}', file=sys.stderr)
    os.execv(binary, [binary, *args])


# -- ext commands --------------------------------------------------------------


@ext_app.command()
def install() -> None:
    """Install launcher and zsh completions to PATH."""
    console = rich.console.Console(stderr=True)

    launcher = LauncherInstaller(Path(__file__).resolve())
    path = launcher.install('claude-exec')
    console.print(f'Launcher: [bold]{path}[/bold]')
    console.print(f'  -> {Path(__file__).resolve()}')

    COMPLETION_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMPLETION_DIR / '_claude-exec'
    dest.write_text(_generate_zsh_completion())
    console.print(f'Completion: [bold]{dest}[/bold]')

    console.print('\nRestart shell to activate.')


@ext_app.command(name='resume-completions', hidden=True)
def resume_completions(
    project: Annotated[str, typer.Argument(help='Project path (default: CWD)')] = '',
) -> None:
    """Output session completions for zsh (hidden — called by tab completion)."""
    index = SessionIndex(project or os.getcwd())
    for entry in index.completions():
        title = entry.title.replace(':', '\\:')
        print(f'{entry.session_id}\t{title}\t{entry.mtime:.0f}')


@ext_app.command()
def uninstall() -> None:
    """Remove launcher and zsh completions."""
    console = rich.console.Console(stderr=True)

    launcher_path = Path.home() / '.local' / 'bin' / 'claude-exec'
    if launcher_path.exists():
        launcher_path.unlink()
        console.print(f'Removed: {launcher_path}')

    completion_path = COMPLETION_DIR / '_claude-exec'
    if completion_path.exists():
        completion_path.unlink()
        console.print(f'Removed: {completion_path}')


# -- Venv activation -----------------------------------------------------------


def _activate_venv(venv_path: Path) -> None:
    """Set env vars, clean stale paths, and prepend .venv/bin to PATH.

    Removes stale ``.venv/bin`` entries from other projects (e.g., PyCharm's
    JediTerm auto-activates the main project's venv, which is wrong in a
    worktree) and IDE-specific paths that shouldn't leak into Claude's
    subprocess environment.
    """
    venv_bin = str(venv_path / 'bin')
    os.environ['VIRTUAL_ENV'] = str(venv_path)
    os.environ['CLAUDE_EXEC_VENV'] = str(venv_path)
    os.environ['PATH'] = venv_bin + ':' + _clean_path(venv_bin)


def _clean_path(keep: str) -> str:
    """Remove stale .venv/bin entries and IDE-injected paths from PATH."""
    entries = os.environ.get('PATH', '').split(':')
    removed: list[str] = []
    cleaned: list[str] = []
    for e in entries:
        if e != keep and _is_stale_path(e):
            removed.append(e)
        else:
            cleaned.append(e)
    if removed:
        print(f'claude-exec: cleaned {len(removed)} PATH entries:', file=sys.stderr)
        for r in removed:
            print(f'  - {r}', file=sys.stderr)
    return ':'.join(cleaned)


def _is_stale_path(entry: str) -> bool:
    """True if this PATH entry is a .venv/bin from another project or prior activation."""
    return '/.venv/bin' in entry


# -- Effort flag injection -----------------------------------------------------


def _inject_effort_flag(args: Sequence[str]) -> Sequence[str]:
    """Pass ``--effort`` if configured in settings.json and not already on the command line.

    Reads ``CLAUDE_CODE_EFFORT_LEVEL`` from the settings.json env block
    directly, because the env block isn't applied to ``process.env`` until
    inside the Claude binary — our launcher runs before that.

    The settings.json env block correctly sets the env var for API inference,
    but ``appState.effortValue`` is populated from the top-level
    ``effortLevel`` setting (not the env var), causing the ``/model`` picker
    to show the wrong effort level. Passing ``--effort`` explicitly on the
    CLI sets ``appState`` correctly at startup.

    Related bugs:
        https://github.com/anthropics/claude-code/issues/35155
        https://github.com/anthropics/claude-code/issues/39015
        https://github.com/anthropics/claude-code/issues/39846
    """
    if '--effort' in args:
        return args

    effort = os.environ.get('CLAUDE_CODE_EFFORT_LEVEL', '') or _read_settings_effort()
    if not effort:
        return args

    try:
        effort = pydantic.TypeAdapter(EffortLevel).validate_python(effort)
    except pydantic.ValidationError as e:
        raise LaunchError(f'CLAUDE_CODE_EFFORT_LEVEL={effort!r}: {e.errors()[0]["msg"]}') from None

    return [*args, '--effort', effort]


type EffortLevel = Literal['low', 'medium', 'high', 'max']


def _read_settings_effort() -> EffortLevel | str:
    """Read and validate CLAUDE_CODE_EFFORT_LEVEL from ~/.claude/settings.json env block."""
    settings_path = Path.home() / '.claude' / 'settings.json'
    if not settings_path.exists():
        return ''
    data = json.loads(settings_path.read_text())
    env_block = data.get('env')
    if not isinstance(env_block, dict):
        return ''
    value = str(env_block.get('CLAUDE_CODE_EFFORT_LEVEL', ''))
    if not value:
        return ''
    try:
        return pydantic.TypeAdapter(EffortLevel).validate_python(value)
    except pydantic.ValidationError as e:
        raise LaunchError(f'CLAUDE_CODE_EFFORT_LEVEL={value!r} in {settings_path}: {e.errors()[0]["msg"]}') from None


# -- Resume title resolution ---------------------------------------------------


def _resolve_resume_arg(args: Sequence[str]) -> Sequence[str]:
    """Resolve ``--resume <title>`` to ``--resume <session-id>`` if the value matches a custom title.

    Claude Code's ``--resume`` flag cannot find sessions by custom title.
    The resume picker searches the stat-log index and PID-keyed session
    files — both ephemeral — but never reads the durable ``custom-title``
    records in the JSONL files where titles actually live. This is a known,
    widely-reported bug with no Anthropic response or fix timeline.

    This function bridges the gap: if the ``--resume`` value matches a
    session's custom title (case-insensitive), it replaces the value with
    the session UUID that Claude Code can resolve. If the value is already
    a UUID or matches nothing, it passes through unchanged.

    Removable: delete this function and the call in ``main()`` when Claude
    Code fixes title-based session lookup.

    Related bugs:
        https://github.com/anthropics/claude-code/issues/47550
            /resume does not find sessions by custom title
        https://github.com/anthropics/claude-code/issues/47158
            sessions-index.json does not index custom-title records
        https://github.com/anthropics/claude-code/issues/46371
            /rename prints a resume command that doesn't work
        https://github.com/anthropics/claude-code/issues/43963
            /resume does not search by session names
        https://github.com/anthropics/claude-code/issues/26134
            /rename does not persist where --resume can find it
    """
    args = list(args)
    for flag in ('--resume', '-r'):
        if flag not in args:
            continue
        idx = args.index(flag)
        if idx + 1 >= len(args):
            break
        value = args[idx + 1]
        normalized = value.lower().strip()
        for entry in SessionIndex(os.getcwd()).completions():
            if entry.title and entry.title.lower().strip() == normalized:
                print(
                    f'claude-exec: --resume {value!r} → {entry.session_id}',
                    file=sys.stderr,
                )
                args[idx + 1] = entry.session_id
                return args
        break
    return args


# -- Zsh completion generation -------------------------------------------------


def _generate_zsh_completion() -> str:
    """Generate a static zsh completion script from FlagDef data.

    Uses ``_arguments`` for rich flag completions (descriptions, file paths,
    choice lists) and a dynamic ``_claude_exec_sessions`` function for
    ``--resume`` session ID completion from ``~/.claude/projects/``.

    No Python callback on TAB — everything runs in native zsh.
    """
    completions_path = Path(__file__).parent / 'claude-exec-completions.py'
    spec = importlib.util.spec_from_file_location('claude_exec_completions', completions_path)
    if spec is None or spec.loader is None:
        raise LaunchError(f'failed to load completions from {completions_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod.__name__] = mod
    spec.loader.exec_module(mod)

    lines: list[str] = [
        '#compdef claude-exec',
        '',
        '# Auto-generated from claude-exec-completions.py (FlagDef data)',
        f'# Claude Code v2.1.92, {__import__("datetime").date.today().isoformat()}',
        '',
    ]

    # Dynamic session completion: calls claude-exec ext resume-completions
    # which uses mmap+rfind for fast title extraction with incremental caching.
    # Output: tab-separated session-id, title, mtime — one per line.
    lines.extend(
        [
            '_claude_exec_sessions() {',
            '  local -a sessions',
            '  local line sid title',
            "  while IFS=$'\\t' read -r sid title _mtime; do",
            '    title="${title//:/\\\\:}"',
            '    if [[ -n "$title" ]]; then',
            '      sessions+=("${sid}:${title}" "${title}:${sid}")',
            '    else',
            '      sessions+=("${sid}")',
            '    fi',
            '  done < <(claude-exec ext resume-completions "$PWD" 2>/dev/null)',
            "  _describe 'session' sessions",
            '}',
            '',
        ]
    )

    # Main completion function
    lines.append('_claude_exec() {')
    lines.append('  _arguments -s -S \\')

    model_aliases = ' '.join(mod.MODEL_ALIASES)

    for flag in mod.ROOT_FLAGS:
        if not flag.documented:
            continue

        arg_spec = _zsh_arg_spec(
            arg_type=flag.arg_type,
            description=flag.description,
            choices=flag.choices,
            model_aliases=model_aliases,
        )

        if flag.short:
            line = f"    '({flag.short} {flag.name})'{{{flag.short},{flag.name}}}'{arg_spec}'"
        else:
            line = f"    '{flag.name}{arg_spec}'"

        lines.append(line + ' \\')

    lines.append("    '1: :_claude_exec_commands' \\")
    lines.append("    '*::arg:->rest'")
    lines.append('')

    lines.extend(
        [
            '  case "$state" in',
            '    rest)',
            '      case "${words[1]}" in',
            '        ext)',
            "          _arguments '*:ext command:(install uninstall)'",
            '          ;;',
            '      esac',
            '      ;;',
            '  esac',
            '}',
            '',
        ]
    )

    lines.append('_claude_exec_commands() {')
    lines.append('  local -a subcommands')
    lines.append('  subcommands=(')
    for sub in mod.SUBCOMMANDS:
        desc = sub.description.replace("'", '')
        lines.append(f"    '{sub.name}:{desc}'")
    lines.append("    'ext:Claude exec management commands'")
    lines.append('  )')
    lines.append("  _describe 'command' subcommands")
    lines.append('}')
    lines.append('')
    lines.append('_claude_exec "$@"')

    return '\n'.join(lines) + '\n'


def _zsh_arg_spec(
    *,
    arg_type: str,
    description: str,
    choices: Sequence[str],
    model_aliases: str,
) -> str:
    """Convert FlagDef fields to zsh ``_arguments`` action syntax."""
    desc = description.replace("'", '').replace('"', '').replace('[', '').replace(']', '')

    if arg_type == 'none':
        return f'[{desc}]'
    if arg_type == 'choice':
        return f'[{desc}]:value:({" ".join(choices)})'
    if arg_type == 'path':
        return f'[{desc}]:file:_files'
    if arg_type == 'model':
        return f'[{desc}]:model:({model_aliases})'
    if arg_type == 'session':
        return f'[{desc}]:session:_claude_exec_sessions'
    return f'[{desc}]:value:'


# -- Session index for resume completions --------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SessionEntry:
    """A resumable session with optional custom title."""

    session_id: str
    title: str
    mtime: float


class CacheEntry(TypedDict):
    """Shape of a single cached session entry on disk."""

    title: str
    mtime: float
    size: int


class SessionIndex:
    """Scans project session files for resume completions with cached title extraction.

    Uses ``mmap`` + ``rfind`` to search backwards for ``custom-title`` records
    without parsing entire JSONL files. Results are cached to disk with
    incremental updates based on file mtime + size.
    """

    TITLE_NEEDLE = b'"custom-title"'
    WORKTREE_NEEDLE = b'"worktree-state"'
    CACHE_FRESHNESS_SECONDS = 10.0

    def __init__(self, project_path: str) -> None:
        encoded = encode_project_path(project_path)
        self._project_dir = get_claude_config_home_dir() / 'projects' / encoded
        self._cache_path = get_claude_workspace_config_home_dir() / 'claude-exec' / f'{encoded}.json'

    def completions(self) -> Sequence[SessionEntry]:
        """Return resumable sessions sorted by recency, using cache when fresh."""
        if not self._project_dir.is_dir():
            return []

        if self._cache_is_fresh():
            return self._from_cache()

        return self._scan_and_cache()

    def _cache_is_fresh(self) -> bool:
        try:
            return (time.time() - self._cache_path.stat().st_mtime) < self.CACHE_FRESHNESS_SECONDS
        except OSError:
            return False

    def _from_cache(self) -> Sequence[SessionEntry]:
        try:
            cache: Mapping[str, CacheEntry] = json.loads(self._cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return self._scan_and_cache()
        return self._to_entries(cache)

    def _scan_and_cache(self) -> Sequence[SessionEntry]:
        cache: dict[str, CacheEntry] = dict(self._load_cache())  # mutable copy for incremental updates

        current: dict[str, tuple[Path, float, int]] = {}
        for entry in os.scandir(self._project_dir):
            if not entry.name.endswith('.jsonl') or entry.name.startswith('agent-'):
                continue
            st = entry.stat()
            current[entry.name[:-6]] = (Path(entry.path), st.st_mtime, st.st_size)

        updated = False
        for sid, (path, file_mtime, file_size) in current.items():
            cached = cache.get(sid)
            if cached is not None and cached.get('mtime', 0) >= file_mtime and cached.get('size', 0) == file_size:
                continue
            cache[sid] = {
                'title': self._extract_title(path) or '',
                'mtime': file_mtime,
                'size': file_size,
            }
            updated = True

        for sid in list(cache):
            if sid not in current:
                del cache[sid]
                updated = True

        if updated:
            self._save_cache(cache)

        return self._to_entries(cache)

    def _load_cache(self) -> Mapping[str, CacheEntry]:
        try:
            return json.loads(self._cache_path.read_text())  # type: ignore[no-any-return]  # json.loads returns Any
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self, cache: Mapping[str, CacheEntry]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache, separators=(',', ':')))
        tmp.replace(self._cache_path)

    @classmethod
    def _extract_title(cls, path: Path) -> str | None:
        """Extract the last custom-title from a JSONL file using mmap rfind."""
        if path.stat().st_size == 0:
            return None
        with open(path, 'rb') as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = mm.rfind(cls.TITLE_NEEDLE)
            if pos == -1:
                return None
            line_start = mm.rfind(b'\n', 0, pos) + 1
            line_end = mm.find(b'\n', pos)
            if line_end == -1:
                line_end = len(mm)
            rec: dict[str, str] = json.loads(mm[line_start:line_end])
            return rec.get('customTitle') or None

    @classmethod
    def _extract_worktree_path(cls, path: Path) -> str | None:
        """Extract the active worktree path from the last worktree-state record.

        Returns None if the session never entered a worktree or exited it
        (``worktreeSession: null`` is the ExitWorktree signal). Sessions may
        enter and exit worktrees multiple times — rfind gets the final state.
        """
        if path.stat().st_size == 0:
            return None
        with open(path, 'rb') as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = mm.rfind(cls.WORKTREE_NEEDLE)
            if pos == -1:
                return None
            line_start = mm.rfind(b'\n', 0, pos) + 1
            line_end = mm.find(b'\n', pos)
            if line_end == -1:
                line_end = len(mm)
            rec: dict[str, object] = json.loads(mm[line_start:line_end])
            session = rec.get('worktreeSession')
            if not isinstance(session, dict):
                return None  # null = exited worktree via ExitWorktree
            return str(session.get('worktreePath', '')) or None

    @staticmethod
    def _to_entries(cache: Mapping[str, CacheEntry]) -> Sequence[SessionEntry]:
        entries = [
            SessionEntry(
                session_id=sid,
                title=info.get('title', ''),
                mtime=info.get('mtime', 0.0),
            )
            for sid, info in cache.items()
        ]
        entries.sort(key=lambda e: e.mtime, reverse=True)
        return entries


def _resolve_worktree_venv(args: Sequence[str]) -> Path | None:
    """Detect an active worktree from ``--resume`` session and return its venv path.

    Scans the session JSONL for a ``worktree-state`` record. If the session
    is currently in a worktree (not exited), validates the worktree directory
    and its ``.venv`` exist, and returns the venv path for activation.

    Returns None for fresh sessions, bare ``--resume`` (picker mode), exited
    worktrees, or missing/unpopulated worktree venvs.
    """
    # Extract the --resume/-r value (already resolved from title→UUID by _resolve_resume_arg)
    resume_id: str | None = None
    for flag in ('--resume', '-r'):
        if flag in args:
            idx = list(args).index(flag)
            if idx + 1 < len(args):
                resume_id = args[idx + 1]
            break

    if not resume_id:
        return None

    # Construct JSONL path from CWD-based project dir encoding
    encoded = encode_project_path(os.getcwd())
    jsonl_path = get_claude_config_home_dir() / 'projects' / encoded / f'{resume_id}.jsonl'
    if not jsonl_path.exists():
        return None

    worktree_path = SessionIndex._extract_worktree_path(jsonl_path)
    if not worktree_path:
        return None

    # Validate the worktree is still active
    wt = Path(worktree_path)
    git_file = wt / '.git'
    if not git_file.is_file():
        return None

    venv = wt / '.venv'
    if not (venv / 'bin').is_dir():
        return None

    os.environ['CLAUDE_EXEC_WORKTREE'] = worktree_path
    print(f'claude-exec: worktree venv {venv} (from session {resume_id})', file=sys.stderr)
    return venv


def _resolve_binary() -> str:
    """Find the real claude binary via ``which``. Fail fast if not found or self."""
    binary = shutil.which('claude')
    if binary is None:
        raise BinaryNotFoundError('claude not found on PATH')

    if os.path.realpath(binary) == os.path.realpath(__file__):
        raise BinaryNotFoundError('`which claude` resolved to this script — check PATH')

    return binary


# -- Exceptions + error boundary handlers -------------------------------------


class LaunchError(Exception):
    """Base exception for launcher errors."""


class BinaryNotFoundError(LaunchError):
    """Claude binary not found or resolved to self."""


@boundary.handler(BinaryNotFoundError)
def _handle_binary_not_found(exc: BinaryNotFoundError) -> None:
    print(f'claude-exec: {exc}', file=sys.stderr)


@boundary.handler(LaunchError)
def _handle_launch_error(exc: LaunchError) -> None:
    print(f'claude-exec: {exc}', file=sys.stderr)


@boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'claude-exec: {type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
