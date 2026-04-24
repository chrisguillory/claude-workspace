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

Provides zsh tab completion for all Claude Code flags (52 documented + 40
hidden, from v2.1.114 binary analysis), subcommands with descriptions,
model aliases, and project-scoped session completion with custom titles
for ``--resume``.

Also fixes environment issues that the bare ``claude`` binary doesn't handle:
    - Activates ``$PWD/.venv/`` and cleans stale/IDE-injected PATH entries
    - Detects worktree state on resume (repairs trailing-NULL worktreeSession
      records so Claude Code re-chdirs into the worktree)
    - Injects ``--effort`` from settings.json env block
    - Injects ``--thinking-display summarized`` (Opus 4.7 default is
      ``omitted``; makes thinking-block content visible in JSONL)
    - Injects ``--append-system-prompt`` with visible-reasoning bundle
      (gated on ``CLAUDE_EXEC_VISIBLE_REASONING=1``)
    - Injects ``--allow-dangerously-skip-permissions`` so bypass mode is
      Shift+Tab-reachable without starting in it
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
import json
import mmap
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

import pydantic
import rich.console
import typer
from cc_lib import claude_cli_introspection
from cc_lib.cli import LauncherInstaller, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas import CamelModel
from cc_lib.schemas.base import ClosedModel
from cc_lib.types import EffortLevel
from cc_lib.utils import encode_project_path, get_claude_config_home_dir

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

    raw_args = sys.argv[1:]
    if _is_subcommand_invocation(raw_args):
        # Subcommands (update, mcp, config, etc.) have their own flag
        # schemas — don't inject interactive-session flags.
        args: Sequence[str] = raw_args
    else:
        args = _inject_effort_flag(raw_args)
        args = _inject_allow_dangerous_flag(args)
        args = _inject_thinking_display_flag(args)
        args = _inject_visible_reasoning_prompt(args)
    args = _resolve_resume_arg(args)

    # Detect worktree once; both env vars and venv decision derive from it.
    # Must run after _resolve_resume_arg so title→UUID is resolved.
    worktree_path = WorktreeResolver(args).resolve()
    launch_dir = worktree_path or Path.cwd()
    os.environ['CLAUDE_EXEC_LAUNCH_DIR'] = str(launch_dir)
    if worktree_path:
        os.environ['CLAUDE_EXEC_WORKTREE'] = str(worktree_path)

    # Worktree venv takes priority over CWD venv.
    venv_bin = Path.cwd() / '.venv' / 'bin'
    worktree_venv = WorktreeResolver.venv_of(worktree_path)
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


# -- Subcommand detection ------------------------------------------------------


def _is_subcommand_invocation(args: Sequence[str]) -> bool:
    """True if args contain a known claude subcommand, skipping leading global flags.

    Commander.js (Claude Code's CLI framework) accepts global flags before the
    subcommand (`claude --debug mcp list`), so detection must skip flag tokens
    (and their values, for known value-taking flags) to find the first positional.
    """
    candidate = _first_positional(args)
    if candidate is None:
        return False
    return candidate in {sc.name for sc in claude_cli_introspection.SUBCOMMANDS}


def _first_positional(args: Sequence[str]) -> str | None:
    """Return the first positional arg, skipping leading flags and their values."""
    takes_value: dict[str, bool] = {}
    for flag in claude_cli_introspection.ROOT_FLAGS:
        takes_value[flag.name] = flag.arg_type != 'none'
        if flag.short:
            takes_value[flag.short] = flag.arg_type != 'none'

    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith('-'):
            return tok
        if '=' in tok:  # --flag=value is a single token
            i += 1
            continue
        i += 2 if takes_value.get(tok, False) else 1
    return None


# -- Venv activation -----------------------------------------------------------


def _activate_venv(venv_path: Path) -> None:
    """Set env vars, clean stale paths, and prepend .venv/bin to PATH.

    Removes stale ``.venv/bin`` entries from other projects (e.g., PyCharm's
    JediTerm auto-activates the main project's venv, which is wrong in a
    worktree). IDE application paths themselves (``/Applications/PyCharm.app/
    Contents/MacOS`` etc.) are left alone — they're legitimate user PATH
    entries exporting binaries the user expects to invoke (``pycharm``,
    Cursor CLI tooling).
    """
    venv_bin = str(venv_path / 'bin')
    os.environ['VIRTUAL_ENV'] = str(venv_path)
    os.environ['CLAUDE_EXEC_VENV'] = str(venv_path)
    os.environ['PATH'] = venv_bin + ':' + _clean_path(venv_bin)


def _clean_path(keep: str) -> str:
    """Remove stale .venv/bin entries from PATH."""
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
    """True if this PATH entry should be removed during venv activation.

    Only strips ``.venv/bin`` — typically an IDE-injected path from a
    different project (e.g. PyCharm's JediTerm auto-activates the main
    repo's venv, which is wrong when Claude Code runs in a worktree).
    User-exported IDE application paths (``/PyCharm.app/``, ``/.cursor/``)
    are intentionally preserved.
    """
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


# -- Allow-dangerous flag injection --------------------------------------------


def _inject_allow_dangerous_flag(args: Sequence[str]) -> Sequence[str]:
    """Add ``--allow-dangerously-skip-permissions`` so bypass mode is Shift+Tab-reachable.

    This flag (distinct from ``--dangerously-skip-permissions``) adds
    ``bypassPermissions`` to the Shift+Tab cycle WITHOUT starting in it.
    Default behavior: launch in ``default`` mode, toggle into bypass only
    when needed, without having to restart the session. Stable since v2.1.86.

    Skipped if the user explicitly set ``--dangerously-skip-permissions``
    (coupled "start in bypass"), ``--allow-dangerously-skip-permissions``
    (avoid duplicate), or ``--permission-mode bypassPermissions`` (same
    effect). Respects ``--permission-mode <other>`` — bug #17544 confirms
    ``--allow-`` variant composes cleanly with an explicit initial mode.

    Requires ``skipDangerousModePermissionPrompt: true`` in settings.json
    to avoid the one-time warning dialog when Shift+Tab-ing into bypass.

    Refs:
        https://code.claude.com/docs/en/permission-modes
        https://github.com/anthropics/claude-code/issues/28697 (shipped feature)
        https://github.com/anthropics/claude-code/issues/17544 (--permission-mode composes)
        https://github.com/anthropics/claude-code/issues/21062 (still open: unlock by default)
    """
    explicit_bypass_flags = {
        '--allow-dangerously-skip-permissions',
        '--dangerously-skip-permissions',
    }
    if any(a in explicit_bypass_flags for a in args):
        return args

    # Also skip if user passed --permission-mode bypassPermissions explicitly
    for i, a in enumerate(args):
        if a == '--permission-mode' and i + 1 < len(args) and args[i + 1] == 'bypassPermissions':
            return args

    return [*args, '--allow-dangerously-skip-permissions']


# -- Thinking-display flag injection -------------------------------------------


def _inject_thinking_display_flag(args: Sequence[str]) -> Sequence[str]:
    """Pass ``--thinking-display summarized`` so Opus 4.7 thinking stays visible.

    Opus 4.7 (released 2026-04-16) changed the server-side default for the API's
    ``thinking.display`` parameter from ``"summarized"`` to ``"omitted"``.
    Claude Code v2.1.112 does not send the ``display`` field unless this hidden
    CLI flag is supplied, so on 4.7 the assistant's ``thinking`` text is
    silently empty in the session JSONL. Forcing ``summarized`` restores the
    pre-4.7 behavior.

    The binary exposes no settings.json key and no env var for this knob —
    CLI-flag injection is the only persistent workaround (verified against
    v2.1.112's commander registration and request-builder). Opus 4.7 still
    writes the encrypted ``signature`` field even when ``thinking`` is empty
    (for multi-turn continuity), but the user sees nothing downstream.

    The companion ``showThinkingSummaries: true`` settings.json key gates a
    different mechanism (the ``redact-thinking-2026-02-12`` beta header).
    Both levers must be favorable to see thinking content.

    Removable: delete this function and the call in ``main()`` when Anthropic
    either flips the server default back to ``"summarized"`` on Opus 4.7+, or
    exposes ``thinkingDisplay`` as a settings.json / env-var knob.

    Related bugs:
        https://github.com/anthropics/claude-code/issues/48065
        https://github.com/anthropics/claude-code/issues/49268
        https://github.com/anthropics/claude-code/issues/49708
    """
    if any(a == '--thinking-display' or a.startswith('--thinking-display=') for a in args):
        return args
    return [*args, '--thinking-display', 'summarized']


# -- Visible-reasoning system-prompt injection --------------------------------


VISIBLE_REASONING_PROMPT = (
    'Before any non-trivial response, structure your output as two '
    'markdown sections: first "## Reasoning" containing your step-by-step '
    'thinking, then "## Answer" containing your final response. For '
    'trivial single-step tasks (direct lookups, single-file reads, '
    'one-line edits), skip both headings and answer directly.\n\n'
    'Inside the Reasoning section: state any assumptions, note '
    'alternatives considered if relevant, and call out low-confidence '
    'claims. Before moving to the Answer section, self-verify your plan '
    "against the user's stated criteria.\n\n"
    'When making claims about code: never speculate about a file you '
    "haven't read — open it first. Ground any claim about code by "
    'quoting the relevant lines and citing file path plus line numbers.'
)


def _inject_visible_reasoning_prompt(args: Sequence[str]) -> Sequence[str]:
    """Append a system-prompt instruction for visible reasoning + verification.

    Bundles Anthropic-endorsed visibility instructions (manual CoT
    scaffolding, self-verification before finishing, anti-hallucination
    for code) per the 2026 prompt-engineering best-practices page.

    Rationale: Opus 4.7 changed the ``thinking.display`` default to
    ``"omitted"``, making the model's internal reasoning channel opaque
    to external users. This injection routes reasoning into the visible
    response text instead — same-model output, fully persisted to the
    JSONL, greppable and diff-able across sessions. Strictly more useful
    than the thinking-block channel for engineers auditing what the model
    is actually doing. (Anthropic's own Claude Code system prompt uses
    the thinking-block channel — that choice is for end-user coding UX,
    not for debugging the model.)

    Gated on ``CLAUDE_EXEC_VISIBLE_REASONING=1`` in the process
    environment. Set it inline (``CLAUDE_EXEC_VISIBLE_REASONING=1
    claude-exec``) or via ``export`` in the shell.

    Per user preference, uses ``## Reasoning`` / ``## Answer`` markdown
    headings (vs the Anthropic-canonical ``<thinking>`` / ``<answer>``
    XML tags) for dictation-playback ergonomics.

    Fails fast on collision with any user-passed ``--system-prompt``-family
    flag. Silent-skip would hide the conflict; silent-append would create
    ambiguous prompt composition. Error message includes an escape hatch:
    ``env -u CLAUDE_EXEC_VISIBLE_REASONING claude-exec ...``.

    Removable: delete this function, the constant above, and the call in
    ``main()`` when Anthropic restores raw thinking visibility by default
    on Opus 4.7+, or when a cleaner mechanism lands.

    Refs:
        https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
            Anthropic's 2026 best-practices — names this pattern
            "Manual CoT as a fallback".
        Wei et al. 2022, Chain-of-Thought Prompting:
            https://arxiv.org/abs/2201.11903
        Chen, Benton et al. (Anthropic) 2025, faithfulness caveat:
            https://arxiv.org/abs/2505.05410
    """
    if os.environ.get('CLAUDE_EXEC_VISIBLE_REASONING') != '1':
        return args

    collision_flags = (
        '--system-prompt',
        '--system-prompt-file',
        '--append-system-prompt',
        '--append-system-prompt-file',
    )
    for flag in collision_flags:
        for a in args:
            if a == flag or a.startswith(f'{flag}='):
                raise LaunchError(
                    f'CLAUDE_EXEC_VISIBLE_REASONING=1 conflicts with '
                    f'{flag} on the command line. The injection and your '
                    f'explicit flag cannot coexist meaningfully — pass '
                    f'one or the other, not both. To skip injection for '
                    f'this invocation only: '
                    f'`env -u CLAUDE_EXEC_VISIBLE_REASONING claude-exec ...`.'
                )

    return [*args, '--append-system-prompt', VISIBLE_REASONING_PROMPT]


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
    lines: list[str] = [
        '#compdef claude-exec',
        '',
        '# Auto-generated from cc_lib.claude_cli_introspection (FlagDef data)',
        f'# Claude Code v2.1.114, {__import__("datetime").date.today().isoformat()}',
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

    model_aliases = ' '.join(claude_cli_introspection.MODEL_ALIASES)

    for flag in claude_cli_introspection.ROOT_FLAGS:
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
    for sub in claude_cli_introspection.SUBCOMMANDS:
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


class CachedSessionEntry(ClosedModel):
    """On-disk cache shape for a single session index entry."""

    title: str
    mtime: float
    size: int


CACHE_ADAPTER: pydantic.TypeAdapter[Mapping[str, CachedSessionEntry]] = pydantic.TypeAdapter(
    Mapping[str, CachedSessionEntry]
)


class SessionIndex:
    """Scans project session files for resume completions with cached title extraction.

    Uses ``mmap`` + ``rfind`` to search backwards for ``custom-title`` records
    without parsing entire JSONL files. Results are cached to disk with
    incremental updates based on file mtime + size.
    """

    TITLE_NEEDLE = b'"custom-title"'
    CACHE_FRESHNESS_SECONDS = 10.0

    def __init__(self, project_path: str) -> None:
        encoded = encode_project_path(project_path)
        self._project_dir = get_claude_config_home_dir() / 'projects' / encoded
        self._cache_path = Path.home() / '.claude-workspace' / 'claude-exec' / f'{encoded}.json'

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
        cache = self._load_cache()
        if cache is None:
            return self._scan_and_cache()
        return self._to_entries(cache)

    def _scan_and_cache(self) -> Sequence[SessionEntry]:
        cache: dict[str, CachedSessionEntry] = dict(self._load_cache() or {})

        current: dict[str, tuple[Path, float, int]] = {}
        for entry in os.scandir(self._project_dir):
            if not entry.name.endswith('.jsonl') or entry.name.startswith('agent-'):
                continue
            st = entry.stat()
            current[entry.name[:-6]] = (Path(entry.path), st.st_mtime, st.st_size)

        updated = False
        for sid, (path, file_mtime, file_size) in current.items():
            cached = cache.get(sid)
            if cached is not None and cached.mtime >= file_mtime and cached.size == file_size:
                continue
            cache[sid] = CachedSessionEntry(
                title=self._extract_title(path) or '',
                mtime=file_mtime,
                size=file_size,
            )
            updated = True

        for sid in list(cache):
            if sid not in current:
                del cache[sid]
                updated = True

        if updated:
            self._save_cache(cache)

        return self._to_entries(cache)

    def _load_cache(self) -> Mapping[str, CachedSessionEntry] | None:
        try:
            raw = self._cache_path.read_text()
        except OSError:
            return None
        try:
            return CACHE_ADAPTER.validate_json(raw)
        except pydantic.ValidationError:
            return None

    def _save_cache(self, cache: Mapping[str, CachedSessionEntry]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix('.tmp')
        tmp.write_bytes(CACHE_ADAPTER.dump_json(cache))
        tmp.replace(self._cache_path)

    @classmethod
    def _extract_title(cls, path: Path) -> str | None:
        """Extract the last custom-title from a JSONL file using mmap rfind."""
        try:
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
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _to_entries(cache: Mapping[str, CachedSessionEntry]) -> Sequence[SessionEntry]:
        entries = [SessionEntry(session_id=sid, title=info.title, mtime=info.mtime) for sid, info in cache.items()]
        entries.sort(key=lambda e: e.mtime, reverse=True)
        return entries


# -- Worktree resolution on resume --------------------------------------------


class WorktreeSession(CamelModel):
    """Claude Code's ``worktreeSession`` payload from session JSONL.

    Claude Code protocol data — we declare only the field ``claude-exec``
    reads. ``CamelModel``'s ``alias_generator=to_camel`` maps Python
    ``worktree_path`` to JSON ``worktreePath``. The inherited ``extra='allow'``
    default preserves the rest (``originalCwd``, ``worktreeName``,
    ``worktreeBranch``, ``originalBranch``, ``originalHeadCommit``,
    ``sessionId``, ``enteredExisting``, and anything Claude Code adds in
    future versions) so the repair step round-trips byte-for-byte.
    """

    worktree_path: str


@dataclasses.dataclass(frozen=True, slots=True)
class WorktreeScan:
    """Result of scanning a session JSONL for the current worktree state.

    ``worktree_session``
        Typed view of the field claude-exec reads (``worktreePath``), or
        ``None`` if no populated record was found. ``SubsetModel``/``OpenModel``
        drops/preserves unknown fields on dump — for round-trip fidelity use
        ``raw_record``.
    ``raw_record``
        The full original ``worktreeSession`` dict from Claude Code's
        JSONL (with ``originalCwd``, ``originalBranch``, ``worktreeName``,
        etc.), preserved opaquely so the repair-append round-trips the
        full payload. ``None`` iff ``worktree_session`` is ``None``.
    ``saw_null_after``
        True if ``worktreeSession: null`` cleanup records appeared AFTER
        the populated record. Triggers a repair-append so Claude Code's
        own ``v3_()`` chdir on resume sees a populated final record.
    ``explicitly_exited``
        True if an ``ExitWorktree`` tool_use record appeared AFTER the
        populated record. Definitive signal the user intentionally left
        the worktree — overrides ``saw_null_after``, do NOT repair.
    """

    worktree_session: WorktreeSession | None
    raw_record: Mapping[str, object] | None
    saw_null_after: bool
    explicitly_exited: bool


class WorktreeResolver:
    """Detect, report, and repair the worktree state of a resuming session.

    Background — Claude Code writes ``worktreeSession: null`` cleanup
    records into session JSONL at several points:

    - When the user invokes the ``ExitWorktree`` tool
    - Periodic ``reAppendSessionMetadata()`` every ~32 KB of writes
    - Graceful shutdown re-appends session metadata
    - Session resume re-appends metadata onto the fresh JSONL

    By the time ``claude-exec`` reads the JSONL on resume, the LAST
    ``worktree-state`` record is almost always NULL — even when the user
    was genuinely in a worktree. Claude Code's own ``v3_()`` reads that
    final record and decides whether to chdir. Seeing NULL, it stays in
    the main tree, and the user's cwd silently drifts from the venv
    ``claude-exec`` is about to activate.

    The fix is three layers:

    1. Scan backwards through ``worktree-state`` records for the last
       populated one whose ``worktreePath`` directory still has a ``.git``
       file. The ``.git`` check distinguishes "Claude Code wrote a cleanup
       NULL" from "user removed the worktree directory" — the latter
       correctly returns None and we fall back to the main tree.
    2. Look for an ``ExitWorktree`` tool_use record AFTER that populated
       record. If present, the user explicitly exited — return None,
       respect intent, do NOT re-enter even if NULLs followed.
    3. Otherwise, if NULL cleanups appeared after the populated record,
       append a fresh copy of that record. Claude Code's ``v3_()`` reads
       the final record on resume and calls ``process.chdir(worktreePath)``
       — which is what we want.

    One-shot: construct once per ``claude-exec`` invocation, call
    ``resolve()``, discard. No persistent state.
    """

    WORKTREE_NEEDLE = b'"worktree-state"'
    EXIT_WORKTREE_NEEDLE = b'"name":"ExitWorktree"'

    def __init__(self, args: Sequence[str]) -> None:
        self._args = args

    def resolve(self) -> Path | None:
        """Return the worktree to enter on resume, or None for the main tree.

        Returns None for fresh sessions, bare ``--resume`` (picker mode),
        sessions with no stored worktree-state, stale records (the
        worktree directory has been removed), and explicit ``ExitWorktree``
        calls.
        """
        resume_id = self._resume_id()
        if resume_id is None:
            return None

        jsonl_path = self._jsonl_path(resume_id)
        if jsonl_path is None:
            return None

        scan = self.scan(jsonl_path)
        if scan.worktree_session is None or scan.explicitly_exited:
            return None

        if scan.saw_null_after and scan.raw_record is not None:
            self._repair(jsonl_path, scan.raw_record, resume_id)

        return Path(scan.worktree_session.worktree_path)

    @staticmethod
    def venv_of(worktree_path: Path | None) -> Path | None:
        """Return the worktree's populated ``.venv`` directory, or None."""
        if worktree_path is None:
            return None
        venv = worktree_path / '.venv'
        if not (venv / 'bin').is_dir():
            return None
        return venv

    @classmethod
    def scan(cls, path: Path) -> WorktreeScan:
        """Scan a session JSONL for worktree state. See ``WorktreeScan``."""
        empty = WorktreeScan(
            worktree_session=None,
            raw_record=None,
            saw_null_after=False,
            explicitly_exited=False,
        )
        if path.stat().st_size == 0:
            return empty

        saw_null_after = False
        with open(path, 'rb') as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = len(mm)
            while True:
                pos = mm.rfind(cls.WORKTREE_NEEDLE, 0, end)
                if pos == -1:
                    return WorktreeScan(
                        worktree_session=None,
                        raw_record=None,
                        saw_null_after=saw_null_after,
                        explicitly_exited=False,
                    )
                line_start = mm.rfind(b'\n', 0, pos) + 1
                line_end = mm.find(b'\n', pos)
                if line_end == -1:
                    line_end = len(mm)
                rec: dict[str, object] = json.loads(mm[line_start:line_end])
                if rec.get('type') != 'worktree-state':
                    # Incidental match — the needle appeared inside a user
                    # message, tool result, or other record body. Skip.
                    end = line_start
                    continue
                session_dict = rec.get('worktreeSession')
                if isinstance(session_dict, dict):
                    try:
                        parsed = WorktreeSession.model_validate(session_dict)
                    except pydantic.ValidationError:
                        end = line_start
                        continue
                    if (Path(parsed.worktree_path) / '.git').is_file():
                        explicitly_exited = mm.find(cls.EXIT_WORKTREE_NEEDLE, line_end) != -1
                        return WorktreeScan(
                            worktree_session=parsed,
                            raw_record=session_dict,
                            saw_null_after=saw_null_after,
                            explicitly_exited=explicitly_exited,
                        )
                    # Stale record (worktree dir gone). Keep scanning —
                    # maybe an earlier worktree still exists on disk.
                else:
                    # NULL cleanup record written by Claude Code.
                    saw_null_after = True
                end = line_start

    def _resume_id(self) -> str | None:
        """Parse ``--resume <id>`` / ``-r <id>`` from args, or None."""
        for flag in ('--resume', '-r'):
            if flag in self._args:
                idx = list(self._args).index(flag)
                if idx + 1 < len(self._args):
                    return self._args[idx + 1]
                return None
        return None

    @staticmethod
    def _jsonl_path(resume_id: str) -> Path | None:
        """Return the session JSONL path for the current project, or None."""
        encoded = encode_project_path(os.getcwd())
        path = get_claude_config_home_dir() / 'projects' / encoded / f'{resume_id}.jsonl'
        return path if path.exists() else None

    @staticmethod
    def _repair(jsonl_path: Path, raw_record: Mapping[str, object], resume_id: str) -> None:
        """Append a fresh populated worktree-state record to the JSONL.

        Round-trips the original payload byte-for-byte so Claude Code's
        ``v3_()`` on resume reads it and calls ``process.chdir(worktreePath)``.
        """
        record = {
            'type': 'worktree-state',
            'worktreeSession': raw_record,
            'sessionId': resume_id,
        }
        with jsonl_path.open('ab') as fh:
            fh.write(json.dumps(record, separators=(',', ':')).encode() + b'\n')
        worktree_path = raw_record.get('worktreePath') if isinstance(raw_record, dict) else None
        print(
            f'claude-exec: repaired trailing NULL worktree-state → {worktree_path}',
            file=sys.stderr,
        )


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
