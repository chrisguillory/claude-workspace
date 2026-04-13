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
"""Transparent proxy for the claude binary with project venv activation.

Activates ``$PWD/.venv/`` before exec'ing the real ``claude`` binary so that
``process.env.PATH`` has ``.venv/bin`` from birth — flowing to hooks, MCP
servers, statusline, and Bash tool commands equally.

All arguments pass through to the real binary unchanged. The wrapper process
is replaced entirely via ``os.execv`` — no parent zombie, no signal
forwarding, the terminal talks directly to Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pydantic
import rich.console
from cc_lib.cli import LauncherInstaller, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

boundary = ErrorBoundary(exit_code=1)


@boundary
def main() -> None:
    """Entry point: intercept completions, ``ext``, or activate venv + exec claude."""
    if len(sys.argv) > 1 and sys.argv[1] == 'ext':
        sys.argv = [sys.argv[0], *sys.argv[2:]]  # strip 'ext' so typer sees subcommands
        run_app(ext_app)
        return

    venv_bin = Path.cwd() / '.venv' / 'bin'
    if venv_bin.is_dir():
        _activate_venv(venv_bin.parent)

    args = _inject_effort_flag(sys.argv[1:])

    binary = _resolve_binary()
    os.execv(binary, [binary, *args])


def _activate_venv(venv_path: Path) -> None:
    """Set env vars, clean stale paths, and prepend .venv/bin to PATH.

    Removes stale ``.venv/bin`` entries from other projects (e.g., PyCharm's
    JediTerm auto-activates the main project's venv, which is wrong in a
    worktree) and IDE-specific paths that shouldn't leak into Claude's
    subprocess environment.
    """
    venv_bin = str(venv_path / 'bin')
    os.environ['VIRTUAL_ENV'] = str(venv_path)
    os.environ['CLAUDE_LAUNCH_VENV'] = str(venv_path)
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
        print(f'claude-launch: cleaned {len(removed)} PATH entries:', file=sys.stderr)
        for r in removed:
            print(f'  - {r}', file=sys.stderr)
    return ':'.join(cleaned)


def _is_stale_path(entry: str) -> bool:
    """True if this PATH entry should be removed during venv activation."""
    if '/.venv/bin' in entry or '/venv/bin' in entry:
        return True
    if '/PyCharm.app/' in entry or '/WebStorm.app/' in entry:
        return True
    if '/.cursor/' in entry:
        return True
    return False


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

    effort = os.environ.get('CLAUDE_CODE_EFFORT_LEVEL') or _read_settings_effort()
    if not effort:
        return args

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
        raise LaunchError(
            f'CLAUDE_CODE_EFFORT_LEVEL={value!r} in {settings_path}: {e.errors()[0]["msg"]}'
        ) from None


def _generate_zsh_completion() -> str:
    """Generate a static zsh completion script from FlagDef data.

    Uses ``_arguments`` for rich flag completions (descriptions, file paths,
    choice lists) and a dynamic ``_claude_launch_sessions`` function for
    ``--resume`` session ID completion from ``~/.claude/projects/``.

    No Python callback on TAB — everything runs in native zsh.
    """
    completions_path = Path(__file__).parent / 'claude-launch-completions.py'
    spec = importlib.util.spec_from_file_location('claude_launch_completions', completions_path)
    if spec is None or spec.loader is None:
        raise LaunchError(f'failed to load completions from {completions_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod.__name__] = mod
    spec.loader.exec_module(mod)

    lines: list[str] = [
        '#compdef claude-launch',
        '',
        '# Auto-generated from claude-launch-completions.py (FlagDef data)',
        f'# Claude Code v2.1.92, {__import__("datetime").date.today().isoformat()}',
        '',
    ]

    # Dynamic session ID completion function
    lines.extend(
        [
            '# Dynamic session ID completion from ~/.claude/projects/',
            '_claude_launch_sessions() {',
            '  local -a sessions',
            '  local project_dir',
            '  project_dir="$HOME/.claude/projects"',
            '  [[ -d "$project_dir" ]] || return',
            '  for f in "$project_dir"/**/*.jsonl(N.om); do',
            '    sessions+=("${f:t:r}")',
            '  done',
            '  compadd -a sessions',
            '}',
            '',
        ]
    )

    # Main completion function
    lines.append('_claude_launch() {')
    lines.append('  _arguments -s -S \\')

    model_aliases = ' '.join(mod.MODEL_ALIASES)

    for flag in mod.ROOT_FLAGS:
        if not flag.documented:
            continue  # hide undocumented flags from completion

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

    # Positional: first arg is subcommand or prompt
    lines.append("    '1: :_claude_launch_commands' \\")
    lines.append("    '*::arg:->rest'")
    lines.append('')

    # Subcommand dispatch
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

    # Subcommand list function
    lines.append('_claude_launch_commands() {')
    lines.append('  local -a subcommands')
    lines.append('  subcommands=(')
    for sub in mod.SUBCOMMANDS:
        desc = sub.description.replace("'", '')
        lines.append(f"    '{sub.name}:{desc}'")
    lines.append("    'ext:Claude launcher management commands'")
    lines.append('  )')
    lines.append("  _describe 'command' subcommands")
    lines.append('}')
    lines.append('')
    lines.append('_claude_launch "$@"')

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
        return f'[{desc}]:session:_claude_launch_sessions'
    return f'[{desc}]:value:'


def _resolve_binary() -> str:
    """Find the real claude binary via ``which``. Fail fast if not found or self."""
    binary = shutil.which('claude')
    if binary is None:
        raise BinaryNotFoundError('claude not found on PATH')

    if os.path.realpath(binary) == os.path.realpath(__file__):
        raise BinaryNotFoundError('`which claude` resolved to this script — check PATH')

    return binary


# -- ext subcommand app --------------------------------------------------------

ext_app = create_app(help='Claude launcher management commands.')

COMPLETION_DIR = Path.home() / '.config' / 'zsh' / 'completions'


@ext_app.command()
def install() -> None:
    """Install launcher and zsh completions to PATH."""
    console = rich.console.Console(stderr=True)

    # Launcher script
    launcher = LauncherInstaller(Path(__file__).resolve())
    path = launcher.install('claude-launch')
    console.print(f'Launcher: [bold]{path}[/bold]')
    console.print(f'  -> {Path(__file__).resolve()}')

    # Static zsh completion (not typer's eval-based protocol)
    COMPLETION_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMPLETION_DIR / '_claude-launch'
    dest.write_text(_generate_zsh_completion())
    console.print(f'Completion: [bold]{dest}[/bold]')

    console.print('\nRestart shell to activate.')


@ext_app.command()
def uninstall() -> None:
    """Remove launcher and zsh completions."""
    console = rich.console.Console(stderr=True)

    launcher_path = Path.home() / '.local' / 'bin' / 'claude-launch'
    if launcher_path.exists():
        launcher_path.unlink()
        console.print(f'Removed: {launcher_path}')

    completion_path = COMPLETION_DIR / '_claude-launch'
    if completion_path.exists():
        completion_path.unlink()
        console.print(f'Removed: {completion_path}')


# -- Exceptions + error boundary handlers -------------------------------------


class LaunchError(Exception):
    """Base exception for launcher errors."""


class BinaryNotFoundError(LaunchError):
    """Claude binary not found or resolved to self."""


@boundary.handler(BinaryNotFoundError)
def _handle_binary_not_found(exc: BinaryNotFoundError) -> None:
    print(f'claude-launch: {exc}', file=sys.stderr)


@boundary.handler(LaunchError)
def _handle_launch_error(exc: LaunchError) -> None:
    print(f'claude-launch: {exc}', file=sys.stderr)


@boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'claude-launch: {type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
