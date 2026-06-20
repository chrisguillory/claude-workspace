#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
#     "typer>=0.9.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

"""Make a project skill available at user scope by symlinking it into ~/.claude/skills.

The repo stays the single source of truth: ``hoist`` creates a symlink
``~/.claude/skills/<name>`` -> ``<project>/<skill-dir>``, so Claude Code discovers
the skill globally while every edit lands in the versioned project copy. A
companion script resolves its relative uv deps through the symlink via the
per-skill ``realpath "${CLAUDE_SKILL_DIR}/<script>"`` convention in the skill's
SKILL.md — this CLI never touches a SKILL.md, only the symlink.

We treat ``~/.claude/skills`` as a home for *symlinks* to versioned skills rather
than unversioned hand-copies — this repo's ideal, not a Claude Code requirement (CC
loads a real directory there just fine). So ``hoist`` manages symlinks freely (the
repo is the backup) but won't create or replace a real directory: it fails loudly on
one in its way and leaves the cleanup to you. No copies, no backups.

    claude-hoist-skill hoist skills/recover-session   # symlink into user scope
    claude-hoist-skill unhoist recover-session        # remove the symlink (refuses real dirs)
    claude-hoist-skill list                           # audit: symlinks vs un-hoisted real dirs
    claude-hoist-skill install                        # put claude-hoist-skill on PATH
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

import typer
from cc_lib.cli import add_help_command, add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.utils import get_claude_config_home_dir


class HoistError(Exception):
    """An expected, actionable hoist failure — formatted at the boundary, not as a traceback."""


# Filesystem-safe skill name: leading alnum, then alnum / dot / dash / underscore.
# Defined locally rather than reaching into cc_lib.cli's private LauncherInstaller
# (not exported) — a one-line regex isn't worth the cross-module private coupling.
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

app = create_app(help='Hoist a project skill to user scope via a symlink.')
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


@error_boundary.handler(HoistError)
def _on_hoist_error(exc: HoistError) -> None:
    """Print the actionable message (no traceback); the boundary then exits 1."""
    print(f'error: {exc}', file=sys.stderr)


@app.command()
@error_boundary
def hoist(
    skill_dir: str = typer.Argument(..., help='Project skill directory to hoist (must contain SKILL.md)'),
    name: str | None = typer.Option(None, '--name', help='User-scope name (default: the directory basename)'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Print the action and touch nothing'),
) -> None:
    """Symlink a project skill into ~/.claude/skills/<name>; the repo stays the source of truth.

    Repoints an existing hoist symlink freely (the repo is the backup). Fails loudly on a
    real directory in the way — an unversioned copy is never the ideal state, so you remove
    it and re-hoist rather than the CLI silently relocating or deleting your data.
    """
    src = Path(skill_dir).resolve()
    if not src.is_dir():
        raise HoistError(f'not a directory: {src}')
    if not (src / 'SKILL.md').is_file():
        raise HoistError(f'{src} is not a skill: no SKILL.md')
    _check_hoistable(src)

    skill_name = name or src.name
    if skill_name in ('.', '..') or not SAFE_NAME.match(skill_name):
        raise HoistError(f'unsafe skill name: {skill_name!r}')

    dst = get_claude_config_home_dir() / 'skills' / skill_name

    # Classify the target. Check is_symlink() BEFORE is_dir() — a symlink-to-a-dir
    # satisfies both, and a broken symlink is is_symlink() True / is_dir() False.
    if dst.is_symlink():
        current = _symlink_target(dst)
        if current == src:
            print(f'already hoisted: {dst} -> {src}')
            return
        # A symlink points at versioned content (the repo is the backup), so repoint freely.
        _apply(dry_run, f'repointed {dst}: {current} -> {src}', lambda: _atomic_symlink(src, dst))
        return

    if dst.is_dir():  # a real directory (symlinks handled + returned above)
        raise HoistError(f'{dst} is a real directory — refusing to replace it.')

    if dst.exists():  # a regular file — an unexpected occupant
        raise HoistError(f'{dst} exists and is not a skill symlink; refusing')

    _apply(dry_run, f'hoisted {dst} -> {src}', lambda: _new_symlink(src, dst))


@app.command()
@error_boundary
def unhoist(
    name: str = typer.Argument(..., help='Skill name (or path) to remove from user scope'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Print the action and touch nothing'),
) -> None:
    """Remove a hoisted skill's user-scope symlink. Refuses real directories (never a hand copy)."""
    dst = get_claude_config_home_dir() / 'skills' / Path(name).name
    if not dst.is_symlink():
        if dst.exists():
            raise HoistError(f'{dst} is a real directory — refusing to remove it.')
        print(f'not hoisted: {dst}')
        return
    _apply(dry_run, f'unhoisted {dst}', dst.unlink)


@app.command('list')
@error_boundary
def list_skills() -> None:
    """List user-scope skills: hoisted symlinks (-> target) vs un-hoisted real dirs."""
    skills_dir = get_claude_config_home_dir() / 'skills'
    if not skills_dir.is_dir():
        print(f'no user-scope skills directory: {skills_dir}')
        return
    for entry in sorted(skills_dir.iterdir()):
        if entry.name.startswith('.'):
            continue
        if entry.is_symlink():
            target = _symlink_target(entry)
            flag = '' if target.exists() else '  [BROKEN]'
            print(f'{entry.name}  -> {target}{flag}')
        elif entry.is_dir():
            print(f'{entry.name}  (real dir — not hoisted)')


# -- helpers ------------------------------------------------------------------


def _check_hoistable(src: Path) -> None:
    """Refuse a skill that would break once symlinked.

    A companion script with relative ``[tool.uv.sources]`` resolves those paths from its
    *invoked* location, so once the skill is a symlink under ~/.claude/skills the deps break
    — unless the SKILL.md invokes the script through ``realpath "${CLAUDE_SKILL_DIR}/..."``.
    Deterministic, best-effort: catches scripts referenced by a SKILL.md ``!``-line (the
    common case); scripts invoked elsewhere are outside a SKILL.md lint's reach.
    """
    skill_md = (src / 'SKILL.md').read_text()
    for script in sorted(src.glob('*.py')):
        header = script.read_text()[:2000]  # the PEP 723 block sits at the top
        if '[tool.uv.sources]' not in header or '"../' not in header:
            continue  # no relative workspace deps -> nothing to break
        ref = re.search(rf'!`[^`]*\b{re.escape(script.name)}\b[^`]*`', skill_md)
        if ref is None:
            continue  # not invoked via a !-line -> outside this check's reach
        line = ref.group()
        if 'realpath' not in line or 'CLAUDE_SKILL_DIR' not in line:
            raise HoistError(
                f'{src.name}/SKILL.md invokes {script.name} (which has relative uv deps) without '
                f'`realpath "${{CLAUDE_SKILL_DIR}}/..."` — it would break once hoisted.'
            )


def _symlink_target(link: Path) -> Path:
    """Resolve a symlink's stored target to an absolute path (handles relative links)."""
    raw = Path(os.readlink(link))
    return raw if raw.is_absolute() else (link.parent / raw).resolve()


def _new_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def _atomic_symlink(src: Path, dst: Path) -> None:
    """Repoint dst -> src atomically (temp link + rename) so discovery never sees a gap."""
    tmp = dst.with_name(f'.{dst.name}.tmp.{os.getpid()}')
    tmp.unlink(missing_ok=True)
    os.symlink(src, tmp)
    os.replace(tmp, dst)


def _apply(dry_run: bool, summary: str, action: Callable[[], None]) -> None:
    """Run ``action`` (unless dry-run) and print ``summary``."""
    if dry_run:
        print(f'[dry-run] {summary}')
        return
    action()
    print(summary)


add_install_command(app, script_path=__file__)


if __name__ == '__main__':
    run_app(app)
