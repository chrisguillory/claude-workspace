"""
Command-line interface for claude-session.

Provides commands to archive and restore Claude Code sessions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TypeGuard

import httpx
import typer
from cc_lib.cli import add_completion_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.session_tracker import load_sessions
from cc_lib.utils import encode_project_path, get_claude_config_home_dir

from claude_session.exceptions import ClaudeSessionError, SourceProjectConflictError
from claude_session.launcher import launch_claude_with_session
from claude_session.schemas.operations.lineage import LineageTree
from claude_session.services.archive import SessionArchiveService
from claude_session.services.artifacts.custom_title import extract_custom_title_from_file
from claude_session.services.claude_process import auto_detect_session_id
from claude_session.services.clone import SessionCloneService
from claude_session.services.delete import SessionDeleteService
from claude_session.services.discovery import SessionDiscoveryService
from claude_session.services.info import SessionInfoService
from claude_session.services.lineage import LineageService
from claude_session.services.move import SessionMoveService
from claude_session.services.parser import SessionParserService
from claude_session.services.restore import SessionRestoreService
from claude_session.storage.gist import GistStorage
from claude_session.storage.local import LocalFileSystemStorage

logger = logging.getLogger(__name__)

app = create_app(help='Archive and restore Claude Code sessions.')
add_completion_command(app)
error_boundary = ErrorBoundary(exit_code=1)


# -- Typer infrastructure (private, must precede commands that reference them) --


@app.callback(invoke_without_command=True)
def _configure_logging(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option('--verbose', '-v', help='Show detailed progress')] = False,
    claude_dir: Annotated[
        Path | None,
        typer.Option(
            '--claude-dir',
            help='Claude Code config directory (default: ~/.claude)',
            envvar='CLAUDE_CONFIG_DIR',
        ),
    ] = None,
) -> None:
    """Configure logging and show help when no command given."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stderr, force=True)
    if claude_dir is not None:
        os.environ['CLAUDE_CONFIG_DIR'] = str(claude_dir)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


ArchiveFormat = Literal['json', 'zst']


def _is_archive_format(value: str) -> TypeGuard[ArchiveFormat]:
    """Type guard for valid archive formats."""
    return value in ('json', 'zst')


def _validate_archive_format(value: str | None) -> ArchiveFormat | None:
    """Validate and narrow archive format for typer callback."""
    if value is None:
        return None
    if _is_archive_format(value):
        return value
    raise typer.BadParameter("Must be 'json' or 'zst'")


def _format_age(dt: datetime | None) -> str:
    """Format a datetime as a precise human-readable age.

    Uses two-unit precision (e.g., 2h12m, 3d5h) so that descriptions
    are naturally unique — zsh groups completions with identical text.
    """
    if dt is None:
        return '?'
    total_minutes = int((datetime.now(UTC) - dt).total_seconds() / 60)
    if total_minutes < 60:
        return f'{max(total_minutes, 1)}m'
    hours, mins = divmod(total_minutes, 60)
    if hours < 24:
        return f'{hours}h{mins}m'
    days, hrs = divmod(hours, 24)
    return f'{days}d{hrs}h'


def _complete_session_id(incomplete: str) -> Sequence[tuple[str, str]]:
    """Complete session IDs from tracked sessions. Active first, then newest."""
    db = load_sessions('.')
    matches = [s for s in db.sessions if s.session_id.startswith(incomplete)]
    matches.sort(
        key=lambda s: (
            s.state != 'active',  # active first
            -(s.metadata.process_created_at or datetime.min.replace(tzinfo=UTC)).timestamp(),
        ),
    )
    results: list[tuple[str, str]] = []
    for s in matches:
        title = extract_custom_title_from_file(Path(s.transcript_path))
        parts = [s.state, Path(s.project_dir).name, _format_age(s.metadata.process_created_at)]
        if title:
            parts.append(title)
        results.append((s.session_id, ', '.join(parts)))
    # zsh groups completions with identical descriptions onto one line.
    # Deduplicate by appending the 8-char session prefix to collisions.
    seen: dict[str, int] = {}
    for _sid, desc in results:
        seen.setdefault(desc, 0)
        seen[desc] += 1
    duped = {desc for desc, count in seen.items() if count > 1}
    if duped:
        results = [(sid, f'{desc} ({sid[:8]})') if desc in duped else (sid, desc) for sid, desc in results]
    return results


# -- Commands (public @app.command functions) --


@app.command()
@error_boundary
def archive(
    session_id: Annotated[
        str | None,
        typer.Argument(
            help='Session ID to archive (auto-detected inside Claude Code)',
            autocompletion=_complete_session_id,
        ),
    ] = None,
    output: Annotated[str | None, typer.Argument(help='Output path or gist:// (default: gist://)')] = None,
    format: Annotated[
        ArchiveFormat | None,
        typer.Option(
            '--format',
            '-f',
            help='Archive format: json or zst',
            callback=_validate_archive_format,
        ),
    ] = None,
    gist_token: Annotated[
        str | None,
        typer.Option(
            '--gist-token',
            help='GitHub token (or use GITHUB_TOKEN env or gh CLI)',
        ),
    ] = None,
    gist_visibility: Annotated[
        Literal['public', 'secret'],
        typer.Option(
            '--gist-visibility',
            help='Gist visibility (public or secret)',
        ),
    ] = 'secret',
    gist_description: Annotated[
        str, typer.Option('--gist-description', help='Gist description')
    ] = 'Claude Code Session Archive',
    source_project: Annotated[
        Path | None, typer.Option('--source-project', '--sp', help='Scope to sessions in this project directory')
    ] = None,
) -> None:
    """Archive a Claude Code session to local file or GitHub Gist.

    When run inside Claude Code, session ID is auto-detected if not provided.
    When output is omitted, defaults to uploading to a new GitHub Gist.

    \b
    Examples:
        claude-session archive                    # auto-detect, new gist
        claude-session archive gist://            # same as above
        claude-session archive gist://abc123      # update existing gist
        claude-session archive 019c406c           # specific session
        claude-session archive 019c406c out.json  # local file
    """
    # Disambiguate positional args: if session_id looks like an output target, shift it
    if session_id is not None and output is None and not _is_session_id(session_id):
        output = session_id
        session_id = None

    if session_id is None and source_project is not None:
        raise SourceProjectConflictError

    # Resolve session ID (auto-detect if needed)
    resolved_session_id = _resolve_session_id(session_id)

    # Default output to gist://
    if output is None:
        output = 'gist://'

    asyncio.run(
        _archive_async(
            resolved_session_id, output, format, gist_token, gist_visibility, gist_description, source_project
        )
    )


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
@error_boundary
def restore(
    ctx: typer.Context,
    archive: Annotated[str, typer.Argument(help='Archive path or Gist URL (gist://<gist-id> or file path)')],
    target_project: Annotated[
        Path | None, typer.Option('--target-project', '--tp', help='Target project directory (default: current)')
    ] = None,
    no_translate: Annotated[bool, typer.Option('--no-translate', help="Don't translate file paths")] = False,
    in_place: Annotated[
        bool, typer.Option('--in-place', help='Restore with original session ID (verbatim restore)')
    ] = False,
    launch: Annotated[bool, typer.Option('--launch', '-l', help='Launch Claude Code after restore')] = False,
    gist_token: Annotated[str | None, typer.Option('--gist-token', help='GitHub token for private gists')] = None,
) -> None:
    """Restore a Claude Code session from local file or GitHub Gist.

    By default, restore creates a new session ID (UUIDv7) for the restored session.
    Use --in-place to restore with the original session ID for verbatim restoration.

    \b
    Extra arguments after -- are passed to claude CLI:
        claude-session restore ARCHIVE --launch -- --chrome
    """
    asyncio.run(_restore_async(archive, target_project, not no_translate, in_place, launch, gist_token, ctx.args))


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
@error_boundary
def clone(
    ctx: typer.Context,
    session_id: Annotated[
        str | None,
        typer.Argument(
            help='Session ID to clone (auto-detected inside Claude Code)',
            autocompletion=_complete_session_id,
        ),
    ] = None,
    source_project: Annotated[
        Path | None, typer.Option('--source-project', '--sp', help='Scope to sessions in this project directory')
    ] = None,
    target_project: Annotated[
        Path | None, typer.Option('--target-project', '--tp', help='Target project directory (default: current)')
    ] = None,
    no_translate: Annotated[bool, typer.Option('--no-translate', help="Don't translate file paths")] = False,
    launch: Annotated[bool, typer.Option('--launch', '-l', help='Launch Claude Code after clone')] = False,
) -> None:
    """Clone a session directly (no archive file needed).

    When run inside Claude Code, session ID is auto-detected if not provided.

    \b
    Examples:
        claude-session clone                  # clone current session
        claude-session clone SESSION_ID       # clone specific session
        claude-session clone SESSION_ID -l    # clone and launch
    """
    if session_id is None and source_project is not None:
        raise SourceProjectConflictError

    detected = auto_detect_session_id()
    if launch and detected is not None:
        typer.secho('Error: --launch cannot be used inside Claude Code.', fg=typer.colors.RED, err=True)
        typer.echo('Use claude --resume <session-id> after cloning instead.', err=True)
        raise SystemExit(1)

    # Use cached detection result to avoid a second process tree walk
    if session_id is None:
        session_id = detected

    asyncio.run(
        _clone_async(
            _resolve_session_id(session_id), source_project, target_project, not no_translate, launch, ctx.args
        )
    )


@app.command()
@error_boundary
def delete(
    session_id: Annotated[
        str | None,
        typer.Argument(
            help='Session ID (full or prefix). Auto-detected inside Claude Code.',
            autocompletion=_complete_session_id,
        ),
    ] = None,
    force: Annotated[bool, typer.Option('--force', '-f', help='Required to delete native (UUIDv4) sessions')] = False,
    terminate: Annotated[
        bool, typer.Option('--terminate', '-t', help='Terminate running Claude process before deletion')
    ] = False,
    no_backup: Annotated[bool, typer.Option('--no-backup', help="Don't keep a backup file for undo")] = False,
    dry_run: Annotated[bool, typer.Option('--dry-run', help='Preview what would be deleted')] = False,
    source_project: Annotated[
        Path | None, typer.Option('--source-project', '--sp', help='Scope to sessions in this project directory')
    ] = None,
) -> None:
    """Delete session artifacts with auto-backup.

    By default, only cloned/restored sessions (UUIDv7) can be deleted.
    Native Claude sessions (UUIDv4) require --force.

    If the session is currently running, use --terminate to kill the Claude
    process before deletion. This prevents the session file from being
    recreated when the process exits.

    A backup is saved to ~/.claude-workspace/claude-session/deleted/ for undo capability.
    Use 'restore --in-place' on the backup to undo.

    \b
    Examples:
        claude-session delete                           # delete current session
        claude-session delete SESSION_ID                # delete specific session
        claude-session delete SESSION_ID --force        # delete native session
        claude-session delete SESSION_ID --terminate    # kill running session first
        claude-session delete SESSION_ID --dry-run      # preview without deleting
    """
    if session_id is None and source_project is not None:
        raise SourceProjectConflictError
    asyncio.run(_delete_async(_resolve_session_id(session_id), force, terminate, no_backup, dry_run, source_project))


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
@error_boundary
def move(
    ctx: typer.Context,
    session_id: Annotated[
        str | None,
        typer.Argument(
            help='Session ID (full or prefix). Auto-detected inside Claude Code.',
            autocompletion=_complete_session_id,
        ),
    ] = None,
    source_project: Annotated[
        Path | None, typer.Option('--source-project', '--sp', help='Scope to sessions in this project directory')
    ] = None,
    target_project: Annotated[
        Path | None, typer.Option('--target-project', '--tp', help='Target project directory (default: current)')
    ] = None,
    force: Annotated[bool, typer.Option('--force', '-f', help='Required to move native (UUIDv4) sessions')] = False,
    terminate: Annotated[
        bool, typer.Option('--terminate', '-t', help='Terminate running Claude process before move')
    ] = False,
    no_backup: Annotated[bool, typer.Option('--no-backup', help="Don't keep a backup file for undo")] = False,
    dry_run: Annotated[bool, typer.Option('--dry-run', help='Preview what would be moved')] = False,
    launch: Annotated[
        bool, typer.Option('--launch', '-l', help='Launch Claude Code in target project after move')
    ] = False,
) -> None:
    """Move a session from one project to another.

    Relocates project-specific artifacts (JSONL, tool results, session memory)
    to the target project. Path references are translated. Session ID is preserved.

    By default, only cloned/restored sessions (UUIDv7) can be moved.
    Native Claude sessions (UUIDv4) require --force.

    If the session is currently running, use --terminate to kill the Claude
    process before moving.

    \b
    Extra arguments after -- are passed to claude CLI:
        claude-session move SESSION_ID --launch -- --chrome

    \b
    Examples:
        claude-session move 019b5232               # current project
        claude-session move 019b5232 --target-project ~/proj-b
        claude-session move 019b5232 --dry-run     # preview
        claude-session move 019b5232 --launch      # move and resume
        claude-session move a1b2c3d4 --force       # native session
    """
    if session_id is None and source_project is not None:
        raise SourceProjectConflictError

    detected = auto_detect_session_id()
    if launch and detected is not None:
        typer.secho('Error: --launch cannot be used inside Claude Code.', fg=typer.colors.RED, err=True)
        typer.echo('Use claude --resume <session-id> after moving instead.', err=True)
        raise SystemExit(1)

    if session_id is None:
        session_id = detected

    asyncio.run(
        _move_async(
            _resolve_session_id(session_id),
            source_project,
            target_project,
            force,
            terminate,
            no_backup,
            dry_run,
            launch,
            ctx.args,
        )
    )


@app.command()
@error_boundary
def lineage(
    session_id: Annotated[
        str | None,
        typer.Argument(
            help='Session ID (full or prefix). Auto-detected inside Claude Code.',
            autocompletion=_complete_session_id,
        ),
    ] = None,
    format: Annotated[
        Literal['text', 'tree', 'json'], typer.Option('--format', '-f', help='Output format: text, tree, or json')
    ] = 'text',
    source_project: Annotated[
        Path | None, typer.Option('--source-project', '--sp', help='Scope to sessions in this project directory')
    ] = None,
) -> None:
    """Show the lineage (parent-child relationships) for a session.

    When run inside Claude Code without a session ID, auto-detects the current session.

    \b
    Examples:
        claude-session lineage
        claude-session lineage 019b53ff
        claude-session lineage c3bac5a6 --format tree
    """
    if session_id is None and source_project is not None:
        raise SourceProjectConflictError
    asyncio.run(_lineage_async(_resolve_session_id(session_id), format, source_project))


@app.command()
@error_boundary
def info(
    session_id: Annotated[
        str | None,
        typer.Argument(
            help='Session ID (full or prefix). Auto-detected inside Claude Code.',
            autocompletion=_complete_session_id,
        ),
    ] = None,
    format: Annotated[
        Literal['text', 'json'], typer.Option('--format', '-f', help='Output format: text or json')
    ] = 'text',
    source_project: Annotated[
        Path | None, typer.Option('--source-project', '--sp', help='Scope to sessions in this project directory')
    ] = None,
) -> None:
    """Display comprehensive information about a session.

    Shows session context including ID, project path, file locations,
    origin (how it was created), state, and characteristics.

    When run inside Claude Code without a session ID, auto-detects the current session.

    \b
    Examples:
        claude-session info
        claude-session info 019b53ff
        claude-session info c3bac5a6 --format json
    """
    if session_id is None and source_project is not None:
        raise SourceProjectConflictError
    asyncio.run(_info_async(_resolve_session_id(session_id), format, source_project))


def main() -> None:
    """Entry point for CLI."""
    run_app(app)


# -- Private helpers --


def _render_lineage_tree(tree: LineageTree) -> None:
    """Render a LineageTree with proper box-drawing characters."""

    def render_node(node_id: str, prefix: str, is_last: bool, is_root: bool) -> None:
        if is_root:
            connector = ''
            child_prefix = ''
        else:
            connector = '└─ ' if is_last else '├─ '
            child_prefix = prefix + ('   ' if is_last else '│  ')

        node = tree.nodes[node_id]
        title_suffix = f' ({node.custom_title})' if node.custom_title else ''
        line = f'{prefix}{connector}{node_id}{title_suffix}'
        if node_id == tree.queried_session_id:
            typer.secho(line, fg=typer.colors.GREEN)
        else:
            typer.echo(line)

        children = tree.nodes[node_id].children
        for i, child_id in enumerate(children):
            render_node(child_id, child_prefix, is_last=i == len(children) - 1, is_root=False)

    render_node(tree.root_session_id, '', is_last=True, is_root=True)


def _resolve_session_id(session_id: str | None) -> str:
    """Resolve session_id, auto-detecting from Claude Code if not provided."""
    if session_id is not None:
        return session_id
    detected = auto_detect_session_id()
    if detected is None:
        typer.secho('Error: Not running inside Claude Code.', fg=typer.colors.RED, err=True)
        typer.echo('Provide a session ID: claude-session <command> <session-id>', err=True)
        raise SystemExit(1)
    return detected


def _is_session_id(value: str) -> bool:
    """Check if a string looks like a session ID (8+ hex digits/hyphens)."""
    return len(value) >= 8 and bool(re.fullmatch(r'[0-9a-f][0-9a-f-]*', value))


def _encode_project_filter(project: Path | None) -> Path | None:
    """Encode a project directory path for session discovery filtering."""
    if project is None:
        return None
    return get_claude_config_home_dir() / 'projects' / encode_project_path(project.resolve())


def _get_github_token_cli(gist_token: str | None) -> str | None:
    """Resolve GitHub token from flag, environment, or gh CLI.

    Checks in order:
    1. --gist-token flag value
    2. GITHUB_TOKEN environment variable
    3. `gh auth token` command (GitHub CLI)
    """
    if gist_token:
        return gist_token
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        return token
    try:
        result = subprocess.run(
            ['gh', 'auth', 'token'],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass  # gh CLI not installed
    return None


@error_boundary.handler(ClaudeSessionError)
@error_boundary.handler(FileNotFoundError)
@error_boundary.handler(FileExistsError)
def _handle_user_error(exc: Exception) -> None:
    typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)


# -- Private async implementations --


async def _archive_async(
    session_id: str,
    output: str,
    format: Literal['json', 'zst'] | None,
    gist_token: str | None,
    gist_visibility: Literal['public', 'secret'],
    gist_description: str,
    source_project: Path | None,
) -> None:
    """Async implementation of archive command."""

    # Find the session
    discovery = SessionDiscoveryService()
    project_filter = _encode_project_filter(source_project)
    session_info = await discovery.find_session_by_id(session_id, project_filter=project_filter)

    if not session_info:
        if source_project:
            typer.secho(
                f'Error: Session not found: {session_id} in project {source_project}', fg=typer.colors.RED, err=True
            )
        else:
            typer.secho(f'Error: Session not found: {session_id}', fg=typer.colors.RED, err=True)
        typer.echo(f'Searched in: {project_filter or discovery.claude_sessions_dir}', err=True)
        raise SystemExit(1)

    logger.info('Found session in folder: %s', session_info.session_folder)

    # Parse output parameter - check if it's a Gist URL
    use_gist = output.startswith('gist://')
    gist_id = None

    if use_gist:
        # Extract gist ID if provided: gist://abc123 or just gist://
        gist_id = output[7:] if len(output) > 7 else None

        # Get token from CLI flag, environment, or gh CLI
        token = _get_github_token_cli(gist_token)
        if not token:
            typer.secho('Error: GitHub token required for Gist storage.', fg=typer.colors.RED, err=True)
            typer.echo('Provide via one of:')
            typer.echo('  1. gh auth login (GitHub CLI - recommended)')
            typer.echo('  2. Set GITHUB_TOKEN environment variable')
            typer.echo('  3. Pass --gist-token flag')
            typer.echo()
            typer.echo('To create a token manually:')
            typer.echo('  1. Go to https://github.com/settings/tokens')
            typer.echo('  2. Generate new token (classic)')
            typer.echo("  3. Select 'gist' scope")
            typer.echo('  4. Copy the token')
            raise SystemExit(1)

    # Use the resolved full session ID (supports prefix matching)
    resolved_session_id = session_info.session_id

    # Initialize services
    with tempfile.TemporaryDirectory(prefix='claude-session-') as temp_dir:
        parser_service = SessionParserService()
        archive_service = SessionArchiveService(
            session_id=resolved_session_id,
            temp_dir=Path(temp_dir),
            parser_service=parser_service,
            session_folder=session_info.session_folder,  # Use folder directly from discovery
        )

        # Create storage backend
        storage: GistStorage | LocalFileSystemStorage
        if use_gist:
            # Generate filename for Gist with correct extension
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            ext = '.json.zst' if format == 'zst' else '.json'
            filename = f'session-{resolved_session_id[:8]}-{timestamp}{ext}'

            if token is None:
                raise typer.BadParameter('GitHub token required for Gist storage')
            storage = GistStorage(
                token=token,
                gist_id=gist_id,
                visibility=gist_visibility,
                description=gist_description,
            )
            output_path = None  # Don't pass to archive service
            logger.info('Creating Gist archive: %s', filename)
        else:
            # Local filesystem
            output_file = Path(output)
            storage = LocalFileSystemStorage(output_file.parent.resolve())
            output_path = str(output)
            filename = output_file.name
            logger.info('Creating archive: %s', output)

        # Create archive
        metadata = await archive_service.create_archive(storage=storage, output_path=output_path, format_param=format)

        # Print success
        if use_gist and isinstance(storage, GistStorage):
            typer.secho('✓ Archive uploaded to GitHub Gist!', fg=typer.colors.GREEN)
            typer.echo(f'  URL: {metadata.file_path}')
            typer.echo(f'  Gist ID: {storage.gist_id}')
            typer.echo(f'  Format: {metadata.format}')
            typer.echo(f'  Size: {metadata.size_mb} MB')
            typer.echo(f'  Records: {metadata.session_records:,} session, {metadata.agent_records:,} agent')
            typer.echo()
            typer.echo('To restore, use:')
            typer.secho(f'  claude-session restore gist://{storage.gist_id}', fg=typer.colors.CYAN)
        else:
            typer.secho('✓ Archive created successfully!', fg=typer.colors.GREEN)
            typer.echo(f'  Path: {metadata.file_path}')
            typer.echo(f'  Format: {metadata.format}')
            typer.echo(f'  Size: {metadata.size_mb} MB')
            typer.echo(f'  Records: {metadata.session_records:,} session, {metadata.agent_records:,} agent')
            typer.echo(f'  Files: {metadata.file_count}')
            for file_meta in metadata.files:
                logger.info('    - %s: %d records', file_meta.filename, file_meta.record_count)


async def _restore_async(
    archive: str,
    target_project: Path | None,
    translate_paths: bool,
    in_place: bool,
    launch: bool,
    gist_token: str | None,
    extra_args: Sequence[str],
) -> None:
    """Async implementation of restore command."""

    # Check if it's a Gist URL
    if archive.startswith('gist://'):
        gist_id = archive[7:]

        if not gist_id:
            typer.secho('Error: Gist ID required in format gist://<gist-id>', fg=typer.colors.RED, err=True)
            raise SystemExit(1)

        # Get token (public gists don't need auth for reading, but use it if provided)
        token = _get_github_token_cli(gist_token) or ''

        logger.info('Downloading from Gist: %s', gist_id)

        # Create Gist storage
        storage = GistStorage(token=token, gist_id=gist_id)

        # Find the archive file in the gist (look for .json or .json.zst)
        async with httpx.AsyncClient() as client:
            headers = {'Accept': 'application/vnd.github.v3+json', 'X-GitHub-Api-Version': '2022-11-28'}
            if token:
                headers['Authorization'] = f'Bearer {token}'

            response = await client.get(f'https://api.github.com/gists/{gist_id}', headers=headers)

            if response.status_code == 404:
                typer.secho(f'Error: Gist not found: {gist_id}', fg=typer.colors.RED, err=True)
                typer.echo('Check the gist ID and ensure it exists.')
                raise SystemExit(1)

            response.raise_for_status()
            gist_data = response.json()

            # Find archive file (including base64-encoded variants)
            files = gist_data['files']
            archive_file = None
            for filename in files:
                # Strip .b64 suffix for format detection
                base_name = filename[:-4] if filename.endswith('.b64') else filename
                if base_name.endswith(('.json', '.json.zst')):
                    archive_file = filename
                    break

            if not archive_file:
                typer.secho(f'Error: No archive file found in gist {gist_id}', fg=typer.colors.RED, err=True)
                typer.echo(f'Available files: {", ".join(files.keys())}')
                raise SystemExit(1)

        # Download to temp file
        logger.info('Downloading %s...', archive_file)
        data = await storage.load(archive_file)

        # Use logical filename (without .b64) so restore service detects format correctly
        logical_filename = archive_file[:-4] if archive_file.endswith('.b64') else archive_file
        with tempfile.NamedTemporaryFile(suffix=f'-{logical_filename}', delete=False) as temp_file:
            temp_file.write(data)
            archive_path = Path(temp_file.name)

        logger.info('Downloaded %s bytes', f'{len(data):,}')

    else:
        # Local file
        archive_path = Path(archive)

        # Validate archive exists
        if not archive_path.exists():
            typer.secho(f'Error: Archive not found: {archive_path}', fg=typer.colors.RED, err=True)
            raise SystemExit(1)

    # Determine target project path
    if target_project:
        project_path = target_project.resolve()
        if not project_path.exists():
            typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
            raise SystemExit(1)
    else:
        project_path = Path.cwd()

    logger.info('Restoring to project: %s', project_path)

    # Initialize restore service
    restore_service = SessionRestoreService(project_path)

    # Restore the archive
    logger.info('Loading archive: %s', archive_path)
    if in_place:
        logger.info('In-place mode: restoring with original session ID')
    result = await restore_service.restore_archive(
        archive_path=str(archive_path),
        translate_paths=translate_paths,
        in_place=in_place,
    )

    # Clean up temp file if we downloaded from Gist
    if archive.startswith('gist://'):
        archive_path.unlink()

    # Print success
    typer.secho('✓ Session restored successfully!', fg=typer.colors.GREEN)
    typer.echo(f'  New session ID: {result.new_session_id}')
    typer.echo(f'  Original session ID: {result.original_session_id}')
    typer.echo(f'  Project: {result.project_path}')
    typer.echo(f'  Mode: {"in-place" if result.was_in_place else "fork"}')
    typer.echo()
    typer.echo('  Files restored:')
    typer.echo(f'    - Session: 1 main + {len(result.agent_files)} agents')
    if result.plan_files_restored:
        typer.echo(f'    - Plans: {result.plan_files_restored}')
    if result.tool_results_restored:
        typer.echo(f'    - Tool results: {result.tool_results_restored}')
    if result.todos_restored:
        typer.echo(f'    - Todos: {result.todos_restored}')
    if result.tasks_restored:
        typer.echo(f'    - Tasks: {result.tasks_restored}')
    typer.echo()
    typer.echo(f'  Records: {result.main_records_restored:,} main + {result.agent_records_restored:,} agent')
    typer.echo(f'  Paths translated: {result.paths_translated}')

    if launch:
        typer.echo()
        typer.echo('Launching Claude Code...')
        launch_claude_with_session(result.new_session_id, extra_args=extra_args)
        # Note: launch_claude_with_session uses execvp, so we never reach here
    else:
        typer.echo()
        typer.echo('To continue this session, run:')
        typer.secho(f'  claude --resume {result.new_session_id}', fg=typer.colors.CYAN)


async def _clone_async(
    session_id: str,
    source_project: Path | None,
    target_project: Path | None,
    translate_paths: bool,
    launch: bool,
    extra_args: Sequence[str],
) -> None:
    """Async implementation of clone command."""

    # Determine target project path
    if target_project:
        project_path = target_project.resolve()
        if not project_path.exists():
            typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
            raise SystemExit(1)
    else:
        project_path = Path.cwd()

    logger.info('Cloning to project: %s', project_path)

    # Initialize clone service
    clone_service = SessionCloneService(project_path)

    # Clone the session
    project_filter = _encode_project_filter(source_project)
    result = await clone_service.clone(
        source_session_id=session_id,
        translate_paths=translate_paths,
        project_filter=project_filter,
    )

    # Print success
    typer.secho('✓ Session cloned successfully!', fg=typer.colors.GREEN)
    typer.echo(f'  New session ID: {result.new_session_id}')
    typer.echo(f'  Original session ID: {result.original_session_id}')
    typer.echo(f'  Project: {result.project_path}')
    typer.echo()
    typer.echo('  Files cloned:')
    typer.echo(f'    - Session: 1 main + {len(result.agent_files)} agents')
    if result.plan_files_restored:
        typer.echo(f'    - Plans: {result.plan_files_restored}')
    if result.tool_results_restored:
        typer.echo(f'    - Tool results: {result.tool_results_restored}')
    if result.todos_restored:
        typer.echo(f'    - Todos: {result.todos_restored}')
    typer.echo()
    typer.echo(f'  Records: {result.main_records_restored:,} main + {result.agent_records_restored:,} agent')
    typer.echo(f'  Paths translated: {result.paths_translated}')

    if launch:
        typer.echo()
        typer.echo('Launching Claude Code...')
        launch_claude_with_session(result.new_session_id, extra_args=extra_args)
        # Note: launch_claude_with_session uses execvp, so we never reach here
    else:
        typer.echo()
        typer.echo('To continue this session, run:')
        typer.secho(f'  claude --resume {result.new_session_id}', fg=typer.colors.CYAN)


async def _delete_async(
    session_id: str,
    force: bool,
    terminate: bool,
    no_backup: bool,
    dry_run: bool,
    source_project: Path | None,
) -> None:
    """Async implementation of delete command."""

    # Resolve session ID prefix to full ID
    # When --source-project is provided, use it to disambiguate discovery
    info_service = SessionInfoService()
    project_filter = _encode_project_filter(source_project)
    session_info = await info_service.resolve_session(session_id, project_filter=project_filter)
    full_session_id = session_info.session_id

    # Check if session is running
    is_running, running_pid = info_service.is_session_running(full_session_id)

    if is_running:
        assert running_pid is not None  # is_running=True guarantees this
        if dry_run:
            # Dry-run: show warning but continue
            typer.secho(f'Warning: Session is currently running (PID {running_pid})', fg=typer.colors.YELLOW)
        elif not terminate:
            # Running without --terminate: error
            typer.secho(
                f'Error: Session {full_session_id[:12]}... is currently running (PID {running_pid}).',
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo('Use --terminate to kill the process before deletion.', err=True)
            raise SystemExit(1)
        else:
            # Running with --terminate: will terminate
            logger.info('Session is running (PID %s), will terminate before deletion', running_pid)

    # Initialize delete service
    # When --source-project is explicit, use it (user intent wins over discovery)
    # Otherwise, use discovered session_folder (handles cross-directory correctly)
    if source_project:
        project_path = source_project.resolve()
        if not project_path.exists():
            typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
            raise SystemExit(1)
        delete_service = SessionDeleteService(project_path=project_path)
    else:
        delete_service = SessionDeleteService(session_folder=session_info.session_folder)

    # Pass PID to terminate if running and --terminate was specified
    terminate_pid = running_pid if is_running and terminate and not dry_run else None

    # Delete the session
    result = await delete_service.delete_session(
        session_id=full_session_id,
        force=force,
        no_backup=no_backup,
        dry_run=dry_run,
        terminate_pid_before_delete=terminate_pid,
    )

    if not result.success:
        typer.secho(f'Error: {result.error_message}', fg=typer.colors.RED, err=True)
        raise SystemExit(1)

    if dry_run:
        typer.secho('Dry run - would delete:', fg=typer.colors.YELLOW)
        typer.echo(f'  Session ID: {result.session_id}')
        typer.echo(f'  Files: {result.files_deleted}')
        typer.echo(f'  Directories: {len(result.directories_removed)}')
        typer.echo(f'  Size: {result.size_freed_bytes:,} bytes')
        for path in result.deleted_files:
            logger.info('    - %s', path)
        for path in result.directories_removed:
            logger.info('    dir: %s', path)
    else:
        typer.secho('✓ Session deleted successfully!', fg=typer.colors.GREEN)
        typer.echo(f'  Session ID: {result.session_id}')
        typer.echo()
        typer.echo('  Files deleted:')
        typer.echo(f'    - Session: {result.session_files_deleted}')
        if result.plan_files_deleted:
            typer.echo(f'    - Plans: {result.plan_files_deleted}')
        if result.tool_results_deleted:
            typer.echo(f'    - Tool results: {result.tool_results_deleted}')
        if result.todos_deleted:
            typer.echo(f'    - Todos: {result.todos_deleted}')
        if result.tasks_deleted:
            typer.echo(f'    - Tasks: {result.tasks_deleted}')
        typer.echo()
        typer.echo(f'  Directories removed: {len(result.directories_removed)}')
        typer.echo(f'  Size freed: {result.size_freed_bytes:,} bytes')
        typer.echo(f'  Duration: {result.duration_ms:.0f}ms')
        if result.backup_path:
            typer.echo(f'  Backup: {result.backup_path}')
            typer.echo()
            typer.echo('To undo, run:')
            typer.secho(f'  claude-session restore --in-place {result.backup_path}', fg=typer.colors.CYAN)


async def _move_async(
    session_id: str,
    source_project: Path | None,
    target_project: Path | None,
    force: bool,
    terminate: bool,
    no_backup: bool,
    dry_run: bool,
    launch: bool,
    extra_args: Sequence[str],
) -> None:
    """Async implementation of move command."""

    # Resolve session ID prefix to full ID
    info_service = SessionInfoService()
    project_filter = _encode_project_filter(source_project)
    session_info = await info_service.resolve_session(session_id, project_filter=project_filter)
    full_session_id = session_info.session_id

    # Check if session is running
    is_running, running_pid = info_service.is_session_running(full_session_id)

    if is_running:
        assert running_pid is not None
        if dry_run:
            typer.secho(f'Warning: Session is currently running (PID {running_pid})', fg=typer.colors.YELLOW)
        elif not terminate:
            typer.secho(
                f'Error: Session {full_session_id[:12]}... is currently running (PID {running_pid}).',
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo('Use --terminate to kill the process before moving.', err=True)
            raise SystemExit(1)
        else:
            logger.info('Session is running (PID %s), will terminate before move', running_pid)

    # Determine target project path
    if target_project:
        project_path = target_project.resolve()
        if not project_path.exists():
            typer.secho(f'Error: Project directory does not exist: {project_path}', fg=typer.colors.RED, err=True)
            raise SystemExit(1)
    else:
        project_path = Path.cwd()

    # Pass PID to terminate if running and --terminate was specified
    terminate_pid = running_pid if is_running and terminate and not dry_run else None

    # Move the session
    move_service = SessionMoveService(project_path)
    result = await move_service.move_session(
        session_id=full_session_id,
        force=force,
        no_backup=no_backup,
        dry_run=dry_run,
        terminate_pid=terminate_pid,
    )

    if dry_run:
        typer.secho('Dry run - would move:', fg=typer.colors.YELLOW)
        typer.echo(f'  Session ID: {result.session_id}')
        typer.echo(f'  From: {result.source_project}')
        typer.echo(f'  To: {result.target_project}')
        typer.echo(f'  Files: {result.files_moved}')
        typer.echo(f'  Paths translated: {result.paths_translated}')
    else:
        typer.secho('✓ Session moved successfully!', fg=typer.colors.GREEN)
        typer.echo(f'  Session ID: {result.session_id}')
        typer.echo(f'  From: {result.source_project}')
        typer.echo(f'  To: {result.target_project}')
        typer.echo()
        typer.echo(f'  Files written: {result.files_moved}')
        typer.echo(f'  Files deleted: {result.files_deleted}')
        typer.echo(f'  Paths translated: {result.paths_translated}')
        typer.echo(f'  Duration: {result.duration_ms:.0f}ms')
        if result.backup_path:
            typer.echo(f'  Backup: {result.backup_path}')
            typer.echo()
            typer.echo('To undo, run:')
            typer.secho(f'  claude-session restore --in-place {result.backup_path}', fg=typer.colors.CYAN)

        for warning in result.warnings:
            typer.secho(f'  Warning: {warning}', fg=typer.colors.YELLOW)

        if launch:
            typer.echo()
            typer.echo('Launching Claude Code...')
            launch_claude_with_session(result.session_id, extra_args=extra_args)
            # Note: launch_claude_with_session uses execvp, so we never reach here
        else:
            typer.echo()
            typer.echo('To continue this session, run:')
            typer.secho(f'  claude --resume {result.session_id}', fg=typer.colors.CYAN)


async def _lineage_async(session_id: str, format: Literal['text', 'tree', 'json'], source_project: Path | None) -> None:
    """Async implementation of lineage command."""
    # Resolve session ID via discovery (supports prefix matching + project filter)
    info_service = SessionInfoService()
    project_filter = _encode_project_filter(source_project)
    session_info = await info_service.resolve_session(session_id, project_filter=project_filter)
    full_session_id = session_info.session_id

    lineage_service = LineageService()
    tree = lineage_service.get_full_tree(full_session_id)

    if tree is None:
        typer.secho(f'No lineage found for {full_session_id}', fg=typer.colors.YELLOW)
        typer.echo('(This is either a native session or lineage tracking was not enabled)')
        return

    if format == 'text':
        node = tree.nodes[tree.queried_session_id]
        typer.echo(f'Session: {node.session_id}')
        if node.custom_title:
            typer.echo(f'Title:   {node.custom_title}')
        if node.cloned_at is not None:
            # Child node — show operation metadata
            typer.echo(f'Parent:  {node.parent_id}')
            typer.echo(f'Cloned:  {node.cloned_at}')
            typer.echo(f'Method:  {node.method}')
            typer.echo(f'Source:  {node.parent_project_path}')
            typer.echo(f'Target:  {node.target_project_path}')
            typer.echo(f'Machine: {node.target_machine_id}')
            if node.parent_machine_id:
                if node.is_cross_machine:
                    typer.secho(f'Source Machine: {node.parent_machine_id} (cross-machine)', fg=typer.colors.YELLOW)
                else:
                    typer.echo(f'Source Machine: {node.parent_machine_id} (same machine)')
            if node.archive_path:
                typer.echo(f'Archive: {node.archive_path}')
            if node.paths_translated:
                typer.secho('Paths translated: yes', fg=typer.colors.CYAN)
        else:
            # Root node — no operation metadata
            typer.secho(f'Root session with {len(node.children)} clone(s)', fg=typer.colors.CYAN)
            for child_id in node.children:
                child_node = tree.nodes[child_id]
                title_suffix = f' ({child_node.custom_title})' if child_node.custom_title else ''
                typer.echo(f'  └─ {child_id}{title_suffix}')

    elif format == 'tree':
        _render_lineage_tree(tree)

    elif format == 'json':
        typer.echo(tree.model_dump_json(indent=2))


async def _info_async(session_id: str, format: Literal['text', 'json'], source_project: Path | None) -> None:
    """Async implementation of info command."""
    info_service = SessionInfoService()
    project_filter = _encode_project_filter(source_project)
    context = await info_service.get_info(session_id, project_filter=project_filter)

    if format == 'json':
        typer.echo(context.model_dump_json(indent=2))
        return

    # Text format
    typer.echo(f'Session: {context.session_id}')
    if context.custom_title:
        typer.echo(f'Title: {context.custom_title}')
    typer.echo(f'Project: {context.project_path}')
    typer.echo()

    # Origin section
    typer.secho('Origin:', bold=True)
    source_display: str = context.source
    if context.parent_id:
        source_display += f' (parent: {context.parent_id[:12]}...)'
    typer.echo(f'  Source: {source_display}')
    typer.echo(f'  State: {context.state}')
    typer.echo(f'  Native: {"yes" if context.is_native else "no (cloned/restored)"}')
    if context.has_lineage:
        typer.secho('  Lineage: tracked', fg=typer.colors.CYAN)
    typer.echo()

    # Temporal section - Authoritative timestamps
    typer.secho('Timestamps:', bold=True)
    if context.first_message_at:
        typer.echo(f'  First Message: {context.first_message_at}')
    if context.process_created_at:
        typer.echo(f'  Process Created: {context.process_created_at}')
    if context.session_ended_at:
        reason_suffix = f' ({context.session_end_reason})' if context.session_end_reason else ''
        typer.echo(f'  Session Ended: {context.session_ended_at}{reason_suffix}')
    if context.crash_detected_at:
        typer.secho(f'  Crash Detected: {context.crash_detected_at}', fg=typer.colors.RED)
    if context.cloned_at:
        typer.secho(f'  Cloned: {context.cloned_at}', fg=typer.colors.CYAN)
    typer.echo()

    # Files section
    typer.secho('Files:', bold=True)
    typer.echo(f'  Session: {context.session_file}')
    typer.echo(f'  Debug: {context.debug_file}')

    # Environment section (if available)
    if context.machine_id or context.claude_pid or context.claude_version:
        typer.echo()
        typer.secho('Environment:', bold=True)
        if context.machine_id:
            typer.echo(f'  Machine: {context.machine_id}')
        if context.claude_pid:
            typer.echo(f'  Claude PID: {context.claude_pid}')
        if context.claude_version:
            typer.echo(f'  Claude Version: {context.claude_version}')
        if context.temp_dir:
            typer.echo(f'  Temp dir: {context.temp_dir}')


if __name__ == '__main__':
    main()
