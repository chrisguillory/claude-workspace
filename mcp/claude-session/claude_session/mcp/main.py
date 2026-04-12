"""
Claude Code Session MCP Server.

Provides tools for archiving and managing Claude Code session files.

Setup:
    claude mcp add --scope user claude-session -- uvx --refresh --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/claude-session claude-session-mcp

Example:
    # Save current session to temp file
    save_current_session()

    # Save with compression to specific path
    save_current_session(output_path='/path/to/archive.json.zst', format='zst')
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cc_lib.utils import encode_project_path, get_claude_config_home_dir
from mcp.server.fastmcp import Context, FastMCP

from claude_session.exceptions import RunningSessionDeletionError, RunningSessionMoveError
from claude_session.schemas.operations.archive import ArchiveMetadata
from claude_session.schemas.operations.context import SessionContext
from claude_session.schemas.operations.delete import DeleteResult
from claude_session.schemas.operations.gist import GistArchiveResult
from claude_session.schemas.operations.lineage import LineageTree
from claude_session.schemas.operations.move import MoveResult
from claude_session.schemas.operations.restore import RestoreResult
from claude_session.services.archive import SessionArchiveService
from claude_session.services.claude_process import find_ancestor_claude_pid, resolve_session_id_from_pid
from claude_session.services.clone import SessionCloneService
from claude_session.services.delete import SessionDeleteService
from claude_session.services.info import CurrentSessionContext, SessionInfoService
from claude_session.services.lineage import LineageService
from claude_session.services.move import SessionMoveService
from claude_session.services.parser import SessionParserService
from claude_session.services.restore import SessionRestoreService
from claude_session.storage.gist import GistStorage
from claude_session.storage.local import LocalFileSystemStorage

__all__ = [
    'ClaudeContext',
    'ServerState',
    'lifespan',
    'logger',
    'main',
    'register_tools',
    'server',
]


logger = logging.getLogger(__name__)


# -- Helpers -------------------------------------------------------------------


def _encode_project_filter(source_project: str | None) -> Path | None:
    """Encode a project directory path for session discovery filtering."""
    if source_project is None:
        return None
    resolved = Path(source_project).resolve()
    return get_claude_config_home_dir() / 'projects' / encode_project_path(resolved)


# -- Types ---------------------------------------------------------------------


@dataclass(frozen=True)
class ServerState:
    """
    Immutable server state initialized at startup.

    Contains all services and configuration needed for tool execution.
    """

    session_id: str
    project_path: Path
    claude_pid: int
    temp_dir: tempfile.TemporaryDirectory[str]
    parser_service: SessionParserService
    archive_service: SessionArchiveService


@dataclass(frozen=True)
class ClaudeContext:
    """Context information about the Claude Code session."""

    claude_pid: int
    project_dir: pathlib.Path


# -- Lifespan Management -------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(mcp_server: FastMCP) -> AsyncIterator[None]:
    """
    Manage server lifecycle and state initialization.

    Creates ServerState with all services at startup and cleans up on shutdown.
    """
    # Discover Claude context (PID and project directory)
    claude_context = _find_claude_context()
    project_path = claude_context.project_dir

    # Discover session ID from claude-workspace sessions.json
    session_id = _discover_session_id(claude_context.claude_pid)

    # Create temp directory (cleaned up on exit)
    temp_dir = tempfile.TemporaryDirectory(prefix='claude-session-')
    temp_path = Path(temp_dir.name)

    try:
        # Initialize services
        parser_service = SessionParserService()
        archive_service = SessionArchiveService(
            session_id=session_id,
            temp_dir=temp_path,
            parser_service=parser_service,
            project_path=project_path,  # Real project path from lsof
            claude_pid=claude_context.claude_pid,  # For process-based version detection
        )

        # Create immutable state
        state = ServerState(
            session_id=session_id,
            project_path=project_path,
            claude_pid=claude_context.claude_pid,
            temp_dir=temp_dir,
            parser_service=parser_service,
            archive_service=archive_service,
        )

        # Register tools with closure over state
        register_tools(state)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            stream=sys.stderr,
        )

        logger.info('Session ID: %s', session_id)
        logger.info('Project: %s', project_path)
        logger.info('Temp dir: %s', temp_path)

        yield  # Setup successful; application active

    finally:
        # Cleanup temp directory
        temp_dir.cleanup()
        logger.info('Cleaned up temp directory')


# -- Server Setup --------------------------------------------------------------

server = FastMCP('claude-session', lifespan=lifespan)


# -- Tool Registration (Closure Pattern) ---------------------------------------


def register_tools(state: ServerState) -> None:
    """
    Register MCP tools with closure over server state.

    Args:
        state: Server state containing services
    """

    @server.tool()
    async def save_current_session(
        output_path: str | None = None,
        format: Literal['json', 'zst'] | None = None,
        storage_backend: Literal['local'] = 'local',
        ctx: Context[Any, Any, Any] | None = None,
    ) -> ArchiveMetadata:
        """
        Save current Claude Code session to archive.

        Creates a complete archive of the current session including all agent
        sessions. By default, saves to a temporary file that is cleaned up on
        server shutdown.

        Args:
            output_path: Optional custom output path (default: temp file)
            format: Optional format override ('json' or 'zst')
            storage_backend: Storage backend to use (default: 'local')

        Returns:
            Archive metadata with file path, size, and record counts

        Examples:
            # Save to temp file (default)
            result = await save_current_session()
            # Returns: ArchiveMetadata(file_path='/tmp/.../session-abc123.json', ...)

            # Save to specific path with compression
            result = await save_current_session(
                output_path='/path/to/archive.json.zst',
                format='zst'
            )
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Create storage backend
        if storage_backend == 'local':
            if output_path:
                # User-specified path - use parent directory
                storage = LocalFileSystemStorage(Path(output_path).parent)
            else:
                # Temp file - use temp directory
                storage = LocalFileSystemStorage(Path(state.temp_dir.name))
        else:
            raise ValueError(f'Unsupported storage backend: {storage_backend}')

        # Create archive
        metadata = await state.archive_service.create_archive(
            storage=storage, output_path=output_path, format_param=format
        )

        return metadata

    @server.tool()
    async def restore_session(
        archive_path: str,
        translate_paths: bool = True,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> RestoreResult:
        """
        Restore a saved session archive with a new session ID.

        Works like Claude's teleport feature - creates a new session ID
        while preserving the full conversation history. Loads the session
        into ~/.claude/projects/ allowing you to continue the conversation.

        Args:
            archive_path: Path to session archive file (JSON or .zst)
            translate_paths: Translate paths to current project (default: True)

        Returns:
            RestoreResult with new session ID and restoration details

        Examples:
            # Restore from Downloads
            result = await restore_session('/Users/chris/Downloads/claude-session.json')
            # Returns: RestoreResult(new_session_id='abc123...', ...)

            # Restore without path translation
            result = await restore_session(
                archive_path='/path/to/archive.json',
                translate_paths=False
            )

        Note:
            After restoration, use `claude --resume {new_session_id}` in CLI
            to continue the conversation with the restored history.
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Create restore service for current project
        restore_service = SessionRestoreService(state.project_path)

        # Restore the archive
        result = await restore_service.restore_archive(
            archive_path=archive_path,
            translate_paths=translate_paths,
        )

        logger.info('Session restored with new ID: %s', result.new_session_id)
        logger.info('Use: claude --resume %s', result.new_session_id)

        return result

    @server.tool()
    async def clone_session(
        source_session_id: str | None = None,
        translate_paths: bool = True,
        source_project: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> RestoreResult:
        """
        Clone a session directly without creating an archive file.

        Faster alternative to archive+restore when cloning sessions on the
        same machine. Creates a new session with a fresh UUIDv7 session ID,
        copying all conversation history from the source session.

        Args:
            source_session_id: Session ID to clone (full UUID or prefix).
                              If None, clones the current session.
            translate_paths: Translate paths to current project (default: True)
            source_project: Scope session lookup to this project directory
                           (for disambiguation when multiple projects have the same session ID)

        Returns:
            RestoreResult with new session ID and clone details

        Examples:
            # Clone the current session (fork yourself)
            result = await clone_session()

            # Clone by full session ID
            result = await clone_session('dbae570a-8b88-43e0-a6da-71d649ec07b0')

            # Clone by prefix (must be unique)
            result = await clone_session('dbae570a')

            # Clone without path translation
            result = await clone_session(
                source_session_id='dbae570a',
                translate_paths=False
            )

        Note:
            After cloning, use `claude --resume {new_session_id}` in CLI
            to continue the conversation with the cloned history.
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Default to current session if not specified
        target_id = source_session_id or state.session_id

        # Create clone service for current project
        clone_service = SessionCloneService(state.project_path)

        # Clone the session
        project_filter = _encode_project_filter(source_project)
        result = await clone_service.clone(
            source_session_id=target_id,
            translate_paths=translate_paths,
            project_filter=project_filter,
        )

        logger.info('Session cloned with new ID: %s', result.new_session_id)
        logger.info('Use: claude --resume %s', result.new_session_id)

        return result

    @server.tool()
    async def delete_session(
        session_id: str,
        force: bool = False,
        terminate_running: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        source_project: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> DeleteResult:
        """
        Delete session artifacts with auto-backup.

        By default, only cloned/restored sessions (UUIDv7) can be deleted.
        Native Claude sessions (UUIDv4) require force=True.

        If the session is currently running (another Claude process), set
        terminate_running=True to kill it before deletion. For self-deletion
        (deleting the current session), termination is automatic.

        Deletion is atomic with rollback on failure. A backup is always
        created for rollback capability. The no_backup flag only controls
        whether the backup is kept after successful deletion (for undo).

        Use restore --in-place on the backup to undo a delete.

        Args:
            session_id: Session ID to delete
            force: Required to delete native (UUIDv4) sessions
            terminate_running: Terminate running Claude process before deletion
            no_backup: Don't keep a backup file for undo
            dry_run: If True, show what would be deleted without actually deleting
            source_project: Scope session lookup to this project directory

        Returns:
            DeleteResult with deletion details and backup path

        Examples:
            # Delete a cloned session (auto-backup created)
            result = await delete_session('019b5232-1234-7abc-...')
            # Returns: DeleteResult(backup_path='~/.claude-workspace/claude-session/deleted/...', ...)

            # Preview what would be deleted
            result = await delete_session('019b5232-...', dry_run=True)

            # Delete native session (requires force)
            result = await delete_session('a1b2c3d4-...', force=True)

            # Delete a running session (requires terminate_running)
            result = await delete_session('019b5232-...', terminate_running=True)

            # Delete without backup
            result = await delete_session('019b5232-...', no_backup=True)

        Note:
            To undo a delete, run:
            claude-session restore --in-place <backup_path>
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Resolve prefix to full session ID
        info_service = SessionInfoService()
        project_filter = _encode_project_filter(source_project)
        session_info = await info_service.resolve_session(session_id, project_filter=project_filter)
        full_session_id = session_info.session_id

        # Create delete service - use discovered session_folder for correct path resolution
        delete_service = SessionDeleteService(session_folder=session_info.session_folder)

        # Determine if termination is needed
        is_self_delete = full_session_id == state.session_id

        if is_self_delete:
            # Self-deletion: auto-terminate (no flag needed)
            terminate_pid = state.claude_pid if not dry_run else None
        else:
            # Other session: check if running
            is_running, running_pid = info_service.is_session_running(full_session_id)
            if is_running:
                if dry_run:
                    logger.info('Warning: Session is currently running (PID %s)', running_pid)
                    terminate_pid = None
                elif not terminate_running:
                    # Running without terminate_running: raise exception
                    if running_pid is None:
                        raise RuntimeError(f'is_session_running returned True but pid is None for {full_session_id}')
                    raise RunningSessionDeletionError(full_session_id, running_pid)
                else:
                    # Running with terminate_running: will terminate
                    logger.info('Session is running (PID %s), will terminate before deletion', running_pid)
                    terminate_pid = running_pid
            else:
                terminate_pid = None

        # Delete the session
        result = await delete_service.delete_session(
            session_id=full_session_id,
            force=force,
            no_backup=no_backup,
            dry_run=dry_run,
            terminate_pid_before_delete=terminate_pid,
        )

        if result.success:
            if dry_run:
                logger.info(
                    'Dry run: would delete %d files, %d directories (%s bytes)',
                    result.files_deleted,
                    len(result.directories_removed),
                    f'{result.size_freed_bytes:,}',
                )
            else:
                logger.info(
                    'Deleted %d files, %d directories (%s bytes)',
                    result.files_deleted,
                    len(result.directories_removed),
                    f'{result.size_freed_bytes:,}',
                )
                if result.backup_path:
                    logger.info('Backup: %s', result.backup_path)
                    logger.info('To undo: claude-session restore --in-place %s', result.backup_path)
        else:
            logger.error('Delete failed: %s', result.error_message)

        return result

    @server.tool()
    async def move_session(
        session_id: str,
        target_project: str | None = None,
        source_project: str | None = None,
        force: bool = False,
        terminate_running: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> MoveResult:
        """
        Move a session from one project to another.

        Relocates project-specific artifacts (JSONL files, tool results,
        session memory) to the target project directory. Path references
        in records are translated. The original session ID is preserved.

        Global artifacts (plans, todos, tasks) stay in place since they
        are keyed by session ID, not project path.

        By default, only cloned/restored sessions (UUIDv7) can be moved.
        Native Claude sessions (UUIDv4) require force=True.

        If the session is currently running, set terminate_running=True
        to kill it before moving. Self-move (moving the current session)
        auto-terminates.

        Args:
            session_id: Session ID to move (full UUID or prefix)
            target_project: Target project directory path. Defaults to
                          the current project (where this MCP server runs).
            source_project: Scope session lookup to this project directory
                           (for disambiguation when multiple projects have the same session ID)
            force: Required to move native (UUIDv4) sessions
            terminate_running: Terminate running Claude process before move
            no_backup: Don't keep backup after successful move
            dry_run: Preview what would happen without making changes

        Returns:
            MoveResult with move details and resume command

        Examples:
            # Move a session to current project (default target)
            result = await move_session('019b5232-1234-7abc-...')

            # Preview what would happen
            result = await move_session('019b5232-...', dry_run=True)

            # Move to a specific project
            result = await move_session('019b5232-...', target_project='/path/to/other-project')

            # Move a native session
            result = await move_session('a1b2c3d4-...', force=True)

            # Move a running session
            result = await move_session('019b5232-...', terminate_running=True)
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Default target to current project
        target_path = Path(target_project) if target_project else state.project_path

        # Resolve prefix to full session ID
        info_service = SessionInfoService()
        project_filter = _encode_project_filter(source_project)
        session_info = await info_service.resolve_session(session_id, project_filter=project_filter)
        full_session_id = session_info.session_id

        # Determine if termination is needed
        is_self_move = full_session_id == state.session_id

        if is_self_move:
            terminate_pid = state.claude_pid if not dry_run else None
        else:
            is_running, running_pid = info_service.is_session_running(full_session_id)
            if is_running:
                if dry_run:
                    logger.info('Warning: Session is currently running (PID %s)', running_pid)
                    terminate_pid = None
                elif not terminate_running:
                    if running_pid is None:
                        raise RuntimeError(f'is_session_running returned True but pid is None for {full_session_id}')
                    raise RunningSessionMoveError(full_session_id, running_pid)
                else:
                    logger.info('Session is running (PID %s), will terminate before move', running_pid)
                    terminate_pid = running_pid
            else:
                terminate_pid = None

        # Create move service and execute
        move_service = SessionMoveService(target_path)
        result = await move_service.move_session(
            session_id=full_session_id,
            force=force,
            no_backup=no_backup,
            dry_run=dry_run,
            terminate_pid=terminate_pid,
        )

        if dry_run:
            logger.info(
                'Dry run: would move %d files from %s to %s',
                result.files_moved,
                result.source_project,
                result.target_project,
            )
        else:
            logger.info('Moved %d files, deleted %d from source', result.files_moved, result.files_deleted)
            if result.backup_path:
                logger.info('Backup: %s', result.backup_path)
            logger.info('Resume: %s', result.resume_command)

        for warning in result.warnings:
            logger.error('Warning: %s', warning)

        return result

    @server.tool()
    async def session_lineage(
        session_id: str | None = None,
        source_project: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> LineageTree | None:
        """
        Get lineage tree for a session.

        Returns the complete lineage tree containing the queried session —
        all ancestors and descendants with full operation metadata per node.
        Access tree.nodes[tree.queried_session_id] for the queried node's details.

        Args:
            session_id: Session ID to look up. Accepts full UUID or prefix.
                       If None, uses the current session.
            source_project: Scope session lookup to this project directory

        Returns:
            LineageTree with all nodes, or None if session has no lineage
            (native session with no clones).

        Examples:
            # Get lineage tree for current session
            result = await session_lineage()

            # Get lineage tree for specific session
            result = await session_lineage('019b5232')
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        target_id_or_prefix = session_id or state.session_id

        # Resolve prefix to full session ID and validate session exists
        info_service = SessionInfoService()
        project_filter = _encode_project_filter(source_project)
        session_info = await info_service.resolve_session(target_id_or_prefix, project_filter=project_filter)
        target_id = session_info.session_id

        logger.info('Looking up lineage for session: %s...', target_id[:12])

        lineage_service = LineageService()
        tree = lineage_service.get_full_tree(target_id)

        if tree is None:
            logger.info('No lineage found (native session, no clones)')
            return None

        logger.info('Lineage tree: %d nodes, root=%s...', len(tree.nodes), tree.root_session_id[:12])
        return tree

    @server.tool()
    async def get_session_info(
        session_id: str | None = None,
        source_project: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> SessionContext:
        """
        Get comprehensive information about a session.

        Returns context about a session including session ID, project path,
        session files, origin (how it was created), state, and characteristics.

        Data sources:
        - Session files (~/.claude/projects/)
        - Claude-workspace tracking (~/.claude-workspace/sessions.json)
        - Lineage tracking (~/.claude-workspace/claude-session/lineage.json)

        For other sessions on this machine (tracked in sessions.json), claude_pid
        and machine_id are available from historical data. temp_dir is only
        available for the current session.

        Args:
            session_id: Session ID (full UUID or prefix) to query.
                       If None, returns info for the current session.
            source_project: Scope session lookup to this project directory

        Returns:
            SessionContext with comprehensive session information.

        Examples:
            # Get current session info
            info = await get_session_info()
            # Returns: SessionContext(session_id='...', source='startup', ...)

            # Get info for another session
            info = await get_session_info('019b5232')
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Build current session context for enrichment
        current_context = CurrentSessionContext(
            session_id=state.session_id,
            project_path=state.project_path,
            claude_pid=state.claude_pid,
            temp_dir=state.temp_dir.name,
        )

        # Determine target session
        target_id = session_id or state.session_id

        # Get session info via service
        info_service = SessionInfoService()
        project_filter = _encode_project_filter(source_project)
        return await info_service.get_info(target_id, current_context, project_filter=project_filter)

    @server.tool()
    async def save_session_to_gist(
        session_id: str | None = None,
        gist_id: str | None = None,
        visibility: Literal['public', 'secret'] = 'secret',
        description: str = 'Claude Code Session Archive',
        source_project: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> GistArchiveResult:
        """
        Save a session to GitHub Gist.

        Creates a new gist or updates an existing one with the session archive.
        Requires GitHub authentication via GITHUB_TOKEN env var or gh CLI.

        Args:
            session_id: Session to save (full UUID or prefix). Defaults to current session.
            gist_id: Existing gist ID to update. If None, creates a new gist.
            visibility: 'public' or 'secret' (default: secret)
            description: Gist description
            source_project: Scope session lookup to this project directory

        Returns:
            GistArchiveResult with gist URL, ID, and archive metadata.

        Authentication:
            Token is obtained from (in order):
            1. GITHUB_TOKEN environment variable
            2. `gh auth token` (GitHub CLI)

            If neither is available, an error is raised with setup instructions.

        Examples:
            # Save current session to new secret gist
            result = await save_session_to_gist()

            # Save specific session to public gist
            result = await save_session_to_gist(
                session_id='019b5232',
                visibility='public'
            )

            # Update an existing gist
            result = await save_session_to_gist(gist_id='abc123xyz')
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Get GitHub token
        token = _get_github_token()
        if not token:
            raise ValueError(
                'GitHub authentication required.\n\n'
                'Set up authentication using one of:\n'
                '1. Set GITHUB_TOKEN environment variable\n'
                '2. Install and authenticate GitHub CLI: gh auth login\n\n'
                'To create a token manually:\n'
                '  1. Go to https://github.com/settings/tokens\n'
                '  2. Generate new token (classic)\n'
                "  3. Select 'gist' scope\n"
                '  4. Set as GITHUB_TOKEN in your environment'
            )

        # Determine target session (resolve prefix if needed)
        target_id_or_prefix = session_id or state.session_id

        # Resolve prefix to full session ID
        info_service = SessionInfoService()
        project_filter = _encode_project_filter(source_project)
        session_info = await info_service.resolve_session(target_id_or_prefix, project_filter=project_filter)
        target_id = session_info.session_id

        logger.info('Saving session to Gist: %s...', target_id[:12])

        # Determine correct PID for version detection:
        # - If archiving current session: use live PID from state
        # - If archiving other session: get historical PID from sessions.json (if available)
        is_current_session = target_id == state.session_id
        if is_current_session:
            archive_claude_pid: int | None = state.claude_pid
        else:
            # Get target session's PID from workspace sessions.json
            workspace_session = info_service._load_workspace_session(target_id)
            archive_claude_pid = workspace_session.metadata.claude_pid if workspace_session else None

        # Create Gist storage backend
        storage = GistStorage(
            token=token,
            gist_id=gist_id,
            visibility=visibility,
            description=description,
        )

        archive_service = SessionArchiveService(
            session_id=target_id,
            temp_dir=Path(state.temp_dir.name),
            parser_service=state.parser_service,
            session_folder=session_info.session_folder,
            claude_pid=archive_claude_pid,  # Target session's PID (not current session's PID)
        )

        # Create archive (zst for better compression - auto base64-encoded for Gist)
        metadata = await archive_service.create_archive(
            storage=storage,
            output_path=None,  # Let storage generate path
            format_param='zst',  # Zstd compression (6x smaller than JSON)
        )

        # Get gist ID from storage (set after creation)
        final_gist_id = storage.gist_id or ''

        logger.info('Session uploaded to Gist: %s', metadata.file_path)
        logger.info('Gist ID: %s', final_gist_id)
        logger.info('To restore: claude-session restore gist://%s', final_gist_id)

        return GistArchiveResult(
            gist_url=metadata.file_path,  # GistStorage returns html_url as file_path
            gist_id=final_gist_id,
            session_id=target_id,
            format=metadata.format,
            size_mb=metadata.size_mb,
            session_records=metadata.session_records,
            agent_records=metadata.agent_records,
            file_count=metadata.file_count,
            restore_command=f'claude-session restore gist://{final_gist_id}',
        )


# -- Server Entry Point --------------------------------------------------------


def main() -> None:
    """Run the MCP server."""
    server.run()


# -- Private Helpers -----------------------------------------------------------


def _find_claude_context() -> ClaudeContext:
    """
    Find Claude process and extract its context (PID, project directory).

    Uses shared process tree walk to find Claude, then lsof to determine CWD.

    Returns:
        ClaudeContext with PID and project directory

    Raises:
        RuntimeError: If Claude process cannot be found or CWD cannot be determined
    """
    pid = find_ancestor_claude_pid()
    if pid is None:
        raise RuntimeError('Could not find Claude process in parent tree')

    # Get Claude's CWD using lsof
    result = subprocess.run(
        ['lsof', '-p', str(pid), '-a', '-d', 'cwd'],
        check=False,
        capture_output=True,
        text=True,
    )

    cwd = None
    for line in result.stdout.split('\n'):
        if 'cwd' in line:
            parts = line.split()
            if len(parts) >= 9:
                cwd = pathlib.Path(' '.join(parts[8:]))
                break

    if not cwd:
        raise RuntimeError(f'Found Claude process (PID {pid}) but could not determine CWD')

    # Verify by checking if Claude has config dir files open
    result = subprocess.run(['lsof', '-p', str(pid)], check=False, capture_output=True, text=True)

    config_dir = str(get_claude_config_home_dir())
    claude_files = []
    for line in result.stdout.split('\n'):
        if config_dir in line:
            parts = line.split()
            if len(parts) >= 9:
                file_path = ' '.join(parts[8:])
                claude_files.append(file_path)

    if not claude_files:
        raise RuntimeError(
            f'Found Claude process (PID {pid}) with CWD {cwd}, '
            f'but no {config_dir}/ files are open - may not be a Claude project'
        )

    return ClaudeContext(claude_pid=pid, project_dir=cwd)


def _discover_session_id(claude_pid: int) -> str:
    """
    Discover Claude Code session ID from claude-workspace sessions.json.

    Delegates to shared session resolution with MCP-specific retry count
    (10 attempts) to handle the startup race with SessionStart hook.

    Args:
        claude_pid: PID of the Claude process (from _find_claude_context)

    Returns:
        Session ID (UUID string)

    Raises:
        RuntimeError: If no matching session found or multiple active sessions match
    """
    session_id = resolve_session_id_from_pid(claude_pid, max_attempts=10)
    if session_id is None:
        sessions_file = pathlib.Path.home() / '.claude-workspace' / 'sessions.json'
        raise RuntimeError(
            f'Could not find active session for Claude PID {claude_pid} in {sessions_file} '
            f'after 10 attempts. Ensure claude-workspace SessionStart hook is configured.'
        )
    return session_id


def _get_github_token() -> str | None:
    """
    Get GitHub token from environment or gh CLI.

    Checks in order:
    1. GITHUB_TOKEN environment variable
    2. `gh auth token` command (GitHub CLI)

    Returns:
        GitHub token string, or None if not available
    """
    # 1. Check environment
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        return token

    # 2. Try gh CLI
    result = subprocess.run(
        ['gh', 'auth', 'token'],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    return None


if __name__ == '__main__':
    main()
