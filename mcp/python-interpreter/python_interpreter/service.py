"""Python interpreter domain logic.

Protocol-agnostic service layer for code execution, variable management,
and interpreter lifecycle. Used by both MCP tools and the HTTP bridge.
"""

from __future__ import annotations

import asyncio
import datetime
import pathlib
import subprocess
import sys
import tempfile
import typing

from local_lib.utils import encode_project_path, humanize_seconds

from python_interpreter.discovery import discover_session_id, find_claude_context
from python_interpreter.manager import ExternalInterpreterManager, InterpreterConfig
from python_interpreter.models import InterpreterInfo, SessionInfo

__all__ = [
    'ServerState',
    'PythonInterpreterService',
    'LoggerProtocol',
    'SimpleLogger',
    'PackageInstallationError',
    'CHARACTER_LIMIT',
]


# Constants
CHARACTER_LIMIT = 25_000

# Import name to PyPI package name mappings for common mismatches
# Philosophy: Explicit curated list (secure) vs dynamic lookup (complex/risky)
IMPORT_TO_PACKAGE_MAP = {
    'aws_cdk': 'aws-cdk-lib',
    'bs4': 'beautifulsoup4',
    'dateutil': 'python-dateutil',
    'OpenSSL': 'pyOpenSSL',
    'PIL': 'pillow',
    'psycopg2': 'psycopg2-binary',
    'skimage': 'scikit-image',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
}


class PackageInstallationError(Exception):
    """Raised when auto-installation of a package fails."""


class LoggerProtocol(typing.Protocol):
    """Protocol for logger - allows service to be MCP-agnostic."""

    async def info(self, message: str) -> None: ...
    async def warning(self, message: str) -> None: ...
    async def error(self, message: str) -> None: ...


class SimpleLogger:
    """Simple logger for HTTP endpoint - logs to stdout."""

    async def info(self, message: str) -> None:
        print(f'INFO: {message}')

    async def warning(self, message: str) -> None:
        print(f'WARNING: {message}')

    async def error(self, message: str) -> None:
        print(f'ERROR: {message}')


class ServerState:
    """Container for all server state - initialized once at startup, never Optional."""

    @classmethod
    def create(cls) -> typing.Self:
        """Factory method to create and initialize server state - fails fast if anything goes wrong."""
        started_at = datetime.datetime.now(datetime.UTC)

        # Find Claude context (PID, project directory) by walking process tree
        claude_context = find_claude_context()
        print(f'Claude context: PID={claude_context.claude_pid}, Project={claude_context.project_dir}', file=sys.stderr)

        # Discover session ID from claude-workspace sessions.json
        session_id = discover_session_id(claude_context.claude_pid)
        print(f'Discovered session ID: {session_id}', file=sys.stderr, flush=True)

        # Compute transcript path using Claude Code's path encoding convention
        encoded_project_path = encode_project_path(claude_context.project_dir)
        transcript_path = (
            pathlib.Path.home() / '.claude' / 'projects' / encoded_project_path / f'{session_id}.jsonl'
        ).resolve(strict=True)

        # Initialize temp directory for large outputs
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = pathlib.Path(temp_dir.name)
        print(f'Temp directory for large outputs: {output_dir}', file=sys.stderr)

        # Remove stale socket if it exists
        if claude_context.socket_path.exists():
            claude_context.socket_path.unlink()

        print(f'Unix socket path: {claude_context.socket_path}', file=sys.stderr)

        # Initialize external interpreter manager
        interpreter_manager = ExternalInterpreterManager(claude_context.project_dir)

        # Auto-start builtin interpreter using current Python executable
        builtin_config = InterpreterConfig(
            name='builtin',
            python_path=pathlib.Path(sys.executable),
        )
        interpreter_manager.add_interpreter(builtin_config)
        print('Builtin interpreter started', file=sys.stderr)

        return cls(
            session_id=session_id,
            started_at=started_at,
            project_dir=claude_context.project_dir,
            socket_path=claude_context.socket_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            temp_dir=temp_dir,
            claude_pid=claude_context.claude_pid,
            interpreter_manager=interpreter_manager,
        )

    def __init__(
        self,
        session_id: str,
        started_at: datetime.datetime,
        project_dir: pathlib.Path,
        socket_path: pathlib.Path,
        transcript_path: pathlib.Path,
        output_dir: pathlib.Path,
        temp_dir: tempfile.TemporaryDirectory[str],
        claude_pid: int,
        interpreter_manager: ExternalInterpreterManager,
    ) -> None:
        # Identity
        self.session_id = session_id
        self.started_at = started_at

        # Path configuration
        self.project_dir = project_dir
        self.socket_path = socket_path
        self.transcript_path = transcript_path
        self.output_dir = output_dir

        # Resources
        self.temp_dir = temp_dir
        self.claude_pid = claude_pid
        self.interpreter_manager = interpreter_manager


class PythonInterpreterService:
    """Python interpreter service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: ServerState) -> None:
        self.state = state

    async def execute(self, code: str, logger: LoggerProtocol, interpreter: str = 'builtin') -> str:
        """Execute Python code in persistent scope.

        Args:
            code: Python code to execute
            logger: Logger instance
            interpreter: Interpreter name (defaults to 'builtin' with auto-install)

        Returns:
            Plain string output (stdout + stderr + result + truncation notice)
        """
        await logger.info(f"Executing in '{interpreter}' ({len(code)} chars)")

        # Auto-install retry loop (only for builtin)
        max_attempts = 3 if interpreter == 'builtin' else 1
        for attempt in range(max_attempts):
            response = await asyncio.to_thread(self.state.interpreter_manager.execute, interpreter, code)

            # Handle auto-install for builtin
            if response.error_type == 'ModuleNotFoundError' and response.module_name and interpreter == 'builtin':
                if attempt < max_attempts - 1:
                    await logger.info(f"Auto-installing '{response.module_name}'")
                    install_msg = await asyncio.to_thread(_install_package, response.module_name)
                    await logger.info(install_msg)
                    continue

            # Return error if present
            if response.error:
                return response.error

            # Combine output
            parts = [response.stdout, response.stderr, response.result]
            output = '\n'.join(p for p in parts if p)

            # Handle truncation
            if len(output) > CHARACTER_LIMIT:
                timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
                file_path = self.state.output_dir / f'output_{timestamp}.txt'
                file_path.write_text(output, encoding='utf-8')
                separator = '=' * 50
                await logger.warning(f'Output truncated: {len(output)} chars')
                return f'{output[:CHARACTER_LIMIT]}\n\n{separator}\n# OUTPUT TRUNCATED\n# Original size: {len(output):,} chars\n# Full output: {file_path}\n{separator}'

            return output

        return response.error or 'Auto-install failed'

    async def reset(self, logger: LoggerProtocol) -> str:
        """Clear all variables from builtin interpreter scope."""
        await logger.info('Resetting builtin interpreter scope')

        response = await asyncio.to_thread(self.state.interpreter_manager.reset, 'builtin')

        if response.error:
            raise RuntimeError(f'Reset failed: {response.error}')

        return response.result

    async def list_vars(self, logger: LoggerProtocol, interpreter: str = 'builtin') -> str:
        """List all user-defined variables in persistent scope.

        Args:
            logger: Logger instance
            interpreter: Interpreter name (defaults to 'builtin')
        """
        await logger.info(f"Listing variables in '{interpreter}'")

        response = await asyncio.to_thread(self.state.interpreter_manager.list_vars, interpreter)

        if response.error:
            raise RuntimeError(f'List vars failed: {response.error}')

        return response.result

    async def get_session_info(self, logger: LoggerProtocol) -> SessionInfo:
        """Get comprehensive session and server metadata."""
        await logger.info('Getting session info')

        uptime_seconds = (datetime.datetime.now(datetime.UTC) - self.state.started_at).total_seconds()
        uptime = humanize_seconds(uptime_seconds)

        return SessionInfo(
            session_id=self.state.session_id,
            project_dir=str(self.state.project_dir),
            socket_path=str(self.state.socket_path),
            transcript_path=str(self.state.transcript_path),
            output_dir=str(self.state.output_dir),
            claude_pid=self.state.claude_pid,
            started_at=self.state.started_at,
            uptime=uptime,
        )

    async def add_interpreter(self, config: InterpreterConfig, logger: LoggerProtocol) -> InterpreterInfo:
        """Add and start an external interpreter.

        Args:
            config: Interpreter configuration
            logger: Logger instance

        Returns:
            InterpreterInfo with runtime details
        """
        await logger.info(f"Adding interpreter '{config.name}' ({config.python_path})")

        pid, started_at = await asyncio.to_thread(self.state.interpreter_manager.add_interpreter, config)

        return InterpreterInfo(
            name=config.name,
            type='external',
            python_path=str(config.python_path),
            cwd=str(config.cwd or self.state.project_dir),
            pid=pid,
            started_at=started_at,
            uptime='0s',
            has_startup_script=config.startup_script is not None,
        )

    async def stop_interpreter(self, name: str, logger: LoggerProtocol) -> str:
        """Stop an external interpreter.

        Args:
            name: Interpreter name
            logger: Logger instance
        """
        await logger.info(f"Stopping interpreter '{name}'")
        await asyncio.to_thread(self.state.interpreter_manager.stop_interpreter, name)
        return f"Interpreter '{name}' stopped"

    async def list_interpreters(self, logger: LoggerProtocol) -> list[InterpreterInfo]:
        """List all interpreters (builtin and external).

        Args:
            logger: Logger instance

        Returns:
            List of InterpreterInfo
        """
        await logger.info('Listing interpreters')

        result = []
        for name, config, pid, started_at in self.state.interpreter_manager.get_interpreters():
            uptime_seconds = (datetime.datetime.now(datetime.UTC) - started_at).total_seconds()
            result.append(
                InterpreterInfo(
                    name=name,
                    type='builtin' if name == 'builtin' else 'external',
                    python_path=str(config.python_path),
                    cwd=str(config.cwd or self.state.project_dir),
                    pid=pid,
                    started_at=started_at,
                    uptime=humanize_seconds(uptime_seconds),
                    has_startup_script=config.startup_script is not None,
                )
            )

        return result


def _install_package(import_name: str) -> str:
    """Auto-install a missing package using uv pip install.

    Args:
        import_name: Name of the module that failed to import

    Returns:
        Success message with installation details

    Raises:
        PackageInstallationError: If installation fails for any reason
    """
    package_name = IMPORT_TO_PACKAGE_MAP.get(import_name, import_name)

    try:
        result = subprocess.run(
            ['uv', 'pip', 'install', '--python', sys.executable, package_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise PackageInstallationError(f'Timeout (>60s) while installing {package_name}')

    if result.returncode == 0:
        details = result.stdout.strip() if result.stdout.strip() else 'No output'
        if import_name != package_name:
            return f'Auto-installed {package_name} (for import {import_name})\n{details}'
        else:
            return f'Auto-installed {package_name}\n{details}'
    else:
        stderr = result.stderr.strip() if result.stderr else 'No error details'
        raise PackageInstallationError(f'Failed to install {package_name}\n{stderr}')
