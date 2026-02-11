"""Python interpreter domain logic.

Protocol-agnostic service layer for code execution, variable management,
and interpreter lifecycle. Used by both MCP tools and the HTTP bridge.
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import importlib
import io
import pathlib
import subprocess
import sys
import tempfile
import typing
from collections.abc import Set

from local_lib.utils import encode_project_path, humanize_seconds

from python_interpreter.discovery import discover_session_id, find_claude_context
from python_interpreter.manager import ExternalInterpreterManager, InterpreterConfig
from python_interpreter.models import InterpreterInfo, SessionInfo, TruncationInfo

__all__ = [
    'ServerState',
    'PythonInterpreterService',
    'LoggerProtocol',
    'SimpleLogger',
    'PackageInstallationError',
    'MaxInstallAttemptsError',
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


class MaxInstallAttemptsError(Exception):
    """Raised when maximum installation attempts are exceeded."""


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

        # Python execution scope - persists across executions (builtin interpreter only)
        self.scope_globals: dict[str, typing.Any] = {}

        # Snapshot of sys.modules at startup - used by reset() to only reload user-imported modules
        self._initial_modules: Set[str] = set(sys.modules.keys())


class PythonInterpreterService:
    """Python interpreter service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: ServerState) -> None:
        self.state = state

    async def execute(self, code: str, logger: LoggerProtocol, interpreter: str | None = None) -> str:
        """Execute Python code in persistent scope.

        Args:
            code: Python code to execute
            logger: Logger instance
            interpreter: Interpreter name (None = builtin with auto-install)

        Returns:
            Plain string output (stdout + stderr + result + truncation notice)
        """
        if interpreter is None:
            # Builtin interpreter with auto-install
            await logger.info(f'Executing in builtin interpreter ({len(code)} chars)')

            result, truncation_info = self._execute_with_file_handling(code)

            if truncation_info:
                await logger.warning(f'Output truncated: {truncation_info.original_size} chars')
                separator = '=' * 50
                return f'{result}\n\n{separator}\n# OUTPUT TRUNCATED\n# Original size: {truncation_info.original_size:,} chars\n# Full output: {truncation_info.file_path}\n{separator}'

            return result
        else:
            # External interpreter
            await logger.info(f"Executing in '{interpreter}' ({len(code)} chars)")

            response = self.state.interpreter_manager.execute(interpreter, code)

            if response.get('error'):
                error: str = response['error']
                return error

            # Combine output
            parts = [response.get('stdout', ''), response.get('stderr', ''), response.get('result', '')]
            output = '\n'.join(p for p in parts if p)

            # Handle truncation
            if len(output) > CHARACTER_LIMIT:
                timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
                file_path = self.state.output_dir / f'output_{timestamp}.txt'
                file_path.write_text(output, encoding='utf-8')
                separator = '=' * 50
                return f'{output[:CHARACTER_LIMIT]}\n\n{separator}\n# OUTPUT TRUNCATED\n# Original size: {len(output):,} chars\n# Full output: {file_path}\n{separator}'

            return output

    async def reset(self, logger: LoggerProtocol) -> str:
        """Clear all variables and reload user-imported modules from disk.

        Only reloads modules imported after server startup (not the server's own dependencies).
        """
        await logger.info('Resetting Python interpreter scope')

        # Clear execution scope
        var_count = len([k for k in self.state.scope_globals if not k.startswith('__')])
        self.state.scope_globals.clear()

        # Reload only modules imported by user code (not present at server startup)
        modules_to_reload = []
        for name, module in list(sys.modules.items()):
            if name in self.state._initial_modules:
                continue
            if module is None:
                continue
            if not hasattr(module, '__file__') or module.__file__ is None:
                continue
            modules_to_reload.append(module)

        reload_count = 0
        for module in reversed(modules_to_reload):
            try:
                importlib.reload(module)
                reload_count += 1
                await logger.info(f'Reloaded: {module.__name__}')
            except Exception as e:
                await logger.warning(f'Failed to reload {module.__name__}: {e}')

        await logger.info(f'Reset complete - cleared {var_count} vars, reloaded {reload_count} modules')
        return f'Scope reset - cleared {var_count} vars, reloaded {reload_count} modules'

    async def list_vars(self, logger: LoggerProtocol, interpreter: str | None = None) -> str:
        """List all user-defined variables in persistent scope.

        Args:
            logger: Logger instance
            interpreter: Interpreter name (None = builtin)
        """
        if interpreter is None:
            await logger.info('Listing builtin interpreter variables')

            if not self.state.scope_globals:
                return 'No variables defined'

            user_vars = [name for name in self.state.scope_globals if not name.startswith('__')]
            if not user_vars:
                return 'No variables defined'

            return ', '.join(sorted(user_vars))
        else:
            await logger.info(f"Listing variables in '{interpreter}'")
            return self.state.interpreter_manager.list_vars(interpreter)

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

        pid, started_at = self.state.interpreter_manager.add_interpreter(config)

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
        self.state.interpreter_manager.stop_interpreter(name)
        return f"Interpreter '{name}' stopped"

    async def list_interpreters(self, logger: LoggerProtocol) -> list[InterpreterInfo]:
        """List all interpreters (builtin and external).

        Args:
            logger: Logger instance

        Returns:
            List of InterpreterInfo
        """
        await logger.info('Listing interpreters')

        builtin = InterpreterInfo(
            name='builtin',
            type='builtin',
        )

        external_infos = []
        for name, config, pid, started_at in self.state.interpreter_manager.get_interpreters():
            uptime_seconds = (datetime.datetime.now(datetime.UTC) - started_at).total_seconds()
            external_infos.append(
                InterpreterInfo(
                    name=name,
                    type='external',
                    python_path=str(config.python_path),
                    cwd=str(config.cwd or self.state.project_dir),
                    pid=pid,
                    started_at=started_at,
                    uptime=humanize_seconds(uptime_seconds),
                    has_startup_script=config.startup_script is not None,
                )
            )

        return [builtin] + external_infos

    def _execute_with_file_handling(self, code: str) -> tuple[str, TruncationInfo | None]:
        """Execute code and handle large output by saving to temp file."""
        result = _execute_code(code, self.state.scope_globals)

        if len(result) > CHARACTER_LIMIT:
            timestamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
            filename = f'output_{timestamp}.txt'
            file_path = self.state.output_dir / filename

            file_path.write_text(result, encoding='utf-8')

            truncation_info = TruncationInfo(
                file_path=str(file_path),
                original_size=len(result),
                truncated_at=CHARACTER_LIMIT,
            )

            return result[:CHARACTER_LIMIT], truncation_info

        return result, None


# Private helper functions for builtin code execution


def _detect_expression(code: str) -> tuple[bool, str | None]:
    """Detect if last line is an expression. Returns (is_expr, last_line)."""
    try:
        tree = ast.parse(code)
        if not tree.body:
            return False, None

        last_node = tree.body[-1]
        if isinstance(last_node, ast.Expr):
            last_line = ast.unparse(last_node.value)
            return True, last_line
        return False, None
    except SyntaxError:
        return False, None


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


def _execute_code(code: str, scope_globals: dict[str, typing.Any]) -> str:
    """Execute code in provided scope. Auto-installs missing packages.

    Args:
        code: Python code to execute
        scope_globals: Global scope dictionary for execution

    Returns:
        Output string from successful execution
    """
    max_install_attempts = 3
    successfully_installed: set[str] = set()
    failed_installs: set[str] = set()

    while True:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            try:
                is_expr, last_line = _detect_expression(code)

                if is_expr and last_line:
                    lines = code.splitlines()
                    code_without_last = '\n'.join(lines[:-1])

                    if code_without_last.strip():
                        exec(code_without_last, scope_globals)

                    result = eval(last_line, scope_globals)

                    stdout_value = stdout_capture.getvalue()
                    stderr_value = stderr_capture.getvalue()

                    if stdout_value or stderr_value:
                        return (stdout_value + stderr_value).rstrip()
                    else:
                        return repr(result) if result is not None else ''
                else:
                    exec(code, scope_globals)

                    stdout_value = stdout_capture.getvalue()
                    stderr_value = stderr_capture.getvalue()
                    return (stdout_value + stderr_value).rstrip()

            except ModuleNotFoundError as e:
                module_name = e.name

                total_attempts = len(successfully_installed) + len(failed_installs)
                should_attempt = (
                    module_name
                    and module_name not in successfully_installed
                    and module_name not in failed_installs
                    and total_attempts < max_install_attempts
                )

                if should_attempt and module_name is not None:
                    try:
                        message = _install_package(module_name)
                        successfully_installed.add(module_name)
                        print(message, file=sys.stderr)
                        continue

                    except PackageInstallationError as install_error:
                        failed_installs.add(module_name)
                        raise PackageInstallationError(f'{install_error}\n\nOriginal import error: {e}') from e
                else:
                    if total_attempts >= max_install_attempts:
                        raise MaxInstallAttemptsError(
                            f'Maximum installation attempts ({max_install_attempts}) exceeded.\n'
                            f'Successfully installed: {successfully_installed}\n'
                            f'Failed: {failed_installs}'
                        ) from e
                    elif module_name in failed_installs:
                        raise PackageInstallationError(f"Package '{module_name}' already failed to install") from e
                    else:
                        raise
