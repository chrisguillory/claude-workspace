"""Python interpreter domain logic.

Protocol-agnostic service layer for code execution and interpreter lifecycle.
Used by both MCP tools and the HTTP bridge.
"""

from __future__ import annotations

import asyncio
import datetime
import pathlib
import subprocess
import sys
import tempfile
import typing
from collections.abc import Sequence

from local_lib.utils import encode_project_path, humanize_seconds

from python_interpreter.discovery import discover_session_id, find_claude_context
from python_interpreter.manager import ExternalInterpreterManager, InterpreterConfig
from python_interpreter.models import (
    InterpreterInfo,
    InterpreterSource,
    JetBrainsRunConfig,
    JetBrainsSDKEntry,
    SavedInterpreterConfig,
)
from python_interpreter.registry import InterpreterRegistryManager

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
        ).resolve(strict=False)

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

        # Initialize interpreter registry for saved configurations
        interpreter_registry = InterpreterRegistryManager()
        saved_count = len(interpreter_registry.list_saved())
        if saved_count:
            print(f'Loaded {saved_count} saved interpreter config(s)', file=sys.stderr)

        # Discover JetBrains IDE interpreter configs
        jetbrains_sdks: Sequence[JetBrainsSDKEntry] = ()
        jetbrains_runs: Sequence[JetBrainsRunConfig] = ()
        registry = interpreter_registry.load()
        if registry.discover_jetbrains:
            from python_interpreter.jetbrains import discover_run_configs, discover_sdk_entries

            jetbrains_sdks = tuple(discover_sdk_entries(claude_context.project_dir))
            jetbrains_runs = tuple(discover_run_configs(claude_context.project_dir, sdk_entries=jetbrains_sdks))
            if jetbrains_sdks or jetbrains_runs:
                print(
                    f'  JetBrains: {len(jetbrains_sdks)} SDK(s), {len(jetbrains_runs)} console config(s)',
                    file=sys.stderr,
                )

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
            interpreter_registry=interpreter_registry,
            jetbrains_sdks=jetbrains_sdks,
            jetbrains_runs=jetbrains_runs,
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
        interpreter_registry: InterpreterRegistryManager,
        jetbrains_sdks: Sequence[JetBrainsSDKEntry],
        jetbrains_runs: Sequence[JetBrainsRunConfig],
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
        self.interpreter_registry = interpreter_registry

        # JetBrains discovered configs
        self.jetbrains_sdks = jetbrains_sdks
        self.jetbrains_runs = jetbrains_runs


class PythonInterpreterService:
    """Python interpreter service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: ServerState) -> None:
        self.state = state
        self._interpreter_sources: dict[str, InterpreterSource] = {'builtin': 'builtin'}
        self._start_locks: dict[str, asyncio.Lock] = {}

    def _get_start_lock(self, name: str) -> asyncio.Lock:
        """Get or create a per-interpreter startup lock."""
        if name not in self._start_locks:
            self._start_locks[name] = asyncio.Lock()
        return self._start_locks[name]

    async def _auto_start_interpreter(self, name: str, logger: LoggerProtocol) -> None:
        """Auto-start a stopped interpreter from known config.

        Resolution order: builtin -> saved registry -> jetbrains-run -> jetbrains-sdk.
        """
        config: InterpreterConfig
        source: InterpreterSource

        if name == 'builtin':
            config = InterpreterConfig(
                name='builtin',
                python_path=pathlib.Path(sys.executable),
            )
            source = 'builtin'
        else:
            # Check saved registry
            saved = self.state.interpreter_registry.get(name)
            if saved is not None:
                resolved_path = pathlib.Path(saved.python_path)
                if not resolved_path.is_absolute():
                    resolved_path = self.state.project_dir / resolved_path
                config = InterpreterConfig(
                    name=name,
                    python_path=resolved_path,
                    cwd=pathlib.Path(saved.cwd) if saved.cwd else None,
                    env=dict(saved.env) if saved.env else None,
                    startup_script=saved.startup_script,
                )
                source = 'saved'
            else:
                # Check JetBrains run configs
                run_match = next((c for c in self.state.jetbrains_runs if c.name == name), None)
                if run_match is not None:
                    if run_match.python_path is None:
                        raise ValueError(
                            f"JetBrains config '{name}' has unresolved SDK. "
                            f'Register it with an explicit python_path via register_interpreter.'
                        )
                    startup_script = _build_jetbrains_startup_script(run_match)
                    config = InterpreterConfig(
                        name=name,
                        python_path=pathlib.Path(run_match.python_path),
                        cwd=pathlib.Path(run_match.cwd) if run_match.cwd else None,
                        env=dict(run_match.env) if run_match.env else None,
                        startup_script=startup_script,
                    )
                    source = 'jetbrains-run'
                else:
                    # Check JetBrains SDK entries
                    sdk_match = next((e for e in self.state.jetbrains_sdks if e.name == name), None)
                    if sdk_match is not None:
                        config = InterpreterConfig(
                            name=name,
                            python_path=pathlib.Path(sdk_match.python_path),
                        )
                        source = 'jetbrains-sdk'
                    else:
                        raise ValueError(
                            f"Interpreter '{name}' not found. "
                            f'Use register_interpreter to create one, '
                            f'or list_interpreters to see available options.'
                        )

        await logger.info(f"Auto-starting '{name}' ({source}, {config.python_path})")
        await asyncio.to_thread(self.state.interpreter_manager.add_interpreter, config)
        self._interpreter_sources[name] = source

    async def execute(self, code: str, logger: LoggerProtocol, interpreter: str = 'builtin') -> str:
        """Execute Python code in persistent scope.

        Auto-starts stopped interpreters from known configs (saved, JetBrains, builtin).

        Args:
            code: Python code to execute
            logger: Logger instance
            interpreter: Interpreter name (defaults to 'builtin' with auto-install)

        Returns:
            Plain string output (stdout + stderr + result + truncation notice)
        """
        await logger.info(f"Executing in '{interpreter}' ({len(code)} chars)")

        # Auto-start if not running (with per-interpreter lock to prevent races)
        if not await asyncio.to_thread(self.state.interpreter_manager.is_running, interpreter):
            async with self._get_start_lock(interpreter):
                if not await asyncio.to_thread(self.state.interpreter_manager.is_running, interpreter):
                    await self._auto_start_interpreter(interpreter, logger)

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

    async def register_interpreter(
        self,
        name: str,
        python_path: pathlib.Path,
        logger: LoggerProtocol,
        cwd: pathlib.Path | None = None,
        env: dict[str, str] | None = None,
        startup_script: str | None = None,
        description: str | None = None,
    ) -> InterpreterInfo:
        """Save a new interpreter configuration to disk. Does NOT start it.

        The interpreter auto-starts on next execute() call targeting it.
        If already running, the config is updated on disk but takes effect after restart.
        """
        if name == 'builtin':
            raise ValueError("Cannot register 'builtin' — it uses the server's Python automatically")

        await logger.info(f"Registering interpreter '{name}' ({python_path})")

        saved_config = SavedInterpreterConfig(
            python_path=str(python_path),
            cwd=str(cwd) if cwd else None,
            env=env,
            startup_script=startup_script,
            description=description,
        )
        self.state.interpreter_registry.save_interpreter(name, saved_config)
        await logger.info(f"Saved interpreter '{name}' to registry")

        # Single atomic call — avoids TOCTOU race between is_running and get_interpreters
        running = await asyncio.to_thread(self.state.interpreter_manager.get_interpreters)
        for r_name, r_config, r_pid, r_started_at in running:
            if r_name == name:
                uptime_seconds = (datetime.datetime.now(datetime.UTC) - r_started_at).total_seconds()
                return InterpreterInfo(
                    name=name,
                    source='saved',
                    state='running',
                    python_path=str(r_config.python_path),
                    cwd=str(r_config.cwd or self.state.project_dir),
                    has_startup_script=r_config.startup_script is not None,
                    description=description,
                    pid=r_pid,
                    started_at=r_started_at,
                    uptime=humanize_seconds(uptime_seconds),
                )

        return InterpreterInfo(
            name=name,
            source='saved',
            state='stopped',
            python_path=str(python_path),
            cwd=str(cwd) if cwd else None,
            has_startup_script=startup_script is not None,
            description=description,
            pid=None,
            started_at=None,
            uptime=None,
        )

    async def stop_interpreter(self, name: str, logger: LoggerProtocol, remove: bool) -> str:
        """Stop an interpreter subprocess.

        All interpreters (including builtin) can be stopped.
        Stopped interpreters auto-restart on next execute() call.
        If remove=True, also deletes the saved config permanently.
        """
        await logger.info(f"Stopping interpreter '{name}'")
        await asyncio.to_thread(self.state.interpreter_manager.stop_interpreter, name)
        self._interpreter_sources.pop(name, None)

        if remove:
            removed = self.state.interpreter_registry.remove_interpreter(name)
            if removed:
                return f"Interpreter '{name}' stopped and config removed"
            return f"Interpreter '{name}' stopped (no saved config to remove)"

        return f"Interpreter '{name}' stopped (will auto-restart on next execute)"

    async def list_interpreters(self, logger: LoggerProtocol) -> list[InterpreterInfo]:
        """List all interpreters (running, saved, and discovered)."""
        await logger.info('Listing interpreters')

        running = self.state.interpreter_manager.get_interpreters()
        running_names = {name for name, _, _, _ in running}
        saved_configs = self.state.interpreter_registry.list_saved()

        result: list[InterpreterInfo] = []

        # Running interpreters
        for name, config, pid, started_at in running:
            uptime_seconds = (datetime.datetime.now(datetime.UTC) - started_at).total_seconds()
            saved = saved_configs.get(name)

            source: InterpreterSource
            if name in self._interpreter_sources:
                source = self._interpreter_sources[name]
            elif name == 'builtin':
                source = 'builtin'
            elif saved is not None:
                source = 'saved'
            else:
                source = 'saved'

            result.append(
                InterpreterInfo(
                    name=name,
                    source=source,
                    state='running',
                    python_path=str(config.python_path),
                    cwd=str(config.cwd or self.state.project_dir),
                    has_startup_script=config.startup_script is not None,
                    description=saved.description if saved else None,
                    pid=pid,
                    started_at=started_at,
                    uptime=humanize_seconds(uptime_seconds),
                )
            )

        # Saved-but-stopped interpreters
        for name, saved_config in saved_configs.items():
            if name in running_names:
                continue
            result.append(
                InterpreterInfo(
                    name=name,
                    source='saved',
                    state='stopped',
                    python_path=saved_config.python_path,
                    cwd=saved_config.cwd,
                    has_startup_script=saved_config.startup_script is not None,
                    description=saved_config.description,
                    pid=None,
                    started_at=None,
                    uptime=None,
                )
            )

        # JetBrains discovered SDK entries
        listed_names = running_names | set(saved_configs.keys())
        for entry in self.state.jetbrains_sdks:
            if entry.name in listed_names:
                continue
            listed_names.add(entry.name)
            result.append(
                InterpreterInfo(
                    name=entry.name,
                    source='jetbrains-sdk',
                    state='stopped',
                    python_path=entry.python_path,
                    cwd=None,
                    has_startup_script=False,
                    description=entry.version,
                    pid=None,
                    started_at=None,
                    uptime=None,
                )
            )

        # JetBrains discovered run configs (console only)
        for run_config in self.state.jetbrains_runs:
            if run_config.name in listed_names:
                continue
            listed_names.add(run_config.name)
            result.append(
                InterpreterInfo(
                    name=run_config.name,
                    source='jetbrains-run',
                    state='stopped',
                    python_path=run_config.python_path,
                    cwd=run_config.cwd,
                    has_startup_script=run_config.script_name is not None,
                    description=f'from {pathlib.Path(run_config.xml_path).name}',
                    pid=None,
                    started_at=None,
                    uptime=None,
                )
            )

        return result


def _build_jetbrains_startup_script(config: JetBrainsRunConfig) -> str | None:
    """Build startup script from JetBrains SCRIPT_NAME + PARAMETERS."""
    if not config.script_name:
        return None

    script_path = pathlib.Path(config.script_name)
    if not script_path.exists():
        return None

    parts = ['import sys']

    # Set sys.argv to match what JetBrains would provide
    argv_items = [repr(str(script_path))]
    if config.parameters:
        import shlex

        argv_items.extend(repr(param) for param in shlex.split(config.parameters))
    parts.append(f'sys.argv = [{", ".join(argv_items)}]')

    # Read and exec the script
    parts.append(f'exec(__import__("pathlib").Path({str(script_path)!r}).read_text())')

    return '\n'.join(parts)


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
