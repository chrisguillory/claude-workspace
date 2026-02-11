"""Python Interpreter MCP Server.

Persistent Python execution environment with multi-interpreter support.
Exposes tools via FastMCP (stdio) and an HTTP bridge (Unix socket) for heredoc usage.

Architecture:
    Claude Code ─[stdio]─> FastMCP server
    mcp-py-client ─[HTTP]─> FastAPI on /tmp/python-interpreter-{pid}.sock
    External interpreters ─[subprocess]─> driver.py (length-prefixed JSON)

Tools: execute, reset, list_vars, get_session_info,
       add_interpreter, stop_interpreter, list_interpreters

Setup:
    uv tool install --editable mcp/python-interpreter
    claude mcp add --scope user python-interpreter -- mcp-py-server
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import sys
import traceback
import typing
from collections.abc import Mapping

import fastapi
import mcp.server.fastmcp
import mcp.types
import uvicorn
from local_lib.utils import DualLogger

from python_interpreter.manager import InterpreterConfig
from python_interpreter.models import ExecuteRequest, InterpreterInfo, SavedInterpreterConfig, SessionInfo
from python_interpreter.service import PythonInterpreterService, ServerState, SimpleLogger

__all__ = [
    'server',
    'main',
]


# Create FastMCP server with lifespan (defined below)
server: mcp.server.fastmcp.FastMCP

# Create FastAPI app for HTTP bridge
fastapi_app = fastapi.FastAPI(title='Python Interpreter HTTP Bridge')


def _format_saved_summary(
    saved: Mapping[str, SavedInterpreterConfig],
    running_names: set[str],
    max_shown: int = 5,
) -> str:
    """Format saved interpreter list for injection into tool descriptions."""
    if not saved:
        return ''

    items = []
    for name, config in list(saved.items())[:max_shown]:
        state = 'running' if name in running_names else 'stopped'
        desc = f' — {config.description}' if config.description else ''
        items.append(f'{name} ({state}{desc})')

    summary = ', '.join(items)
    remaining = len(saved) - max_shown
    if remaining > 0:
        summary += f' ... and {remaining} more'

    return f'**Saved interpreters:** {summary}'


def _inject_saved_interpreters(base_docstring: str, saved_summary: str) -> str:
    """Inject saved interpreter summary into tool description before Args."""
    if not saved_summary:
        return base_docstring

    if 'Args:' in base_docstring:
        args_pos = base_docstring.find('Args:')
        before_args = base_docstring[:args_pos].rstrip()
        after_args = base_docstring[args_pos:]
        return f'{before_args}\n\n{saved_summary}\n\n{after_args}'
    return f'{base_docstring.rstrip()}\n\n{saved_summary}'


def register_tools(service: PythonInterpreterService) -> None:
    """Register service methods as MCP tools via closures."""
    # Load saved interpreters for docstring injection
    saved_configs = service.state.interpreter_registry.list_saved()
    running = service.state.interpreter_manager.get_interpreters()
    running_names = {name for name, _, _, _ in running}
    saved_summary = _format_saved_summary(saved_configs, running_names)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Execute Python Code',
            destructiveHint=False,
            idempotentHint=False,
            readOnlyHint=False,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    async def execute(
        code: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        interpreter: str = 'builtin',
    ) -> str:
        """Execute Python code in persistent scope.

        Variables, imports, functions, and classes persist across calls. The last expression
        is auto-evaluated (no need to print). Returns stdout/stderr output, repr() of the
        last expression, or full tracebacks on error. Outputs >25,000 chars are truncated
        with full output saved to a temp file.

        The builtin interpreter auto-installs missing packages via uv on ModuleNotFoundError.
        External interpreters use whatever packages exist in their Python environment.

        IMPORTANT: For better readability in approval prompts, prefer the Bash client:
            mcp-py-client <<'PY'
            print("Hello")
            PY

        Args:
            code: Python code to execute
            interpreter: Interpreter name (defaults to 'builtin' with auto-install)
        """
        logger = DualLogger(ctx)
        return await service.execute(code, logger, interpreter)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Reset Python Scope',
            destructiveHint=True,
            idempotentHint=True,
            readOnlyHint=False,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    async def reset(ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any]) -> str:
        """Clear all variables, imports, and functions from the builtin interpreter scope.

        Destructive and cannot be undone. Returns count of items removed. Does not
        affect external interpreters - use stop_interpreter + add_interpreter to reset those.
        """
        logger = DualLogger(ctx)
        return await service.reset(logger)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='List Python Variables',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    async def list_vars(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        interpreter: str = 'builtin',
    ) -> str:
        """List all user-defined variables in persistent scope.

        Returns alphabetically sorted names of variables, functions, classes, and imports.
        Filters out Python builtins (names starting with '__').

        Args:
            interpreter: Interpreter name (defaults to 'builtin')
        """
        logger = DualLogger(ctx)
        return await service.list_vars(logger, interpreter)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Session Info',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def get_session_info(ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any]) -> SessionInfo:
        """Get comprehensive session and server metadata including session ID, paths, PID, start time, and uptime."""
        logger = DualLogger(ctx)
        return await service.get_session_info(logger)

    ADD_INTERPRETER_BASE = """Add and start an external Python interpreter.

        Creates a subprocess using a different Python executable (e.g., project venv).
        No auto-install — uses whatever packages are in that Python environment.
        python_path can be relative to the project directory (e.g., '.venv/bin/python').

        Set save=True to persist the configuration. Saved interpreters appear as
        'stopped' in list_interpreters after server restart and can be re-started
        by calling add_interpreter again with the same name and python_path.

        Args:
            name: Unique name for this interpreter
            python_path: Path to Python executable (absolute or relative to project dir)
            cwd: Working directory (defaults to project dir)
            env: Additional environment variables
            startup_script: Python code to run after starting (e.g., imports)
            save: Persist config to disk for reuse across server restarts
            description: Human-readable description (stored if save=True)"""

    @server.tool(
        description=_inject_saved_interpreters(ADD_INTERPRETER_BASE, saved_summary),
        annotations=mcp.types.ToolAnnotations(
            title='Add External Interpreter',
            destructiveHint=False,
            idempotentHint=False,
            readOnlyHint=False,
            openWorldHint=False,
        ),
    )
    async def add_interpreter(
        name: str,
        python_path: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        startup_script: str | None = None,
        save: bool = False,
        description: str | None = None,
    ) -> InterpreterInfo:
        logger = DualLogger(ctx)
        # Resolve relative python_path against project dir
        resolved_path = pathlib.Path(python_path)
        if not resolved_path.is_absolute():
            resolved_path = service.state.project_dir / resolved_path
        config = InterpreterConfig(
            name=name,
            python_path=resolved_path,
            cwd=pathlib.Path(cwd) if cwd else None,
            env=env,
            startup_script=startup_script,
        )
        return await service.add_interpreter(config, logger, save=save, description=description)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Stop Interpreter',
            destructiveHint=True,
            idempotentHint=False,
            readOnlyHint=False,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    async def stop_interpreter(
        name: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        remove: bool = False,
    ) -> str:
        """Stop an external interpreter subprocess.

        Saved interpreters transition to 'stopped' (config preserved, shown in
        list_interpreters). Set remove=True to permanently delete the saved config.
        Transient interpreters are always removed. Cannot stop the builtin interpreter.

        Args:
            name: Name of the interpreter to stop (cannot be 'builtin')
            remove: Permanently delete saved config (default: False)
        """
        logger = DualLogger(ctx)
        return await service.stop_interpreter(name, logger, remove=remove)

    LIST_INTERPRETERS_BASE = """List all interpreters (running and saved-but-stopped).

        Returns name, source (builtin/saved/transient), state (running/stopped),
        python_path, cwd, pid, uptime, and configuration details. Saved interpreters
        appear as 'stopped' when not running. Dead interpreters are automatically removed."""

    @server.tool(
        description=_inject_saved_interpreters(LIST_INTERPRETERS_BASE, saved_summary),
        annotations=mcp.types.ToolAnnotations(
            title='List Interpreters',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def list_interpreters(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
    ) -> list[InterpreterInfo]:
        logger = DualLogger(ctx)
        return await service.list_interpreters(logger)


@contextlib.asynccontextmanager
async def lifespan(
    server_instance: mcp.server.fastmcp.FastMCP,
) -> typing.AsyncIterator[None]:
    """Manage server lifecycle - initialization before requests, cleanup after shutdown."""
    state = ServerState.create()
    service = PythonInterpreterService(state)
    register_tools(service)

    # Store service on fastapi_app for HTTP endpoint access
    fastapi_app.state.service = service

    # Start FastAPI in background on Unix socket
    config = uvicorn.Config(fastapi_app, uds=state.socket_path.as_posix(), log_level='warning')
    uvicorn_server = uvicorn.Server(config)
    uvicorn_task = asyncio.create_task(uvicorn_server.serve())

    print('Server initialized', file=sys.stderr)
    print(f'  Output directory: {state.output_dir}', file=sys.stderr)
    print(f'  Unix socket: {state.socket_path}', file=sys.stderr)

    yield

    # Shutdown
    uvicorn_server.should_exit = True
    uvicorn_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await uvicorn_task

    print('Shutting down interpreters...', file=sys.stderr)
    state.interpreter_manager.shutdown_all()
    state.temp_dir.cleanup()
    if state.socket_path.exists():
        state.socket_path.unlink()
    print('Server cleanup complete', file=sys.stderr)


# Initialize FastMCP with lifespan
server = mcp.server.fastmcp.FastMCP('python-interpreter', lifespan=lifespan)


@fastapi_app.exception_handler(Exception)
async def global_exception_handler(request: fastapi.Request, exc: Exception) -> fastapi.responses.JSONResponse:
    """Global exception handler - returns all unhandled exceptions with full traceback."""
    tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    return fastapi.responses.JSONResponse(
        status_code=500,
        content={
            'detail': f'{type(exc).__name__}: {str(exc)}',
            'traceback': tb_str,
        },
    )


# FastAPI dependency functions
def get_interpreter_service(request: fastapi.Request) -> PythonInterpreterService:
    """Retrieve service from app.state."""
    service: PythonInterpreterService = request.app.state.service
    return service


def get_simple_logger() -> SimpleLogger:
    """Create logger instance."""
    return SimpleLogger()


@fastapi_app.post('/execute')
async def http_execute(
    request: ExecuteRequest,
    service: PythonInterpreterService = fastapi.Depends(get_interpreter_service),
    logger: SimpleLogger = fastapi.Depends(get_simple_logger),
) -> dict[str, str]:
    """HTTP endpoint for executing Python code.

    This allows heredoc syntax via mcp-py-client:
        mcp-py-client <<'PY'
        import pandas as pd
        print(pd.__version__)
        PY

    The client auto-discovers this endpoint via Unix socket.
    """
    result = await service.execute(request.code, logger, request.interpreter)
    return {'result': result}


def main() -> None:
    """Main entry point for the Python Interpreter MCP server."""
    print('Starting Python Interpreter MCP server', file=sys.stderr)
    server.run()


if __name__ == '__main__':
    main()
