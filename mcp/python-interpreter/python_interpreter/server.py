"""Python Interpreter MCP Server.

Persistent Python execution environment with multi-interpreter support.
Exposes tools via FastMCP (stdio) and an HTTP bridge (Unix socket) for heredoc usage.

Architecture:
    Claude Code ─[stdio]─> FastMCP server
    mcp-py-client ─[HTTP]─> FastAPI on /tmp/python-interpreter-{pid}.sock
    External interpreters ─[subprocess]─> driver.py (length-prefixed JSON)

Tools: execute, register_interpreter, stop_interpreter, list_interpreters

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

import fastapi
import mcp.server.fastmcp
import mcp.types
import uvicorn
from local_lib.utils import DualLogger

from python_interpreter.models import ExecuteRequest, InterpreterInfo
from python_interpreter.service import PythonInterpreterService, ServerState, SimpleLogger

__all__ = [
    'server',
    'main',
]


# Create FastMCP server with lifespan (defined below)
server: mcp.server.fastmcp.FastMCP

# Create FastAPI app for HTTP bridge
fastapi_app = fastapi.FastAPI(title='Python Interpreter HTTP Bridge')


def _format_interpreter_summary(service: PythonInterpreterService, max_shown: int = 5) -> str:
    """Format saved + discovered interpreter list for injection into tool descriptions."""
    saved = service.state.interpreter_registry.list_saved()
    running = service.state.interpreter_manager.get_interpreters()
    running_names = {name for name, _, _, _ in running}

    items: list[str] = []

    # Saved interpreters
    for name, config in saved.items():
        state = 'running' if name in running_names else 'stopped'
        desc = f' — {config.description}' if config.description else ''
        items.append(f'{name} ({state}, saved{desc})')

    # JetBrains SDK entries
    items.extend(
        f'{entry.name} (jetbrains-sdk)'
        for entry in service.state.jetbrains_sdks
        if entry.name not in running_names and entry.name not in saved
    )

    # JetBrains run configs
    items.extend(
        f'{rc.name} (jetbrains-run)'
        for rc in service.state.jetbrains_runs
        if rc.name not in running_names and rc.name not in saved
    )

    if not items:
        return ''

    shown = items[:max_shown]
    summary = ', '.join(shown)
    remaining = len(items) - max_shown
    if remaining > 0:
        summary += f' ... and {remaining} more'

    return f'**Available interpreters:** {summary}'


def _inject_interpreter_summary(base_docstring: str, summary: str) -> str:
    """Inject interpreter summary into tool description before Args."""
    if not summary:
        return base_docstring

    if 'Args:' in base_docstring:
        args_pos = base_docstring.find('Args:')
        before_args = base_docstring[:args_pos].rstrip()
        after_args = base_docstring[args_pos:]
        return f'{before_args}\n\n{summary}\n\n{after_args}'
    return f'{base_docstring.rstrip()}\n\n{summary}'


def register_tools(service: PythonInterpreterService) -> None:
    """Register service methods as MCP tools via closures."""
    # Build interpreter summary for docstring injection
    interpreter_summary = _format_interpreter_summary(service)

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

        Stopped interpreters auto-start on execute. To reset state, use stop_interpreter
        then execute again (the interpreter restarts fresh).

        The builtin interpreter auto-installs missing packages via uv on ModuleNotFoundError.
        External interpreters use whatever packages exist in their Python environment.

        Tip: To list defined variables: execute("print([x for x in dir() if not x.startswith('_')])")

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

    REGISTER_INTERPRETER_BASE = """Save an external Python interpreter configuration.

        Persists config to disk. Does NOT start the interpreter — it auto-starts
        on the next execute() call targeting it. If the interpreter is already
        running, the config is updated on disk but takes effect after restart
        (stop_interpreter + execute).

        python_path can be relative to the project directory (e.g., '.venv/bin/python').

        Args:
            name: Unique name for this interpreter
            python_path: Path to Python executable
            cwd: Working directory (defaults to project dir)
            env: Additional environment variables
            startup_script: Python code to run after starting (e.g., imports)
            description: Human-readable description"""

    @server.tool(
        description=_inject_interpreter_summary(REGISTER_INTERPRETER_BASE, interpreter_summary),
        annotations=mcp.types.ToolAnnotations(
            title='Register Interpreter',
            destructiveHint=False,
            idempotentHint=False,
            readOnlyHint=False,
            openWorldHint=False,
        ),
    )
    async def register_interpreter(
        name: str,
        python_path: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        startup_script: str | None = None,
        description: str | None = None,
    ) -> InterpreterInfo:
        logger = DualLogger(ctx)

        resolved_path = pathlib.Path(python_path)
        if not resolved_path.is_absolute():
            resolved_path = service.state.project_dir / resolved_path

        return await service.register_interpreter(
            name=name,
            python_path=resolved_path,
            logger=logger,
            cwd=pathlib.Path(cwd) if cwd else None,
            env=env,
            startup_script=startup_script,
            description=description,
        )

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
        """Stop an interpreter subprocess.

        All interpreters (including builtin) can be stopped. Stopped interpreters
        auto-restart on the next execute() call — this is the way to reset state
        (stop + execute = fresh scope).

        Set remove=True to permanently delete a saved config.

        Args:
            name: Name of the interpreter to stop
            remove: Permanently delete saved config (default: False)
        """
        logger = DualLogger(ctx)
        return await service.stop_interpreter(name, logger, remove=remove)

    LIST_INTERPRETERS_BASE = """List all interpreters (running, saved, and discovered).

        Returns name, source (builtin/saved/jetbrains-sdk/jetbrains-run),
        state (running/stopped), python_path, cwd, pid, uptime, and configuration details.
        Saved and JetBrains-discovered interpreters appear as 'stopped' when not running.
        Dead interpreters are automatically removed."""

    @server.tool(
        description=_inject_interpreter_summary(LIST_INTERPRETERS_BASE, interpreter_summary),
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
