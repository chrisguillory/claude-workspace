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

import fastapi
import mcp.server.fastmcp
import mcp.types
import uvicorn
from local_lib.utils import DualLogger

from python_interpreter.manager import InterpreterConfig
from python_interpreter.models import ExecuteRequest, InterpreterInfo, SessionInfo
from python_interpreter.service import PythonInterpreterService, ServerState, SimpleLogger

__all__ = [
    'server',
    'main',
]


# Create FastMCP server with lifespan (defined below)
server: mcp.server.fastmcp.FastMCP

# Create FastAPI app for HTTP bridge
fastapi_app = fastapi.FastAPI(title='Python Interpreter HTTP Bridge')


def register_tools(service: PythonInterpreterService) -> None:
    """Register service methods as MCP tools via closures."""

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

    @server.tool(
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
    ) -> InterpreterInfo:
        """Add and start an external Python interpreter.

        Creates a subprocess using a different Python executable (e.g., project venv).
        No auto-install - uses whatever packages are in that Python environment.

        Args:
            name: Unique name for this interpreter
            python_path: Path to Python executable
            cwd: Working directory (defaults to project dir)
            env: Additional environment variables
            startup_script: Python code to run after starting (e.g., imports)
        """
        logger = DualLogger(ctx)
        config = InterpreterConfig(
            name=name,
            python_path=pathlib.Path(python_path),
            cwd=pathlib.Path(cwd) if cwd else None,
            env=env,
            startup_script=startup_script,
        )
        return await service.add_interpreter(config, logger)

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
    ) -> str:
        """Stop an external interpreter subprocess.

        Kills the subprocess and removes it from the interpreter list. All state in that
        interpreter is lost. Cannot stop the builtin interpreter - it is always running.

        Args:
            name: Name of the interpreter to stop (cannot be 'builtin')
        """
        logger = DualLogger(ctx)
        return await service.stop_interpreter(name, logger)

    @server.tool(
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
        """List all interpreters (builtin and external).

        Returns name, type, python_path, cwd, pid, started_at, uptime, and whether
        a startup script was configured. Dead interpreters are automatically removed.
        """
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
