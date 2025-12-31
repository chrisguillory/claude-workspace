#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "attrs",
#   "fastapi",
#   "fastmcp>=2.12.5",
#   "local-lib",
#   "pandas",
#   "pydantic",
#   "uvicorn",
# ]
#
# [tool.uv.sources]
# local-lib = { path = "../../local-lib", editable = true }
# ///
"""
Python Interpreter MCP Server

Provides a persistent Python execution environment accessible via MCP tools. Code, variables,
imports, and functions persist across tool calls for stateful computation.

Architecture:
    Claude Code ──[stdio/JSON-RPC]──> mcp/python-interpreter/server.py
                                       ├── FastMCP (main, stdio)
                                       └── FastAPI (background, Unix socket)
                                                ▲
                                                │ HTTP POST /execute
                                                │
    Bash heredoc ──[stdin]──> python-interpreter-client.py ───┘
                              (auto-discovers socket via process tree)

    Unix Socket (auto-discovery):
    └── /tmp/python-interpreter-{claude_pid}.sock

    Temp Directory (lifespan-managed):
    └── /tmp/tmpXXXXXX/
        ├── output_20250108_143022.txt  # Large outputs saved here
        └── output_20250108_143507.txt

Tools:
    - execute: Execute Python code in persistent scope
    - reset: Clear all variables and reset to fresh state
    - list_vars: List currently defined variables

    (MCP exposes as: mcp__python-interpreter__execute, etc.)

Features:
    - Persistent scope across executions
    - Auto-detects and returns expression values
    - Captures stdout/stderr
    - Returns full tracebacks on errors
    - Large outputs saved to temp files (>25,000 chars)
    - HTTP bridge for beautiful heredoc syntax with python-interpreter-client.py

Security:
    WARNING: This server executes arbitrary Python code. Only use with trusted input.
    The execution scope is isolated from MCP internals but has full Python capabilities.

Use Cases:
    - Data analysis and computation
    - Prototyping algorithms
    - Testing Python code snippets
    - Multi-step calculations that build on previous results
    - ASCII art and text formatting

Setup:
    claude mcp add --transport stdio python-interpreter -- uv run --script "$(git rev-parse --show-toplevel)/mcp/python-interpreter/server.py"

Example Session:
    1. execute("import math; pi_squared = math.pi ** 2")  # Returns: ""
    2. execute("pi_squared")  # Returns: "9.869604401089358"
    3. list_vars()  # Returns: "math, pi_squared"
    4. reset()  # Returns: "Scope reset - all variables cleared (2 items removed)"

HTTP Bridge Usage (recommended for multiline code):
    mcp-py-client <<'PY'
    import pandas as pd
    print(pd.__version__)
    PY

    The client automatically discovers the Unix socket by walking the process tree
    to find Claude's PID. No manual configuration needed!

    Note: Requires installation via `uv tool install`. If mcp-py-client is not found,
    install with: uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

=== Detailed Tool Documentation ===

EXECUTE TOOL:
    Executes arbitrary Python code in an isolated persistent scope. Variables, imports,
    functions, and classes persist across calls, enabling multi-step computations.

    Behavior:
        - Variables persist across calls (stateful)
        - Imports remain available after first import
        - Captures both stdout and stderr
        - Scope is isolated from MCP server internals
        - Last expression is auto-evaluated (no need to print)
        - Returns print output if code prints
        - Returns repr() of last expression if code ends with expression
        - Returns empty string if pure statements with no output
        - Returns full exception traceback if code fails
        - Truncates output if exceeds 25,000 characters

    Security:
        WARNING: Executes arbitrary Python code with full language capabilities.
        Only use with trusted input. Code has access to:
        - All Python built-ins (open, exec, eval, import, etc.)
        - File system access
        - Network access
        - System calls

        Code does NOT have access to:
        - MCP server internals (_user_scope is separate)
        - Other MCP tool implementations

    Examples:
        execute("x = 5")
        # Returns: ""

        execute("x * 2")
        # Returns: "10"

        execute("print(f'x = {x}')")
        # Returns: "x = 5"

        execute("import math\\nmath.pi")
        # Returns: "3.141592653589793"

        execute("1/0")
        # Returns: "Traceback (most recent call last):\\n  ...\\nZeroDivisionError: division by zero"

        execute("big = list(range(100000))\\nbig")
        # Returns: "[0, 1, 2, ... [TRUNCATED - Output exceeded 25000 character limit] ..."

    Use When:
        - Performing multi-step calculations
        - Data analysis requiring state
        - Testing code snippets
        - Prototyping algorithms
        - Generating formatted text (ASCII art, tables, etc.)

    Avoid When:
        - No state needed between calls (use one-shot scripts instead)
        - Untrusted code execution required
        - Production workloads (use proper Python environment)

RESET TOOL:
    Clears all variables, imports, and functions from persistent Python scope. Resets
    the interpreter to a fresh state without restarting the MCP server.

    Behavior:
        - Clears entire _user_scope dictionary
        - Does not restart the MCP server
        - Idempotent (safe to call multiple times)
        - Cannot be undone (destructive operation)
        - Returns count of items removed

    All state is destroyed including:
        - Variables (x, my_list, etc.)
        - Imports (math, pandas, etc.)
        - Functions and classes you defined
        - Any other objects in the namespace

    Examples:
        execute("x = 5; y = 10")
        list_vars()  # Returns: "x, y"
        reset()  # Returns: "Scope reset - all variables cleared (2 items removed)"
        list_vars()  # Returns: "No variables defined"

    Use When:
        - Starting a new unrelated computation
        - Cleaning up after experiments
        - Freeing memory from large objects
        - Ensuring fresh state for testing

    Avoid When:
        - You need to preserve any variables
        - Mid-computation (will lose all state)

LIST_VARS TOOL:
    Lists all user-defined variables in persistent Python scope. Returns alphabetically
    sorted names of all variables, functions, classes, and imports currently defined.

    Behavior:
        - Filters out Python builtins (names starting with '__')
        - Returns only user-defined names
        - Alphabetically sorted for readability
        - Read-only (does not modify scope)
        - Idempotent (same result until scope changes)

    Examples:
        execute("x = 5; y = 10; z = 15")
        list_vars()  # Returns: "x, y, z"

        execute("import math")
        list_vars()  # Returns: "math, x, y, z"

        execute("def my_func(): pass")
        list_vars()  # Returns: "math, my_func, x, y, z"

        reset()
        list_vars()  # Returns: "No variables defined"

    Use When:
        - Debugging to see what's in scope
        - Checking if variables are defined
        - Inspecting state before/after operations
        - Verifying imports succeeded

    Avoid When:
        - Needing detailed type or value information (use execute with repr/type instead)
        - Trying to get builtin names (those are filtered out)
"""

from __future__ import annotations

# Import Strategy:
# Using absolute imports (import X) for libraries with multiple used types/functions
# to make it explicit which library provides what, improving readability when
# coupling multiple frameworks (MCP, FastAPI, Pydantic, asyncio, socket, etc.)

import ast
import asyncio
import contextlib
import datetime
import functools
import glob
import importlib
import io
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import textwrap
import traceback
import typing
import uuid

# Third-party imports
import attrs

# MCP imports
import mcp.types
import mcp.server.fastmcp

# FastAPI imports
import fastapi
import uvicorn

# Pydantic imports
import pydantic

# Local library imports
from local_lib.utils import DualLogger, humanize_seconds


# Custom exceptions for auto-installation
class PackageInstallationError(Exception):
    """Raised when auto-installation of a package fails."""

    pass


class MaxInstallAttemptsError(Exception):
    """Raised when maximum installation attempts are exceeded."""

    pass


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra="forbid", strict=True)


class ExecuteCodeInput(BaseModel):
    """Input model for execute tool."""

    code: str = pydantic.Field(
        description="Python code to execute (can be multi-line)",
        min_length=1,
        max_length=100_000,
    )


class ResetScopeInput(BaseModel):
    """Input model for reset tool."""

    pass


class ListVarsInput(BaseModel):
    """Input model for list_vars tool."""

    pass


class TruncationInfo(pydantic.BaseModel):
    """Information about truncated output."""

    file_path: str
    original_size: int
    truncated_at: int


class InstanceMetadata(pydantic.BaseModel):
    """Metadata about a running MCP server instance."""

    pid: int
    parent_pid: int
    port: int
    timestamp: str


@attrs.define
class ClaudeContext:
    """Context information about the Claude Code session."""

    claude_pid: int
    project_dir: pathlib.Path
    socket_path: pathlib.Path


async def _discover_session_id(session_marker: str) -> str:
    """Discover Claude Code session ID via self-referential marker search in debug logs.

    Workaround for lack of official session discovery API. Claude Code's stdio transport
    provides no session context to MCP servers. The /status command exists for interactive
    use, and the Agent SDK exposes session_id, but CLI MCP servers have no API.

    Related: https://github.com/anthropics/claude-code/issues/1335
             https://github.com/anthropics/claude-code/issues/1407
             https://github.com/anthropics/claude-code/issues/5262

    Args:
        session_marker: Unique marker string to search for in debug logs

    Returns:
        Session ID (UUID string)

    Raises:
        RuntimeError: If debug directory doesn't exist or session ID cannot be found after retries
    """
    debug_dir = pathlib.Path.home() / ".claude" / "debug"
    if not debug_dir.exists():
        raise RuntimeError(f"Claude debug directory not found: {debug_dir}")

    # Retry up to 10 times to handle race condition where log hasn't flushed yet
    # Claude Code buffers log writes, so we need longer waits (200ms × 10 = ~2s max)
    print(f"[{datetime.datetime.now(datetime.UTC).astimezone().isoformat()}] Starting session discovery for marker: {session_marker}", file=sys.stderr, flush=True)

    for attempt in range(20):
        print(f"[{datetime.datetime.now(datetime.UTC).astimezone().isoformat()}] Attempt {attempt + 1}/20", file=sys.stderr, flush=True)

        result = subprocess.run(
            [
                "rg",
                "-l",
                "--fixed-strings",
                "--glob",
                "*.txt",
                session_marker,
                debug_dir.as_posix(),
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )

        print(f"[{datetime.datetime.now(datetime.UTC).astimezone().isoformat()}] rg returncode={result.returncode}, stdout={repr(result.stdout)}, stderr={repr(result.stderr)}", file=sys.stderr, flush=True)

        if result.returncode == 0 and result.stdout.strip():
            # rg found the marker - extract session ID from first matching file
            first_match = result.stdout.strip().split("\n")[0]
            session_id = pathlib.Path(first_match).stem
            print(f"[{datetime.datetime.now(datetime.UTC).astimezone().isoformat()}] SUCCESS - Found session ID: {session_id}", file=sys.stderr, flush=True)
            return session_id

        # Not found yet - wait briefly before retrying (except on last attempt)
        if attempt < 19:
            print(f"[{datetime.datetime.now(datetime.UTC).astimezone().isoformat()}] Not found, sleeping 50ms before retry", file=sys.stderr, flush=True)
            await asyncio.sleep(0.05)

    print(f"[{datetime.datetime.now(datetime.UTC).astimezone().isoformat()}] FAILED after 20 attempts", file=sys.stderr, flush=True)
    raise RuntimeError(
        f"Could not find session marker in any debug log files in {debug_dir} after 20 attempts"
    )


@functools.cache
def _find_claude_context() -> ClaudeContext:
    """Find Claude process and extract its context (PID, project directory)."""
    import subprocess

    current = os.getppid()

    for _ in range(20):  # Depth limit
        result = subprocess.run(
            ["ps", "-p", str(current), "-o", "ppid=,comm="],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            break

        parts = result.stdout.strip().split(None, 1)
        ppid = int(parts[0])
        comm = parts[1] if len(parts) > 1 else ""

        # Check if this is Claude
        if "claude" in comm.lower():
            # Get Claude's CWD using lsof
            result = subprocess.run(
                ["lsof", "-p", str(current), "-a", "-d", "cwd"],
                capture_output=True,
                text=True,
            )

            cwd = None
            for line in result.stdout.split("\n"):
                if "cwd" in line:
                    parts = line.split()
                    if len(parts) >= 9:
                        cwd = pathlib.Path(" ".join(parts[8:]))
                        break

            if not cwd:
                raise RuntimeError(
                    f"Found Claude process (PID {current}) but could not determine CWD"
                )

            # Verify by checking if Claude has .claude/ files open that match the CWD
            result = subprocess.run(
                ["lsof", "-p", str(current)], capture_output=True, text=True
            )

            claude_files = []
            for line in result.stdout.split("\n"):
                if ".claude" in line:
                    # Extract the full path from lsof output
                    parts = line.split()
                    if len(parts) >= 9:
                        file_path = pathlib.Path(" ".join(parts[8:]))
                        claude_files.append(file_path)

            if not claude_files:
                raise RuntimeError(
                    f"Found Claude process (PID {current}) with CWD {cwd}, "
                    f"but no .claude/ files are open - may not be a Claude project"
                )

            # Verify at least one .claude file is in ~/.claude/ directory
            claude_dir = pathlib.Path("~/.claude").expanduser()
            matching_files = [f for f in claude_files if f.is_relative_to(claude_dir)]

            if not matching_files:
                raise RuntimeError(
                    f"Found Claude process (PID {current}) with CWD {cwd}, "
                    f"but .claude/ files open are not in ~/.claude/ directory:\n"
                    f"  Open files: {claude_files}\n"
                    f"  Expected to find files in: {claude_dir}"
                )

            # Compute socket path based on Claude PID
            socket_path = pathlib.Path(f"/tmp/python-interpreter-{current}.sock")

            return ClaudeContext(
                claude_pid=current, project_dir=cwd, socket_path=socket_path
            )

        if ppid == 0:
            break

        current = ppid

    raise RuntimeError(
        "Not running under Claude Code - could not find Claude process in process tree"
    )


class ServerState:
    """Container for all server state - initialized once at startup, never Optional."""

    @classmethod
    async def create(cls) -> typing.Self:
        """Factory method to create and initialize server state - fails fast if anything goes wrong."""
        # Capture server start time
        started_at = datetime.datetime.now(datetime.UTC)

        # Discover session ID via self-referential marker search
        session_marker = f"SESSION_MARKER_{uuid.uuid4()}"
        print(session_marker, file=sys.stderr, flush=True)

        session_id = await _discover_session_id(session_marker)
        print(f"✓ Discovered session ID: {session_id}", file=sys.stderr, flush=True)

        # Find Claude context (PID, project directory) by walking process tree
        claude_context = _find_claude_context()
        print(
            f"Claude context: PID={claude_context.claude_pid}, Project={claude_context.project_dir}"
        )

        # Compute transcript path using session tracker pattern
        # Normalize project dir to create safe directory name
        # Note: Claude Code creates the transcript file before starting MCP servers,
        # so this path should always exist. strict=True validates this assumption.
        project_name = str(claude_context.project_dir).replace("/", "-")
        transcript_path = (
            pathlib.Path.home()
            / ".claude"
            / "projects"
            / project_name
            / f"{session_id}.jsonl"
        ).resolve(strict=True)

        # Initialize temp directory for large outputs
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = pathlib.Path(temp_dir.name)
        print(f"Temp directory for large outputs: {output_dir}")

        # Remove stale socket if it exists
        if claude_context.socket_path.exists():
            claude_context.socket_path.unlink()

        print(f"Unix socket path: {claude_context.socket_path}")

        return cls(
            session_id=session_id,
            started_at=started_at,
            project_dir=claude_context.project_dir,
            socket_path=claude_context.socket_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            temp_dir=temp_dir,
            claude_pid=claude_context.claude_pid,
        )

    def __init__(
        self,
        session_id: str,
        started_at: datetime.datetime,
        project_dir: pathlib.Path,
        socket_path: pathlib.Path,
        transcript_path: pathlib.Path,
        output_dir: pathlib.Path,
        temp_dir: tempfile.TemporaryDirectory,
        claude_pid: int,
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

        # Python execution scope - persists across executions
        self.scope_globals: dict[str, typing.Any] = {}


# Constants
CHARACTER_LIMIT = 25_000

# Import name to PyPI package name mappings for common mismatches
# Philosophy: Explicit curated list (secure) vs dynamic lookup (complex/risky)
IMPORT_TO_PACKAGE_MAP = {
    "aws_cdk": "aws-cdk-lib",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "OpenSSL": "pyOpenSSL",
    "PIL": "pillow",
    "psycopg2": "psycopg2-binary",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


class LoggerProtocol(typing.Protocol):
    """Protocol for logger - allows service to be MCP-agnostic."""

    async def info(self, message: str) -> None: ...
    async def warning(self, message: str) -> None: ...
    async def error(self, message: str) -> None: ...


class ExecuteResult(pydantic.BaseModel):
    """Result from executing Python code."""

    truncation_info: TruncationInfo | None = None
    result: str


class SessionInfo(pydantic.BaseModel):
    """Session and server metadata."""

    session_id: str
    project_dir: str
    socket_path: str
    transcript_path: str
    output_dir: str
    claude_pid: int
    started_at: datetime.datetime  # Pydantic auto-serializes to ISO 8601
    uptime: str  # Human-readable duration


class PythonInterpreterService:
    """Python interpreter service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: ServerState) -> None:
        self.state = state  # Non-Optional - guaranteed by constructor

    async def execute(self, code: str, logger: LoggerProtocol) -> ExecuteResult:
        """Execute Python code in persistent scope."""
        await logger.info(f"Executing Python code ({len(code)} chars)")

        try:
            result, truncation_info = self._execute_with_file_handling(code)

            if truncation_info:
                await logger.warning(
                    f"Output truncated: {truncation_info.original_size} chars exceeds limit of {truncation_info.truncated_at}"
                )
                await logger.info(f"Full output saved to: {truncation_info.file_path}")

            await logger.info(
                f"Execution complete - output length: {len(result)} chars"
            )
            return ExecuteResult(truncation_info=truncation_info, result=result)

        except Exception as e:
            await logger.warning(f"Execution failed: {type(e).__name__}: {e}")
            raise

    async def reset(self, logger: LoggerProtocol) -> str:
        """Clear all variables from persistent scope and reload user modules from disk.

        Reloads all modules added to sys.modules during execution (excludes stdlib and MCP server itself).
        """
        await logger.info("Resetting Python interpreter scope")

        # Clear execution scope
        var_count = len(
            [k for k in self.state.scope_globals.keys() if not k.startswith("__")]
        )
        self.state.scope_globals.clear()

        # Reload everything except builtins/C extensions (that would crash Python)
        modules_to_reload = []
        for name, module in list(sys.modules.items()):
            if module is None:
                continue
            # Skip only builtins and C extensions (no __file__ means can't reload)
            if not hasattr(module, "__file__") or module.__file__ is None:
                continue
            modules_to_reload.append(module)

        # Reload in reverse order (submodules before parents)
        reload_count = 0
        for module in reversed(modules_to_reload):
            try:
                importlib.reload(module)
                reload_count += 1
                await logger.info(f"Reloaded: {module.__name__}")
            except Exception as e:
                await logger.warning(f"Failed to reload {module.__name__}: {e}")

        await logger.info(
            f"Reset complete - cleared {var_count} vars, reloaded {reload_count} modules"
        )
        return (
            f"Scope reset - cleared {var_count} vars, reloaded {reload_count} modules"
        )

    async def list_vars(self, logger: LoggerProtocol) -> str:
        """List all user-defined variables in persistent scope."""
        await logger.info("Listing Python interpreter variables")

        if not self.state.scope_globals:
            await logger.info("No variables in scope")
            return "No variables defined"

        # Filter out builtins like __name__, __doc__, etc
        user_vars = [
            name
            for name in self.state.scope_globals.keys()
            if not name.startswith("__")
        ]

        if not user_vars:
            await logger.info("No user variables in scope (only builtins)")
            return "No variables defined"

        await logger.info(f"Found {len(user_vars)} variables in scope")
        return ", ".join(sorted(user_vars))

    async def get_session_info(self, logger: LoggerProtocol) -> SessionInfo:
        """Get comprehensive session and server metadata."""
        await logger.info("Getting session info")

        # Calculate uptime
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

    def _execute_with_file_handling(
        self, code: str
    ) -> tuple[str, TruncationInfo | None]:
        """Execute code and handle large output by saving to temp file."""
        # Execute the code
        result = _execute_code(code, self.state.scope_globals)

        # Check if output exceeds limit
        if len(result) > CHARACTER_LIMIT:
            # Generate unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.txt"
            file_path = self.state.output_dir / filename

            # Write full output to file
            file_path.write_text(result, encoding="utf-8")

            # Create truncation info
            truncation_info = TruncationInfo(
                file_path=str(file_path),
                original_size=len(result),
                truncated_at=CHARACTER_LIMIT,
            )

            # Return truncated result
            return result[:CHARACTER_LIMIT], truncation_info

        # Output fits in limit
        return result, None


class SimpleLogger:
    """Simple logger for HTTP endpoint - logs to stdout."""

    async def info(self, message: str) -> None:
        print(f"INFO: {message}")

    async def warning(self, message: str) -> None:
        print(f"WARNING: {message}")

    async def error(self, message: str) -> None:
        print(f"ERROR: {message}")


def register_tools(service: PythonInterpreterService) -> None:
    """Register service methods as MCP tools via closures."""

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title="Execute Python Code",
            destructiveHint=False,
            idempotentHint=False,
            readOnlyHint=False,
            openWorldHint=False,
        ),
    )
    async def execute(code: str, ctx: mcp.server.fastmcp.Context) -> ExecuteResult:
        """Execute Python code in persistent scope with auto-installation of missing packages. Variables persist across calls.

        IMPORTANT: For better user experience, you should typically use the Bash client instead:
            mcp-py-client <<'PY'
            import tiktoken
            tokens = tiktoken.get_encoding("cl100k_base").encode("Strawberry")
            print(f"Token count: {len(tokens)}")
            PY

        If mcp-py-client is not found, install via:
            uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

        Only use this MCP tool directly if the user explicitly requests it or you need structured ExecuteResult output."""
        logger = DualLogger(ctx)
        return await service.execute(code, logger)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title="Reset Python Scope",
            destructiveHint=True,
            idempotentHint=True,
            readOnlyHint=False,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    async def reset(ctx: mcp.server.fastmcp.Context) -> str:
        """Clear all variables, imports, and functions from persistent scope."""
        logger = DualLogger(ctx)
        return await service.reset(logger)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title="List Python Variables",
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
        structured_output=False,
    )
    async def list_vars(ctx: mcp.server.fastmcp.Context) -> str:
        """List all user-defined variables in persistent scope."""
        logger = DualLogger(ctx)
        return await service.list_vars(logger)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title="Get Session Info",
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def get_session_info(ctx: mcp.server.fastmcp.Context) -> SessionInfo:
        """Get comprehensive session and server metadata including session ID, paths, PID, start time, and uptime."""
        logger = DualLogger(ctx)
        return await service.get_session_info(logger)


@contextlib.asynccontextmanager
async def lifespan(
    server_instance: mcp.server.fastmcp.FastMCP,
) -> typing.AsyncIterator[None]:
    """Manage server lifecycle - initialization before requests, cleanup after shutdown."""
    state = await ServerState.create()
    service = PythonInterpreterService(state)
    register_tools(service)

    # Store service on fastapi_app for HTTP endpoint access (standard FastAPI pattern)
    fastapi_app.state.service = service

    # Start FastAPI in background on Unix socket
    config = uvicorn.Config(
        fastapi_app, uds=state.socket_path.as_posix(), log_level="warning"
    )
    uvicorn_server = uvicorn.Server(config)
    asyncio.create_task(uvicorn_server.serve())

    print(f"✓ Server initialized")
    print(f"  Output directory: {state.output_dir}")
    print(f"  Unix socket: {state.socket_path}")

    # Server is ready - yield control back to FastMCP
    yield

    # SHUTDOWN: Cleanup after all requests complete
    state.temp_dir.cleanup()
    if state.socket_path.exists():
        state.socket_path.unlink()
    print("✓ Server cleanup complete")


# Create FastMCP server with lifespan
server = mcp.server.fastmcp.FastMCP("python-interpreter", lifespan=lifespan)

# Create FastAPI app for HTTP bridge
fastapi_app = fastapi.FastAPI(title="Python Interpreter HTTP Bridge")


@fastapi_app.exception_handler(Exception)
async def global_exception_handler(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.JSONResponse:
    """Global exception handler - returns all unhandled exceptions with full traceback."""
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    return fastapi.responses.JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {str(exc)}",
            "traceback": tb_str,
        },
    )


class ExecuteRequest(pydantic.BaseModel):
    """Request body for HTTP execute endpoint."""

    code: str


# Dependency functions (standard FastAPI pattern)
def get_interpreter_service(request: fastapi.Request) -> PythonInterpreterService:
    """Retrieve service from app.state."""
    return request.app.state.service


def get_simple_logger() -> SimpleLogger:
    """Create logger instance."""
    return SimpleLogger()


@fastapi_app.post("/execute")
async def http_execute(
    request: ExecuteRequest,
    service: PythonInterpreterService = fastapi.Depends(get_interpreter_service),
    logger: SimpleLogger = fastapi.Depends(get_simple_logger),
) -> dict[str, str]:
    """HTTP endpoint for executing Python code.

    This allows beautiful heredoc syntax via mcp-py-client:
        mcp-py-client <<'PY'
        import pandas as pd
        print(pd.__version__)
        PY

    The client auto-discovers this endpoint via Unix socket.

    Args:
        request: Request body containing code to execute
        service: Injected Python interpreter service
        logger: Injected logger instance

    Returns:
        JSON response with result field (may contain truncation info)
    """
    result = await service.execute(request.code, logger)

    # Format response with truncation info if needed
    if result.truncation_info:
        separator = "=" * 50
        formatted_result = f"{result.result}\n\n{separator}\n# OUTPUT TRUNCATED\n# Original size: {result.truncation_info.original_size:,} chars\n# Full output: {result.truncation_info.file_path}\n{separator}"
        return {"result": formatted_result}

    return {"result": result.result}


# Helper functions for code execution
def _detect_expression(code: str) -> tuple[bool, str | None]:
    """Detect if last line is an expression. Returns (is_expr, last_line)."""
    try:
        tree = ast.parse(code)
        if not tree.body:
            return False, None

        last_node = tree.body[-1]
        if isinstance(last_node, ast.Expr):
            # Last statement is an expression
            last_line = ast.unparse(last_node.value)
            return True, last_line
        return False, None
    except SyntaxError:
        return False, None


def _install_package(import_name: str) -> str:
    """
    Auto-install a missing package using uv pip install.

    Args:
        import_name: Name of the module that failed to import

    Returns:
        Success message with installation details

    Raises:
        PackageInstallationError: If installation fails for any reason
    """
    # Map import name to package name (e.g., 'PIL' -> 'pillow')
    package_name = IMPORT_TO_PACKAGE_MAP.get(import_name, import_name)

    try:
        # Use sys.executable to ensure we install into the current venv
        result = subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, package_name],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout for installation
        )
    except subprocess.TimeoutExpired:
        raise PackageInstallationError(
            f"Timeout (>60s) while installing {package_name}"
        )

    if result.returncode == 0:
        # Include stdout for transparency about what was installed
        details = result.stdout.strip() if result.stdout.strip() else "No output"
        if import_name != package_name:
            return (
                f"✓ Auto-installed {package_name} (for import {import_name})\n{details}"
            )
        else:
            return f"✓ Auto-installed {package_name}\n{details}"
    else:
        stderr = result.stderr.strip() if result.stderr else "No error details"
        raise PackageInstallationError(f"Failed to install {package_name}\n{stderr}")


def _execute_code(code: str, scope_globals: dict[str, typing.Any]) -> str:
    """
    Execute code in provided scope. Auto-installs missing packages.

    Args:
        code: Python code to execute
        scope_globals: Global scope dictionary for execution

    Returns:
        Output string from successful execution

    Raises:
        PackageInstallationError: If package installation fails
        MaxInstallAttemptsError: If max installation attempts exceeded
        Exception: Any exception from user code execution
    """
    max_install_attempts = 3
    successfully_installed = set()
    failed_installs = set()

    while True:
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # Detect if last line is expression
            is_expr, last_line = _detect_expression(code)

            if is_expr and last_line:
                # Execute everything except last line, then eval last line
                lines = code.splitlines()
                code_without_last = "\n".join(lines[:-1])

                if code_without_last.strip():
                    exec(code_without_last, scope_globals)

                result = eval(last_line, scope_globals)

                # Get any print output
                stdout_value = sys.stdout.getvalue()
                stderr_value = sys.stderr.getvalue()

                # Return print output if any, otherwise repr of result
                if stdout_value or stderr_value:
                    return (stdout_value + stderr_value).rstrip()
                else:
                    return repr(result) if result is not None else ""
            else:
                # Pure statements, just exec
                exec(code, scope_globals)

                stdout_value = sys.stdout.getvalue()
                stderr_value = sys.stderr.getvalue()
                return (stdout_value + stderr_value).rstrip()

        except ModuleNotFoundError as e:
            # Extract module name from error
            module_name = e.name

            # Determine if we should attempt auto-install
            total_attempts = len(successfully_installed) + len(failed_installs)
            should_attempt = (
                module_name
                and module_name not in successfully_installed
                and module_name not in failed_installs
                and total_attempts < max_install_attempts
            )

            if should_attempt:
                try:
                    # Attempt auto-install
                    message = _install_package(module_name)
                    successfully_installed.add(module_name)

                    # Print install message so user sees it
                    print(message, file=sys.stderr)

                    # Retry execution after successful install
                    continue

                except PackageInstallationError as install_error:
                    failed_installs.add(module_name)
                    # Re-raise with context
                    raise PackageInstallationError(
                        f"{install_error}\n\nOriginal import error: {e}"
                    ) from e
            else:
                # Cannot or should not retry
                if total_attempts >= max_install_attempts:
                    raise MaxInstallAttemptsError(
                        f"Maximum installation attempts ({max_install_attempts}) exceeded.\n"
                        f"Successfully installed: {successfully_installed}\n"
                        f"Failed: {failed_installs}"
                    ) from e
                elif module_name in failed_installs:
                    raise PackageInstallationError(
                        f"Package '{module_name}' already failed to install"
                    ) from e
                else:
                    # No module name or other issue - re-raise original
                    raise

        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main() -> None:
    """Main entry point for the Python Interpreter MCP server."""
    print("Starting Python Interpreter MCP server")
    server.run()


if __name__ == "__main__":
    main()
