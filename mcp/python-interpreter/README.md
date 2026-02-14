# Python Interpreter MCP Server

Persistent Python execution environment for Claude Code with automatic package installation and stateful computation.

## Installation

### Quick Start (Recommended)

Install the MCP server and client globally:

```bash
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter
```

This installs two commands to `~/.local/bin/`:
- `mcp-py-server` - The MCP server (used by Claude Code)
- `mcp-py-client` - The CLI client (for readable code execution)

> **Note:** Ensure `~/.local/bin` is in your PATH. Add to `~/.zshrc` or `~/.bashrc`:
> ```bash
> export PATH="$HOME/.local/bin:$PATH"
> ```

### Configure Claude Code

```bash
claude mcp add --scope user python-interpreter -- mcp-py-server
```

This adds the following to `~/.claude.json`:

```json
"python-interpreter": {
  "type": "stdio",
  "command": "mcp-py-server",
  "args": []
}
```

### Auto-approve Client Commands

Add to `.claude/settings.local.json` to skip permission prompts for the client:

```json
{
  "permissions": {
    "allow": [
      "Bash(mcp-py-client:*)"
    ]
  }
}
```

### Check Version

```bash
uv tool list | grep python-interpreter
# python-interpreter-mcp v0.2.0
# - mcp-py-server
# - mcp-py-client
```

### Upgrade

```bash
uv tool upgrade python-interpreter-mcp
```

### Local Development

**Editable install** (recommended for developers):

```bash
uv tool install --editable /path/to/claude-workspace/mcp/python-interpreter
claude mcp add --scope user python-interpreter -- mcp-py-server
```

This gives you:
- Commands in PATH (permission patterns like `Bash(mcp-py-client:*)` work)
- Changes to source files take effect immediately (no reinstall needed)

**Script mode** (alternative, commands not in PATH):

```bash
claude mcp add --scope user python-interpreter -- uv run \
  --project "$(git rev-parse --show-toplevel)/mcp/python-interpreter" \
  --script "$(git rev-parse --show-toplevel)/mcp/python-interpreter/python_interpreter/server.py"
```

Or run the server directly for testing (from repo root):

```bash
# Script mode (uses inline script dependencies)
uv run --directory mcp/python-interpreter --script python_interpreter/server.py

# Entry point mode (uses pyproject.toml dependencies)
uv run --project mcp/python-interpreter mcp-py-server
```

## Tools

### `execute`
Execute Python code in persistent scope. Variables, imports, functions, and classes persist across calls.

- Last expression is auto-evaluated (no need to print)
- Returns stdout/stderr, repr() of last expression, or full tracebacks on error
- Outputs >25,000 chars are truncated with full output saved to temp file
- Builtin interpreter auto-installs missing packages via `uv pip install`

```
execute("x = 5")             → ""
execute("x * 2")             → "10"
execute("print(f'x = {x}')") → "x = 5"
execute("import math; math.pi") → "3.141592653589793"
execute("1/0")                → "Traceback ... ZeroDivisionError: division by zero"
```

### `reset`
Clear all variables, imports, and functions from the builtin interpreter scope. Destructive and cannot be undone.
Returns count of items removed.

### `list_vars`
List currently defined variables in the persistent scope. Returns alphabetically sorted names, filtering out
Python builtins.

### `get_session_info`
Get comprehensive session and server metadata including session ID, project directory, socket path, transcript path,
output directory, Claude PID, start time, and uptime.

### `add_interpreter`
Add and start an external Python interpreter subprocess using a different Python executable (e.g., project venv).
No auto-install — uses whatever packages are in that Python environment. `python_path` can be relative to the
project directory (e.g., `.venv/bin/python`).

Set `save=True` to persist the configuration. Saved interpreters appear as "stopped" in `list_interpreters` after
server restart and can be re-started by calling `add_interpreter` again with the same name and path.

### `stop_interpreter`
Stop an external interpreter subprocess. Cannot stop the builtin interpreter. Saved interpreters transition to
"stopped" (config preserved). Set `remove=True` to permanently delete the saved config.

### `list_interpreters`
List all interpreters — running (builtin, saved, transient) and saved-but-stopped. Returns source, state,
python_path, and runtime metadata.

## Security

**WARNING:** This server executes arbitrary Python code. Only use with trusted input.

Code running in interpreters has access to:
- All Python built-ins (open, exec, eval, import, etc.)
- File system, network, and system calls

Code does **not** have access to MCP server internals — each interpreter runs in an isolated subprocess.

## Example Session

```
1. execute("import math; pi_squared = math.pi ** 2")  → ""
2. execute("pi_squared")                               → "9.869604401089358"
3. list_vars()                                         → "math, pi_squared"
4. reset()                                             → "Scope cleared (2 items removed)"
```

## Features

- **Persistent Python scope** - Variables, imports, and functions persist across executions
- **Auto-installs missing packages** - Detects ImportError and installs packages via uv
- **Large output handling** - Outputs >25K chars saved to temp files with paths returned
- **Readable approval prompts** - See [Why the HTTP Bridge Exists](#why-the-http-bridge-exists) for details
- **Multi-interpreter support** - Run code in the builtin interpreter or external Python environments (e.g., project venvs)
- **Session-scoped resources** - Socket path and temp directory tied to Claude session

## Client Usage

The `mcp-py-client` command lets you execute Python with readable heredoc syntax:

```bash
mcp-py-client <<'PY'
import tiktoken
tokens = tiktoken.get_encoding("cl100k_base").encode("Strawberry")
print(f"Token count: {len(tokens)}")
PY
```

This is what Claude uses internally for multi-line Python execution, giving you readable approval prompts.

## Why the HTTP Bridge Exists

### The Problem: Unreadable Approval Prompts

MCP tools require JSON-formatted input, which means multi-line Python code gets escaped into an unreadable mess. When
Claude uses the `execute` tool, your approval prompt looks like this:

```
❯ python-interpreter - Execute Python Code

code: "import tiktoken\ntokens = tiktoken.get_encoding(\"cl100k_base\").encode(\"Strawberry\")\nprint(f\"Token count: {len(tokens)}\")"
```

All those `\n` and `\"` escapes make it **hard to review what code is about to run**. You're approving code you can
barely read.

### The Solution: Heredoc Syntax via mcp-py-client

The server provides an HTTP bridge that lets Claude use Bash heredoc syntax instead. When Claude uses `mcp-py-client`,
your approval prompt shows clean, readable Python:

```
❯ Bash

mcp-py-client <<'PY'
import tiktoken
tokens = tiktoken.get_encoding("cl100k_base").encode("Strawberry")
print(f"Token count: {len(tokens)}")
PY
```

**Same code, same execution, but you can actually read it before approving.**

Claude automatically chooses `mcp-py-client` for multi-line code to give you readable approval prompts. You don't need
to do anything - the server handles this via the HTTP bridge and Unix socket.

## How This Works

### Architecture

```
Claude Code ──[stdio/JSON-RPC]──> mcp-py-server
                                   ├── FastMCP (main, stdio)
                                   └── FastAPI (background, Unix socket)
                                            ▲
                                            │ HTTP POST /execute
                                            │
Bash heredoc ──[stdin]──> mcp-py-client ───┘
                          (auto-discovers socket via process tree)
```

### Session Discovery

The server discovers its Claude Code session by:
1. Walking the process tree upward to find the Claude Code parent process
2. Using `lsof` to determine Claude's working directory and verify `.claude/` files
3. Looking up the active session matching Claude's PID in `sessions.json` (maintained by claude-workspace hooks)

### Unix Socket

The HTTP bridge listens on a session-scoped Unix socket:
- **Path**: `/tmp/python-interpreter-{claude_pid}.sock`
- **Purpose**: Allows Bash heredoc scripts to execute Python code without spawning new interpreters
- **Discovery**: `mcp-py-client` walks the process tree to find the socket path