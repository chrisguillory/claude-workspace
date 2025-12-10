# Python Interpreter MCP Server

Persistent Python execution environment for Claude Code with automatic package installation and stateful computation.

## Installation

### From GitHub (recommended)

```bash
claude mcp add --scope user python-interpreter -- uvx --from \
  git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter \
  server
```

### Local development

```bash
# From within the claude-workspace repo:
claude mcp add --scope user python-interpreter -- uv run \
  --project "$(git rev-parse --show-toplevel)/mcp/python-interpreter" \
  --script "$(git rev-parse --show-toplevel)/mcp/python-interpreter/server.py"
```

## Tools

### `execute`
Execute Python code in persistent scope. Returns expression values, stdout/stderr, or paths to large outputs.

### `reset`
Clear all variables and reset to fresh state.

### `list_vars`
List currently defined variables in the persistent scope.

### `get_session_info`
Get comprehensive session and server metadata including session ID, project directory, socket path, transcript path,
output directory, Claude PID, start time, and uptime.

## Features

- **Persistent Python scope** - Variables, imports, and functions persist across executions
- **Auto-installs missing packages** - Detects ImportError and installs packages via uv
- **Large output handling** - Outputs >50KB saved to temp files with paths returned
- **Readable approval prompts** - See [Why the HTTP Bridge Exists](#why-the-http-bridge-exists) for details
- **Session-scoped resources** - Socket path and temp directory tied to Claude session

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

### The Solution: Heredoc Syntax via client.py

The server provides an HTTP bridge that lets Claude use Bash heredoc syntax instead. When Claude uses `client.py`, your
approval prompt shows clean, readable Python:

```
❯ Bash

"$(git rev-parse --show-toplevel)"/mcp/python-interpreter/client.py <<'PY'
import tiktoken
tokens = tiktoken.get_encoding("cl100k_base").encode("Strawberry")
print(f"Token count: {len(tokens)}")
PY
```

**Same code, same execution, but you can actually read it before approving.**

Claude automatically chooses `client.py` for multi-line code to give you readable approval prompts. You don't need to do
anything - the server handles this via the HTTP bridge and Unix socket.

## How This Works

### Architecture

```
Claude Code ──[stdio/JSON-RPC]──> mcp/python-interpreter/server.py
                                   ├── FastMCP (main, stdio)
                                   └── FastAPI (background, Unix socket)
                                            ▲
                                            │ HTTP POST /execute
                                            │
Bash heredoc ──[stdin]──> client.py ───────┘
                          (auto-discovers socket via process tree)
```

### Session Discovery

The server discovers its Claude Code session by:
1. Emitting a unique marker to stderr on startup
2. Searching Claude's debug logs for that marker
3. Extracting the session ID from the log filename

### Unix Socket

The HTTP bridge listens on a session-scoped Unix socket:
- **Path**: `/tmp/python-interpreter-{claude_pid}.sock`
- **Purpose**: Allows Bash heredoc scripts to execute Python code without spawning new interpreters
- **Discovery**: `client.py` walks the process tree to find the socket path