# Claude Workspace - Project Preferences

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This repository contains Claude Code customizations for Chris Guillory, including:
- MCP (Model Context Protocol) servers
- Hooks for event-driven automation
- Custom slash commands
- Skills for auto-activating capabilities
- Shared utilities and configuration templates

**Location:** `/Users/chris/claude-workspace`
**GitHub:** `chrisguillory/claude-workspace`

## Documentation Principles

### Core Philosophy

**Terse but complete** - Every word earns its place. Document principles and decisions, not implementation mechanics. Code is self-documenting through clear naming and structure. Keep documentation local to what it describes.

| Concept                       | Meaning                                      |
|-------------------------------|----------------------------------------------|
| **Terse but complete**        | Every word earns its place                   |
| **Principles over mechanics** | Document "why" and "what", not "how"         |
| **Self-documenting**          | Clear naming and structure explain intent    |
| **Documentation locality**    | Keep docs near what they describe            |
| **Public APIs**               | Classes, functions, tools get docstrings     |

### Application

Before writing any documentation, comments, code descriptions, or commit messages, apply these principles to ensure consistency across the codebase.

## Code Quality Principles

Simple, readable, maintainable, explicit. Fail-fast on misconfiguration. Single responsibility per function/validator.

| Principle                                     | Application                                         |
|-----------------------------------------------|-----------------------------------------------------|
| **Explicit over implicit**                    | Clear control flow, avoid hidden behavior           |
| **Fail-fast validation**                      | Validate configuration at startup, not at first use |
| **Single responsibility**                     | One clear purpose per function/class/module         |
| **Type hints everywhere**                     | Strict Pydantic models, function signatures, `| None` syntax |
| **YAGNI & KISS**                              | Build what's needed, keep it simple                 |
| **Worse is Better**                           | Simple, working solutions over perfect designs      |
| **Make It Work, Make It Right, Make It Fast** | In that order                                       |
| **Avoid Premature Optimization**              | Optimize when data shows need                       |
| **Principle of least surprise**               | Code should behave as expected                      |
| **Bubble exceptions**                         | Don't swallow errors, let them propagate            |
| **Async-first libraries**                     | Use async versions (aioboto3, asyncpg) when available |
| **Assertions only in tests/**                | Application code raises exceptions explicitly       |
| **DI via closures**                           | FastMCP pattern for dependency injection           |

## Python Specifics

### Strict Type Safety

**Eliminate ALL untyped dicts** - Use Pydantic models everywhere. Create shared `BaseModel` that enforces strictness:

```python
import pydantic

class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)
```

**Python 3.13 modern syntax:**
- Use `| None` instead of `Optional`
- Use `str | int` instead of `Union[str, int]`
- Full type checker coverage, not redundant hints
- Hint function signatures and class attributes; skip where inferrable

### Python Idioms

- Use `datetime.now(UTC)` not `.utcnow()`
- Async-first libraries when available (aioboto3, asyncpg)
- Assertions only in `tests/` - application code raises exceptions explicitly
- DI via closures for MCP servers (FastMCP pattern)

### Type Aliases with Literals

**Use Python 3.12+ `type` keyword for type aliases** - particularly for Literal types with specific allowed values:

```python
# local_lib/types.py
from typing import Literal

type SessionSource = Literal["startup", "resume", "compact"]
type SessionState = Literal["started", "ended", "crashed"]
```

**Centralize shared types** in `local-lib/local_lib/types.py` for reuse across files:

```python
# session_tracker.py
from local_lib.types import SessionSource, SessionState

class Session(BaseModel):
    source: SessionSource  # Not str - enforces specific values
    state: SessionState
```

**Benefits:**
- Type safety: IDEs and type checkers enforce valid values
- Single source of truth: Change allowed values in one place
- Self-documenting: No need for comments listing valid values
- Pydantic validation: Runtime enforcement of literal values

### Immutable Types in Annotations

**Prefer immutable types in function signatures and class definitions** to signal that values shouldn't be mutated:

```python
from typing import Sequence, Mapping

# Return types - use immutable interfaces
def get_users() -> Sequence[User]:  # Not list[User]
    return [...]

def get_config() -> Mapping[str, str]:  # Not dict[str, str]
    return {...}

# Class attributes - use immutable for collections
class Config(BaseModel):
    allowed_hosts: Sequence[str]  # Not list[str]
    settings: Mapping[str, str]  # Not dict[str, str]
```

**Implementation can use mutable types** - `list` is a `Sequence`, `dict` is a `Mapping`:

```python
def get_users() -> Sequence[User]:
    users: list[User] = []  # Implementation detail
    # ... populate ...
    return users  # list satisfies Sequence
```

### Top-Down Ordering

**Define classes in top-down order** - high-level abstractions first, then dependencies. Use `from __future__ import annotations` for forward references:

```python
from __future__ import annotations

# High-level container first
class SessionDatabase(BaseModel):
    sessions: Sequence[Session] = []

# Main entity second
class Session(BaseModel):
    metadata: SessionMetadata

# Supporting detail last
class SessionMetadata(BaseModel):
    started_at: datetime
```

This mirrors how humans read code: start with the "what" (high-level purpose), then the "how" (implementation details).

### Scripting

Use `uv run --no-project -` with heredoc + PEP 723 inline dependencies:

```bash
uv run --no-project - <<'PY'
# /// script
# dependencies = [
#     "pandas",
# ]
# ///

import pandas as pd
print("Hello!")
PY
```

## Tooling & Workflow

### Dependency Management

Use `uv` CLI for all dependency operations. Never edit `pyproject.toml` directly for add/remove:

```bash
uv {add,remove} package-name
```

### Pre-commit Hooks

Run pre-commit hooks before committing to catch linting/formatting issues:

```bash
git add -A && .githooks/pre-commit
```

If hooks auto-fix files (isort, ruff format), re-stage and run again until clean.

### Directory-Specific Tooling

Check README files before using directory-specific tooling (migrations, scripts, deployments, etc.) for required environment variables, usage patterns, or workflow requirements.

## GitHub & PR Management

**For GitHub.com:** Always use `gh` CLI, never WebFetch tool.

### PR Updates

Use gh REST API for PR updates (not `gh pr edit` which hits deprecated Projects API):

```bash
# Update PR description from file
gh api -X PATCH repos/owner/repo/pulls/PR_NUMBER -F body="$(cat description.md)"
```

### PR Description Guidelines

Document **what's in the PR, not the journey**. Focus on deliverables and results, not iteration history.

**Avoid:**
- Iteration history ("Following PR review...", "Based on feedback...")
- Process details ("Removed dead code", "Fixed typos")
- Implementation steps that led to the final state

**Include:**
- What functionality is being added/changed
- Key architectural decisions and patterns
- Security/operational improvements
- Configuration and setup details
- Testing approach

## Naming Conventions

| Type | Class Name | File Path |
|------|------------|-----------|
| Client | `S3Client` | `clients/s3.py` |
| Service | `SessionTrackerService` | `services/session_tracker.py` |
| Repository | `UserRepository` | `repositories/user.py` |
| MCP Server | `PythonInterpreterServer` | `mcp/python-interpreter/server.py` |

**Directory rules:**
- Directories plural (`clients/`, `services/`) **except acronyms** (`mcp/` not `mcps/`)
- Files and classes singular (`s3.py`, `S3Client`)

## Repository Structure

```
/Users/chris/claude-workspace/
├── .claude/                    # Claude Code configuration
│   └── settings.local.json     # Hook and permission settings
├── hooks/                      # Claude Code hooks (SessionStart, SessionEnd, etc.)
├── mcp/                        # MCP servers (acronym, stays singular)
│   ├── browser-automation/
│   ├── python-interpreter/
│   └── selenium-browser-automation/
├── local-lib/                  # Shared library code
│   └── local_lib/             # Python package
│       ├── session_tracker.py
│       └── utils.py
├── commands/                   # Custom slash commands
├── skills/                     # Auto-activating skills
└── configs/                    # Configuration templates
```

## Architecture

### local-lib Package Structure

We use a **proper Python package** (`local-lib/`) to share code between MCP servers:

```
local-lib/                  # Project root
├── pyproject.toml         # Package metadata (requires hatchling, depends on fastmcp)
└── local_lib/             # Python package (importable as "local_lib")
    ├── __init__.py
    ├── utils.py           # DualLogger and shared utilities
    └── session_tracker.py
```

**Why this structure?**
- Enables **uv inline script dependencies** with relative paths
- Each MCP server references: `local_lib = { path = "../../local-lib/", editable = true }`
- No hardcoded absolute paths - portable across machines
- Standard Python packaging conventions (PEP 517/518)

### MCP Server Organization

MCP servers are organized **by project** (not by type):

```
mcp/
├── browser-automation/
│   └── server.py
├── python-interpreter/
│   ├── server.py
│   └── client.py
└── selenium-browser-automation/
    ├── server.py
    └── src/
```

Related code (server + client) stays together, not split into separate directories.

### Session Tracking

Sessions are tracked via hooks with the following lifecycle:
1. **SessionStart**: Captures session_id, parent_id, project_dir, claude_pid
2. **Live state**: Session active, tracked in sessions.json
3. **SessionEnd**: Marks session as ended with reason and timestamp

Session data model enforces strict typing via Pydantic BaseModel pattern.

## Patterns

### Inline Script Dependencies

All MCP servers use `uv run --script` with inline dependencies:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "fastmcp>=2.12.5",
#   "local_lib",
#   ...
# ]
#
# [tool.uv.sources]
# local_lib = { path = "../../local-lib/", editable = true }
# ///
```

### Shared Logging

All servers import `DualLogger` from `local_lib.utils`:

```python
from local_lib.utils import DualLogger

# In tools:
logger = DualLogger(ctx)
await logger.info("message")
```

This logs to both MCP context (visible in Claude) and stderr (visible in debug logs).

## Configuration

### MCP Server Paths

MCP servers are registered in `~/.claude.json`:

```json
{
  "mcpServers": {
    "python-interpreter": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--script", "/Users/chris/claude-workspace/mcp/python-interpreter/server.py"]
    }
  }
}
```

**Note:** Uses absolute paths in `~/.claude.json` for reliability.

## Development Guidelines

### Adding New MCP Servers

1. Create directory under `mcp/your-server/`
2. Add `server.py` with uv inline dependencies
3. Include `local_lib` in dependencies for shared utilities
4. Use relative path: `local_lib = { path = "../../local-lib/", editable = true }`
5. Import shared code: `from local_lib.utils import DualLogger`

### Adding Shared Utilities

Add to `local-lib/local_lib/` directory. All MCP servers can import them.

## Testing

Test inline dependencies work:
```bash
cd mcp/python-interpreter
uv run --script server.py
```

Should successfully import `local_lib` and start the server.

## Next Steps

- [ ] Set up pre-commit hooks infrastructure (ruff, isort, mypy for type checking)
- [ ] Consider bringing over docs/ structure from underwriting-api (testing-strategy.md, assertions.md, dependency-injection.md)
- [ ] Migrate remaining untyped dicts to strict Pydantic models
- [ ] Session tracking: Rename session states to use domain language