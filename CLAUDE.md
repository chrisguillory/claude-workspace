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
| **Trust event authority**                     | Don't second-guess events with defensive validation |
| **Async-first libraries**                     | Use async versions (aioboto3, asyncpg) when available |
| **Assertions only in tests/**                | Application code raises exceptions explicitly       |
| **DI via closures**                           | FastMCP pattern for dependency injection           |

### Exception Handling

**Bubble exceptions up** - Never swallow errors with try/except that returns defaults. Let exceptions propagate to appropriate handlers.

```python
# ❌ Anti-pattern: Swallowing exceptions
def get_config() -> dict:
    try:
        return load_config()
    except Exception:
        return {}  # Hides configuration errors

# ✅ Correct: Let exceptions bubble
def get_config() -> dict:
    return load_config()  # Caller decides how to handle errors
```

**Trust event authority** - When receiving events, trust they represent what they claim. Fix incorrect events at their source, not with defensive validation in handlers:

```python
# ❌ Anti-pattern: Defensive validation of events
def handle_session_end(session_id: str):
    if not is_session_still_active(session_id):  # Stale check
        return  # Defensive complexity

# ✅ Correct: Trust the event
def handle_session_end(session_id: str):
    mark_session_ended(session_id)  # Event fired = session ended
```

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

### Type Aliases and Encapsulation

**Use Python 3.12+ `type` keyword for type aliases** - particularly for Literal types and reusable field specifications:

```python
# local_lib/types.py
from typing import Literal, Annotated
from datetime import datetime
from pydantic import Field

type SessionSource = Literal["startup", "resume", "compact"]
type SessionState = Literal["active", "exited", "completed", "crashed"]
type JsonDatetime = Annotated[datetime, Field(strict=False)]  # JSON-serializable datetime
```

**Centralize shared types** in `local-lib/local_lib/types.py` for reuse across files:

```python
# session_tracker.py
from local_lib.types import SessionSource, SessionState, JsonDatetime

class Session(BaseModel):
    source: SessionSource  # Not str - enforces specific values
    state: SessionState
    started_at: JsonDatetime  # Reusable type encapsulation
```

**Benefits:**
- Type safety: IDEs and type checkers enforce valid values
- Single source of truth: Change allowed values in one place
- Self-documenting: No need for comments listing valid values
- Pydantic validation: Runtime enforcement of literal values
- DRY: Type encapsulation prevents Field() repetition

### Data Model Design

**Separate orthogonal concerns** - State (current status) and source (origin/history) are independent dimensions:

```python
# ✅ Correct: Orthogonal fields
class Session(BaseModel):
    state: SessionState  # active, exited, completed, crashed
    source: SessionSource  # startup, resume, compact

# ❌ Anti-pattern: Conflated concerns
class Session(BaseModel):
    status: Literal["startup_active", "resume_ended", ...]  # Mixes two concepts
```

**Order fields for readability** - Identity → Status → Location → Details:

```python
class Session(BaseModel):
    # Identity
    id: str

    # Status
    state: SessionState
    source: SessionSource

    # Location
    project_dir: Path

    # Details
    started_at: JsonDatetime
    metadata: dict[str, Any]
```

### Module Organization

**Public-first file structure** with explicit API boundaries:

```python
"""Module docstring."""

from __future__ import annotations  # Enable forward references

# Explicit public API
__all__ = ["Session", "SessionDatabase", "load_sessions"]

# High-level abstractions first (top-down ordering)
class SessionDatabase(BaseModel):
    sessions: Sequence[Session] = []

class Session(BaseModel):
    metadata: SessionMetadata

# Public functions
def load_sessions() -> SessionDatabase:
    ...

# Private helper functions (not in __all__)
def _validate_path(path: Path) -> None:
    ...
```

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

### Python Idioms

- Use `datetime.now(UTC)` not `.utcnow()`
- Async-first libraries when available (aioboto3, asyncpg)
- Assertions only in `tests/` - application code raises exceptions explicitly
- DI via closures for MCP servers (FastMCP pattern)
- **NEVER use `python` or `python3` directly** - always use `uv run`

### JSON Serialization for JavaScript Consumers

**Python snake_case internally, camelCase JSON output** - Use Pydantic v2's `alias_generator` to maintain Pythonic code while producing JavaScript-friendly JSON.

| Layer | Convention | Example |
|-------|------------|---------|
| Python field names | snake_case | `first_name`, `auto_increment` |
| JSON output | camelCase | `firstName`, `autoIncrement` |
| JSON input | Accept both | `populate_by_name=True` |

**Industry standard:** Google APIs, GraphQL, JSON:API spec all use camelCase for JSON. Major exceptions (Stripe, GitHub) provide SDKs that handle the conversion.

**Pydantic v2 pattern:**

```python
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class UserProfile(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,  # Accept both snake_case and camelCase input
    )

    first_name: str           # Python uses snake_case
    email_address: str
    phone_number: str | None = None

# Serialization: use by_alias=True for camelCase output
user.model_dump_json(by_alias=True)
# → {"firstName": "...", "emailAddress": "...", "phoneNumber": null}
```

**When fields need custom aliases** (e.g., `name` → `databaseName`):

```python
class Database(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    database_name: str  # to_camel produces "databaseName" ✓
    version: int        # No transformation needed
```

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
uv run --group dev pre-commit run --all-files
```

If hooks auto-fix files (ruff format), re-stage and run again until clean.

### Commit Workflow

**Critical**: Run pre-commit BEFORE writing commit messages. A failed commit wastes all tokens spent generating the message.

```
1. Stage changes: git add <files>
2. Run pre-commit: uv run --group dev pre-commit run --all-files
3. If pre-commit fails → fix issues → re-stage → re-run pre-commit
4. Only AFTER pre-commit passes → write commit message and commit
```

**Stash discipline**: Don't drop stashes carelessly. `git stash pop` with conflicts does NOT auto-drop - the stash remains as a backup. Other stashes (stash@{1}, stash@{2}) may contain unrelated work from previous sessions.

**MCP server reconnect**: Ask the user to run `/mcp reconnect <server-name>` rather than trying bash commands. The CLI handles this.

**Debug logs**: Session debug logs are at `~/.claude/debug/{session_id}.txt` for troubleshooting MCP server startup failures and other issues.

### Directory-Specific Tooling

Check README files before using directory-specific tooling (migrations, scripts, deployments, etc.) for required environment variables, usage patterns, or workflow requirements.

### Path Documentation

Use `$(git rev-parse --show-toplevel)` for repo-relative paths in examples:

```bash
# ✅ Portable path reference
cd "$(git rev-parse --show-toplevel)/mcp/python-interpreter"

# ❌ Hardcoded absolute path
cd /Users/chris/claude-workspace/mcp/python-interpreter
```

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
| MCP Server | `PythonInterpreterServer` | `mcp/python-interpreter/python_interpreter/server.py` |

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
│       ├── types.py           # Shared type aliases
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
    ├── types.py           # Shared type aliases and annotations
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
│   └── browser_automation/
│       └── server.py
├── python-interpreter/
│   └── python_interpreter/
│       ├── server.py
│       └── client.py
└── selenium-browser-automation/
    └── selenium_browser_automation/
        ├── server.py
        └── scripts/
```

Related code (server + client) stays together, not split into separate directories.

### Session Tracking

Sessions are tracked via hooks with the following lifecycle:
1. **SessionStart**: Captures session_id, parent_id, project_dir, claude_pid
2. **Live state**: Session active, tracked in sessions.json
3. **SessionEnd**: Marks session as exited/completed/crashed with reason and timestamp

Session data model enforces strict typing via Pydantic BaseModel pattern with orthogonal state/source fields.

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

MCP servers are installed via `uv tool install` and registered in `~/.claude.json`:

```bash
# Install globally
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

# Configure Claude Code
claude mcp add --scope user python-interpreter -- mcp-py-server
```

This adds to `~/.claude.json`:

```json
{
  "mcpServers": {
    "python-interpreter": {
      "type": "stdio",
      "command": "mcp-py-server",
      "args": []
    }
  }
}
```

## MCP Server Operations

### Version Management

MCP servers use semantic versioning in `pyproject.toml`. **Always bump version when making changes** that affect installed users:

| Change Type                       | Version Bump | Example       |
|-----------------------------------|--------------|---------------|
| Bug fixes, docstring updates      | Patch        | 0.2.0 → 0.2.1 |
| New features, entry point changes | Minor        | 0.1.0 → 0.2.0 |
| Breaking changes                  | Major        | 0.x.x → 1.0.0 |

Users see versions via `uv tool list` and upgrade via `uv tool upgrade <package-name>`.

### Entry Point Naming Convention

| Component | Pattern                  | Example         |
|-----------|--------------------------|-----------------|
| Server    | `mcp-<shortname>-server` | `mcp-py-server` |
| Client    | `mcp-<shortname>-client` | `mcp-py-client` |

Avoid generic names like `server` that could collide with other tools when installed globally via `uv tool install`.

### Tool Docstrings Guide Claude's Behavior

**Critical:** MCP tool docstrings are what Claude reads to understand how to use tools. When installation or usage patterns change, update:

1. The tool's docstring (what Claude sees)
2. Module-level docstrings (for human reference)
3. README documentation

Include in tool docstrings:
- Preferred usage pattern
- Installation instructions if command not found
- Fallback options

Example pattern:
```python
"""Execute Python code in persistent scope...

IMPORTANT: For better user experience, you should typically use the Bash client instead:
    mcp-py-client <<'PY'
    print("Hello")
    PY

If mcp-py-client is not found, install via:
    uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

Only use this MCP tool directly if the user explicitly requests it or you need structured output."""
```

### Installation Methods

Three installation patterns in order of preference:

**1. Global install** (recommended for users):
```bash
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/<server>
claude mcp add --scope user <name> -- mcp-<shortname>-server
```
- Fast startup (no network call)
- Version locked until explicit upgrade
- Works offline after install

**2. uvx** (always-latest, slower startup):
```bash
claude mcp add --scope user <name> -- uvx --refresh --from \
  git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/<server> \
  mcp-<shortname>-server
```
- Network call every startup
- Always gets latest from git
- No explicit upgrade needed

**3. Local development - editable install** (recommended for developers):
```bash
uv tool install --editable /path/to/claude-workspace/mcp/<server>
claude mcp add --scope user <name> -- mcp-<shortname>-server
```
- Commands in PATH (permission patterns work)
- Changes to source files take effect immediately
- Best of both worlds for active development

**4. Local development - script mode** (alternative):
```bash
claude mcp add --scope user <name> -- uv run \
  --project "$(git rev-parse --show-toplevel)/mcp/<server>" \
  --script "$(git rev-parse --show-toplevel)/mcp/<server>/<package>/server.py"
```
Where `<package>` is the underscored Python package name (e.g., `python_interpreter`).
- Uses local source files directly
- Commands NOT in PATH (permission patterns require absolute paths)
- Useful when you don't want to install globally

### Permission Patterns

Claude Code permission patterns use **literal prefix matching** (no shell expansion):

| Pattern | Works? | Reason |
|---------|--------|--------|
| `Bash(mcp-py-client:*)` | ✅ | Simple command name |
| `Bash("$(git rev-parse ...)":*)` | ❌ | Shell expansion not evaluated |
| `Bash(/absolute/path/client.py:*)` | ✅ | Literal path match |

This is why `uv tool install` is preferred—it creates simple commands in PATH that match easily.

### Client vs MCP Tool

| Component | Use Case | Invocation |
|-----------|----------|------------|
| MCP Tool (`mcp__python-interpreter__execute`) | Structured output, explicit request | Via MCP protocol |
| Bash Client (`mcp-py-client`) | Multiline code, readable approval prompts | Heredoc syntax |

Claude should prefer the Bash client for multiline code because approval prompts show clean, readable Python instead of escaped JSON strings.

### Release Process

When modifying MCP servers:

1. Make code changes
2. Update tool docstrings if usage patterns changed
3. Bump version in `pyproject.toml`
4. Update README if needed
5. Commit with appropriate prefix (`feat:`, `fix:`)
6. Push to git
7. Users upgrade via `uv tool upgrade <package-name>`

## Development Guidelines

### Adding New MCP Servers

1. Create directory under `mcp/your-server/`
2. Add `server.py` with uv inline dependencies
3. Add `pyproject.toml` with entry points following naming convention
4. Include `local_lib` in dependencies for shared utilities
5. Use relative path: `local_lib = { path = "../../local-lib/", editable = true }`
6. Import shared code: `from local_lib.utils import DualLogger`
7. Write tool docstrings that guide Claude's behavior

### Adding Shared Utilities

Add to `local-lib/local_lib/` directory. All MCP servers can import them. Follow public-first organization with explicit `__all__` exports.

## Testing

Test inline dependencies work:
```bash
uv run --directory mcp/python-interpreter --script python_interpreter/server.py
```

Should successfully import `local_lib` and start the server.

Test entry points work:
```bash
uv run --project mcp/python-interpreter mcp-py-server
```

## Next Steps

- [x] Set up pre-commit hooks infrastructure (ruff, isort, mypy for type checking) - Done: `uv run --group dev pre-commit run --all-files`
- [ ] Consider bringing over docs/ structure from underwriting-api (testing-strategy.md, assertions.md, dependency-injection.md)
- [ ] Migrate remaining untyped dicts to strict Pydantic models