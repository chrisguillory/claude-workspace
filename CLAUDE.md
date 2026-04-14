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

Principles describe the north star, not the minimum bar. New code embodies them; existing code converges incrementally when touched. Surface adjacent violations via *Improve project health* — don't hunt.

| Principle                                     | Application                                                                                                    |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **Explicit over implicit**                    | Clear control flow, avoid hidden behavior                                                                      |
| **Improve project health**                    | Surface adjacent staleness in any committed artifact (code, docs, configs) — don't silently fix or ignore      |
| **Fail-fast validation**                      | Validate configuration at startup, not at first use                                                            |
| **Single responsibility**                     | One clear purpose per function/class/module; cross-cutting concerns (errors, logging) belong in infrastructure |
| **Type hints everywhere**                     | Strict Pydantic models, function signatures, `\| None` syntax                                                  |
| **YAGNI & KISS**                              | Build what's needed, keep it simple                                                                            |
| **Worse is Better**                           | Simple, working solutions over perfect designs                                                                 |
| **Make It Work, Make It Right, Make It Fast** | In that order                                                                                                  |
| **Avoid Premature Optimization**              | Optimize when data shows need                                                                                  |
| **Principle of least surprise**               | Code should behave as expected                                                                                 |
| **Bubble exceptions**                         | Easier to Ask Forgiveness (EAFP) over Look Before You Leap (LBYL) — let exceptions propagate to handlers       |
| **Leverage existing infrastructure**          | Don't reimplement what a decorator, handler, or base class already provides                                    |
| **DRY**                                       | Deduplicate behavior, not just text — semantic duplication counts                                              |
| **Trust event authority**                     | Don't second-guess events with defensive validation                                                            |
| **Async-first libraries**                     | Use async versions (aioboto3, asyncpg) when available                                                          |
| **Assertions only in tests/**                 | Application code raises exceptions explicitly                                                                  |
| **Ideal state over backwards compat**         | Dev-only project — rename cleanly, no migration shims                                                          |
| **DI via closures**                           | FastMCP pattern for dependency injection                                                                       |

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

**Actionable errors at boundaries** - At system boundaries (external APIs, file I/O, user input), errors carry enough context to diagnose without reproduction. Include: raw response/input preview, expected format, received format, and relevant request parameters. A `KeyError: 'data'` is fail-fast but not actionable; an error showing the response body, status code, and expected schema is both.

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

**When TO handle exceptions** — only in these cases:
1. You've **actually seen** the exception occur in practice
2. The exception is **expected as part of normal flow** (e.g., file existence checks)
3. The library documentation **explicitly requires** handling specific exceptions

**Don't duplicate your infrastructure** — If a decorator, context manager, or handler already covers an error path, the decorated function should not re-handle it. Inline error handling that duplicates infrastructure behavior is a semantic DRY violation and separation of concerns violation simultaneously:

```python
# ❌ Anti-pattern: Manually reimplementing what ErrorBoundary provides
@boundary
def main() -> int:
    result = subprocess.run([...], check=False)
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stdout)
        return 2  # ErrorBoundary already exits with 2
    return 0

# ✅ Correct: Business logic in main(), error handling at the boundary
@boundary
def main() -> int:
    subprocess.run([...], check=True)
    return 0

@boundary.handler(subprocess.CalledProcessError)
def _handle_subprocess(exc: subprocess.CalledProcessError) -> None:
    sys.stderr.buffer.write(exc.stdout)
```

## Python Specifics

### Strict Type Safety

**Eliminate ALL untyped dicts** - Use Pydantic models everywhere. Four base models in `cc_lib/schemas/base.py` encode
trust levels for `extra` field handling (terminology follows JSON Schema closed/open content models):

| Base Class    | `extra` Policy                                                     | Use When                                                            |
|---------------|--------------------------------------------------------------------|---------------------------------------------------------------------|
| `ClosedModel` | `'forbid'` always                                                  | Internal data we construct — unknown fields = bug                   |
| `StrictModel` | `'allow'` default, `'forbid'` via `CC_STRICT_MODEL_EXTRA_FORBID=1` | Protocol data (Claude Code hooks, MCP) — forward-compatible         |
| `CamelModel`  | Inherits from `StrictModel`                                        | Protocol data with camelCase JSON serialization                     |
| `OpenModel`   | `'allow'` default, `'forbid'` via `CC_OPEN_MODEL_EXTRA_FORBID=1`   | External data (Chrome, CDP) — upstream schema evolves independently |
| `SubsetModel` | `'ignore'` always                                                  | Subset of external data — need 3 fields out of 30                   |

All five share: `frozen=True`, `strict=True`, `validate_default=True`, `serialize_by_alias=True`.

**Python 3.13 modern syntax:**
- Use `| None` instead of `Optional`
- Use `str | int` instead of `Union[str, int]`
- Full type checker coverage, not redundant hints
- Hint function signatures and class attributes; skip where inferrable

### Type Aliases and Encapsulation

**Use Python 3.12+ `type` keyword for type aliases** - particularly for Literal types and reusable field specifications:

```python
# cc_lib/types.py
from typing import Literal, Annotated
from datetime import datetime
from pydantic import Field

type SessionSource = Literal["startup", "resume", "compact"]
type SessionState = Literal["active", "exited", "completed", "crashed"]
type JsonDatetime = Annotated[datetime, Field(strict=False)]  # JSON-serializable datetime
```

**Centralize shared types** in `cc-lib/cc_lib/types.py` for reuse across files:

```python
# session_tracker.py
from cc_lib.types import SessionSource, SessionState, JsonDatetime

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

| Layer              | Convention  | Example                        |
|--------------------|-------------|--------------------------------|
| Python field names | snake_case  | `first_name`, `auto_increment` |
| JSON output        | camelCase   | `firstName`, `autoIncrement`   |
| JSON input         | Accept both | `populate_by_name=True`        |

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

### Markdown Kit

Render markdown documents as PDF or serve with live preview. Located at `tools/markdown-kit/markdown-kit.js`.

```bash
# Live preview (preferred — auto-opens in Safari)
node tools/markdown-kit/markdown-kit.js document.md --serve

# Generate PDF
node tools/markdown-kit/markdown-kit.js document.md

# Generate self-contained HTML
node tools/markdown-kit/markdown-kit.js document.md --html

# Share via secret GitHub gist (with embedded images)
node tools/markdown-kit/markdown-kit.js document.md --secret-gist --embed-images
```

**User config file:** `~/.claude-workspace/tools/markdown-kit/config.yaml` sets persistent defaults (theme, toc-nav, macos-spoken-content, launch, front-matter, show-timestamp, show-filepath). CLI flags override config. Use `--no-<flag>` to disable a config-enabled boolean for one run.

Dependencies auto-install on first run. When the user says "serve with markdown-kit", use `--serve` and report the localhost URL.

### Dependency Management

Use `uv` CLI for all dependency operations. Never edit `pyproject.toml` directly for add/remove:

```bash
uv {add,remove} package-name
```

### Pre-commit Hooks

Run pre-commit hooks before committing to catch linting/formatting issues:

```bash
uv run pre-commit run --all-files || uv run pre-commit run --all-files
```

The `||` retry handles auto-fixers (ruff format, trailing whitespace) that modify files on the first run. The second run verifies clean. If the second run fails, it's a real error.

If mypy fails with missing import errors for workspace member dependencies, sync all packages:

```bash
uv sync --all-groups --all-packages
```

### Commit Workflow

**Critical**: Run pre-commit BEFORE writing commit messages. A failed commit wastes all tokens spent generating the message.

```
1. Stage changes: git add <files>
2. Run pre-commit: uv run pre-commit run --all-files || uv run pre-commit run --all-files
3. If pre-commit still fails → fix real issues → re-stage → re-run pre-commit
4. Only AFTER pre-commit passes → write commit message and commit
```

**Stash discipline**: Don't drop stashes carelessly. `git stash pop` with conflicts does NOT auto-drop - the stash remains as a backup. Other stashes (stash@{1}, stash@{2}) may contain unrelated work from previous sessions.

**Separate style from substance**: By default, renames, reorganizations, and formatting go in one commit; features, fixes, and behavioral changes in another. Mixing them makes diffs hard to review and merges/rollbacks complex.

**Docs travel with code**: When a change affects installation, invocation, or configuration, update README/CLAUDE.md/docstrings in the same commit. Not as a follow-up.

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

| Type       | Class Name                | File Path                                               |
|------------|---------------------------|---------------------------------------------------------|
| Client     | `S3Client`                | `clients/s3.py`                                         |
| Service    | `SessionTrackerService`   | `services/session_tracker.py`                           |
| Repository | `UserRepository`          | `repositories/user.py`                                  |
| MCP Server | `PythonInterpreterServer` | `mcp/python-interpreter/python_interpreter/mcp/main.py` |

**Directory rules:**
- Directories plural (`clients/`, `services/`) **except acronyms** (`mcp/` not `mcps/`)
- Files and classes singular (`s3.py`, `S3Client`)

**Entry point pattern:** `<name>` (CLI), `<name>-mcp` (MCP server), `<name>-daemon` (daemon). Role is suffixed, not prefixed. Source mirrors this: `cli/main.py`, `mcp/main.py`.

### Ordering Conventions

Lists of comparable entries are alphabetically sorted unless there's a semantic reason not to. Applies to workspace members in `pyproject.toml`, entry points, permission allow lists, tables of tools/servers, and import groups.

## Repository Structure

```
/Users/chris/claude-workspace/
├── .claude/                    # Claude Code configuration
│   └── settings.local.json     # Hook and permission settings
├── hooks/                      # Claude Code hooks (SessionStart, SessionEnd, etc.)
├── mcp/                        # MCP servers (acronym, stays singular)
│   ├── claude-session/
│   ├── document-search/
│   ├── playwright-browser/
│   ├── python-interpreter/
│   └── selenium-browser/
├── cc-lib/                  # Shared library code
│   └── cc_lib/             # Python package
│       ├── types.py           # Shared type aliases
│       ├── session_tracker.py
│       └── utils.py
├── commands/                   # Custom slash commands
├── skills/                     # Auto-activating skills
└── configs/                    # Configuration templates
```

## Architecture

### cc-lib Package Structure

We use a **proper Python package** (`cc-lib/`) to share code between MCP servers:

```
cc-lib/                  # Project root
├── pyproject.toml         # Package metadata (requires hatchling, depends on fastmcp)
└── cc_lib/             # Python package (importable as "cc_lib")
    ├── __init__.py
    ├── types.py           # Shared type aliases and annotations
    ├── utils.py           # Shared utilities
    └── session_tracker.py
```

**Why this structure?**
- Enables **uv inline script dependencies** with relative paths
- Each MCP server references: `cc_lib = { path = "../../cc-lib/", editable = true }`
- No hardcoded absolute paths - portable across machines
- Standard Python packaging conventions (PEP 517/518)

### MCP Server Organization

MCP servers are organized **by project** (not by type):

```
mcp/
├── claude-session/
│   └── claude_session/
│       ├── cli/main.py
│       └── mcp/main.py
├── document-search/
│   └── document_search/
│       ├── cli/main.py
│       └── mcp/main.py
├── python-interpreter/
│   └── python_interpreter/
│       ├── cli/main.py
│       └── mcp/main.py
└── selenium-browser/
    └── selenium_browser/
        ├── cli/main.py
        ├── mcp/main.py
        └── scripts/
```

Related code (MCP server + CLI) stays together, split into `mcp/` and `cli/` subdirectories.

### Session Tracking

Sessions are tracked via hooks with the following lifecycle:
1. **SessionStart**: Captures session_id, parent_id, project_dir, claude_pid
2. **Live state**: Session active, tracked in sessions.json
3. **SessionEnd**: Marks session as exited/completed/crashed with reason and timestamp

Session data model enforces strict typing via Pydantic BaseModel pattern with orthogonal state/source fields.

## Patterns

### Inline Script Dependencies

Hooks and standalone scripts use `uv run --script` with inline PEP 723 dependencies:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
```

MCP servers use `pyproject.toml` entry points instead (see Installation Methods).

### PATH-Accessible Scripts (Launcher Pattern)

**Problem:** uv doesn't resolve symlinks before computing relative paths in `[tool.uv.sources]`. A symlink at `~/.local/bin/my-cmd` → `~/project/scripts/my-cmd.py` causes `../cc-lib/` to resolve from `~/.local/bin/` instead of `~/project/scripts/`.

**Solution:** Replace symlinks with launcher scripts that pass the resolved path to `uv run --script`:

```bash
# Install with: scripts/install-launcher.sh scripts/my-cmd.py [command-name]
# Generates ~/.local/bin/my-cmd:
#!/bin/sh
exec uv run --no-project --script "/absolute/path/to/scripts/my-cmd.py" "$@"
```

For typer/click scripts, pass `prog_name` at the call site for consistent help text and tab completion:

```python
if __name__ == '__main__':
    app(prog_name=os.path.basename(sys.argv[0]).removesuffix('.py'))
```

### Shared Logging

All MCP servers use Python's `logging` module with stderr output:

```python
import logging
logger = logging.getLogger(__name__)

# In lifespan() — configure once
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stderr,
)

# In tool functions — use the logger
logger.info("message")  # sync, not await
```

Messages appear in `~/.claude/debug/{session_id}.txt`. Servers with operation logs (like document-search) can attach FileHandlers to the root logger for per-operation log files.

## Configuration

### MCP Server Paths

MCP servers are installed via `uv tool install` and registered in `~/.claude.json`:

```bash
# Install globally
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

# Configure Claude Code
claude mcp add --scope user python-interpreter -- python-interpreter-mcp
```

This adds to `~/.claude.json`:

```json
{
  "mcpServers": {
    "python-interpreter": {
      "type": "stdio",
      "command": "python-interpreter-mcp",
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

| Component | Pattern         | Example                     | Module Path              |
|-----------|-----------------|-----------------------------|--------------------------|
| CLI       | `<name>`        | `python-interpreter`        | `<pkg>.cli.main:main`    |
| MCP       | `<name>-mcp`    | `python-interpreter-mcp`    | `<pkg>.mcp.main:main`    |
| Daemon    | `<name>-daemon` | `claude-remote-bash-daemon` | `<pkg>.daemon:main`      |

The directory name under `mcp/` IS the entry point base name. Role is suffixed, not prefixed.

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

IMPORTANT: For better user experience, you should typically use the CLI instead:
    python-interpreter <<'PY'
    print("Hello")
    PY

If python-interpreter is not found, install via:
    uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

Only use this MCP tool directly if the user explicitly requests it or you need structured output."""
```

### Installation Methods

Four installation patterns in order of preference:

**1. Global install** (recommended for users):
```bash
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/<server>
claude mcp add --scope user <name> -- <name>-mcp
```
- Fast startup (no network call)
- Version locked until explicit upgrade
- Works offline after install

**2. uvx** (always-latest, slower startup):
```bash
claude mcp add --scope user <name> -- uvx --refresh --from \
  git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/<server> \
  <name>-mcp
```
- Network call every startup
- Always gets latest from git
- No explicit upgrade needed

**3. Local development - editable install** (recommended for developers):
```bash
uv tool install --editable /path/to/claude-workspace/mcp/<server>
claude mcp add --scope user <name> -- <name>-mcp
```
- Commands in PATH (permission patterns work)
- Changes to source files take effect immediately
- Best of both worlds for active development

**4. Local development - script mode** (alternative):
```bash
claude mcp add --scope user <name> -- uv run \
  --project "$(git rev-parse --show-toplevel)/mcp/<server>" \
  --script "$(git rev-parse --show-toplevel)/mcp/<server>/<package>/mcp/main.py"
```
Where `<package>` is the underscored Python package name (e.g., `python_interpreter`).
- Uses local source files directly
- Commands NOT in PATH (permission patterns require absolute paths)
- Useful when you don't want to install globally

### Permission Patterns

Claude Code permission patterns use **literal prefix matching** (no shell expansion):

| Pattern                              | Works? | Reason                        |
|--------------------------------------|--------|-------------------------------|
| `Bash(python-interpreter:*)`         | ✅      | Simple command name           |
| `Bash("$(git rev-parse ...)":*)`     | ❌      | Shell expansion not evaluated |
| `Bash(/absolute/path/cli/main.py:*)` | ✅      | Literal path match            |

This is why `uv tool install` is preferred—it creates simple commands in PATH that match easily.

### CLI vs MCP Tool

| Component                                     | Use Case                                  | Invocation       |
|-----------------------------------------------|-------------------------------------------|------------------|
| MCP Tool (`mcp__python-interpreter__execute`) | Structured output, explicit request       | Via MCP protocol |
| CLI (`python-interpreter`)                    | Multiline code, readable approval prompts | Heredoc syntax   |

Claude should prefer the CLI for multiline code because approval prompts show clean, readable text instead of escaped JSON strings.

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
2. Add `mcp/main.py` and `cli/main.py` following the module layout convention
3. Add `pyproject.toml` with entry points following naming convention (`<name>-mcp`, `<name>`)
4. Include `cc_lib` in dependencies for shared utilities
5. Use relative path: `cc_lib = { path = "../../cc-lib/", editable = true }`
6. Configure logging in `lifespan()`: `logging.basicConfig(stream=sys.stderr, ...)`
7. Write tool docstrings that guide Claude's behavior

### Adding Shared Utilities

Add to `cc-lib/cc_lib/` directory. All MCP servers can import them. Follow public-first organization with explicit `__all__` exports.

## Testing

Test inline dependencies work:
```bash
uv run --directory mcp/python-interpreter --script python_interpreter/mcp/main.py
```

Should successfully import `cc_lib` and start the server.

Test entry points work:
```bash
uv run --project mcp/python-interpreter python-interpreter-mcp
```

## Next Steps

- [x] Set up pre-commit hooks infrastructure (ruff, isort, mypy for type checking) - Done: `uv run pre-commit run --all-files`
- [ ] Consider bringing over docs/ structure from underwriting-api (testing-strategy.md, assertions.md, dependency-injection.md)
- [ ] Migrate remaining untyped dicts to strict Pydantic models
