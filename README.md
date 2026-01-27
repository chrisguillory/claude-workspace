# Claude Workspace

A comprehensive repository for Claude Code customizations including MCP servers, hooks, slash commands, skills, and
configuration templates.

**GitHub:** `chrisguillory/claude-workspace`

## Repository Structure

```
claude-workspace/
├── .claude/                    # Claude Code configuration
│   └── settings.local.json     # Hook and permission settings
├── hooks/                      # Event-driven automation
│   ├── session-start.py
│   ├── session-end.py
│   └── print-session-info.py
├── mcp/                        # MCP servers
│   ├── browser-automation/
│   ├── python-interpreter/
│   └── selenium-browser-automation/
├── local-lib/                  # Shared Python package
│   └── local_lib/
│       ├── session_tracker.py
│       └── utils.py
├── commands/                   # Custom slash commands
├── skills/                     # Auto-activating capabilities
├── configs/                    # Configuration templates
├── docs/                       # Architecture documentation
├── CLAUDE.md                   # Coding standards
└── README.md
```

## Installing MCP Servers

MCP servers can be installed globally via `uv tool install` (recommended) or `uvx` from GitHub.

### [Python Interpreter MCP](mcp/python-interpreter)

Persistent Python execution environment with variable retention across calls.

```bash
# Install globally (recommended)
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter

# Configure Claude Code
claude mcp add --scope user python-interpreter -- mcp-py-server

# Check version / Upgrade
uv tool list | grep python-interpreter
uv tool upgrade python-interpreter-mcp

# Alternative: uvx (runs fresh each time, no persistent install)
claude mcp add --scope user python-interpreter -- uvx --refresh --from \
  git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter \
  mcp-py-server

# Local development (live code changes)
claude mcp add --scope user python-interpreter -- uv run \
  --project "$(git rev-parse --show-toplevel)/mcp/python-interpreter" \
  --script "$(git rev-parse --show-toplevel)/mcp/python-interpreter/python_interpreter/server.py"
```

### [Browser Automation MCP](mcp/browser-automation)

Playwright-based browser control with stealth mode for web automation.

```bash
# From GitHub (recommended)
claude mcp add --scope user browser-automation -- uvx --refresh --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/browser-automation browser-server

# From local clone
claude mcp add --scope user browser-automation -- uv run \
  --project ~/claude-workspace/mcp/browser-automation \
  --script ~/claude-workspace/mcp/browser-automation/browser_automation/server.py
```

**Note:** Requires Playwright browsers. After installation, run: `playwright install chromium`

### [Selenium Browser Automation MCP](mcp/selenium-browser-automation)

Selenium with CDP stealth injection to bypass Cloudflare bot detection.

```bash
# From GitHub (recommended)
claude mcp add --scope user selenium-browser-automation -- uvx --refresh --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/selenium-browser-automation selenium-server

# From local clone
claude mcp add --scope user selenium-browser-automation -- uv run \
  --project ~/claude-workspace/mcp/selenium-browser-automation \
  --script ~/claude-workspace/mcp/selenium-browser-automation/selenium_browser_automation/server.py
```

**Note:** Requires Chrome/Chromium installed on the system.

### [Document Search MCP](mcp/document-search)

Semantic search over local documents using Gemini embeddings and Qdrant.

```bash
# Editable install (local development)
uv tool install --editable ~/claude-workspace/mcp/document-search
claude mcp add --scope user document-search -- mcp-document-search

# Start Qdrant
docker compose -f ~/claude-workspace/mcp/document-search/docker-compose.yaml up -d
```

**Note:** Requires a Gemini API key at `~/.claude-workspace/secrets/document_search_api_key`.

### Local Development

For local development with live code changes, use workspace mode from the repo root:

```bash
cd ~/claude-workspace
uv run --project python-interpreter-mcp mcp-py-server
uv run --project browser-automation-mcp browser-server
uv run --project selenium-browser-automation-mcp selenium-server
```

### Dependency Resolution and Reproducibility

This repo includes a `uv.lock` file, but whether it's used depends on your install method:

| Install Method | Uses uv.lock | Dependency Versions |
|----------------|--------------|---------------------|
| `uv sync` / `uv run` (local clone) | Yes | Exact (fully reproducible) |
| `uvx --from git+...` (remote) | No | Resolved fresh from constraints |

**Why uvx ignores lock files:** This is an architectural limitation, not a bug. `uvx` installs from wheels, and wheels cannot contain lock files. The [Astral team has acknowledged this](https://github.com/astral-sh/uv/issues/13410) and is exploring solutions like `uvx --locked`, but none exist yet.

**uvx caching behavior:** Dependencies are cached per Git commit SHA. The *first* install of a given commit resolves fresh from PyPI; subsequent runs reuse the cached resolution. Use `--refresh` to force re-resolution.

**What varies:** With `uvx`, source code is deterministic (same commit = same code), but dependency versions may differ over time as new releases satisfy constraints (e.g., `fastmcp>=2.12.5` might resolve to 2.12.5 today, 2.13.0 next month).

**For full reproducibility:** Clone the repo and use the lock file:
```bash
git clone https://github.com/chrisguillory/claude-workspace.git
cd claude-workspace
uv sync
uv run --project browser-automation-mcp browser-server
```

## Hook Installation

Claude Code supports **4 configuration levels** that merge together, with more specific settings overriding broader ones:

| Level             | File Location                 | Scope             | Use Case                              |
|-------------------|-------------------------------|-------------------|---------------------------------------|
| **Enterprise**    | `managed-settings.json`       | Organization-wide | IT-managed policies (cannot override) |
| **User (Global)** | `~/.claude/settings.json`     | All your projects | Personal automation across all work   |
| **Project**       | `.claude/settings.json`       | Single repository | Team-shared hooks (committed to git)  |
| **Local**         | `.claude/settings.local.json` | Single repository | Personal overrides (git-ignored)      |

### Recommended: Global Installation

We recommend installing hooks globally for session tracking across all projects.

Edit `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/claude-workspace/hooks/session-start.py"
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/claude-workspace/hooks/session-end.py"
          }
        ]
      }
    ]
  }
}
```

Replace `~/claude-workspace` with your absolute path to this repository.
