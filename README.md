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
│   ├── claude-session/
│   ├── document-search/
│   ├── playwright-browser/
│   ├── python-interpreter/
│   └── selenium-browser/
├── cc-lib/                  # Shared Python package
│   └── cc_lib/
│       ├── session_tracker.py
│       └── utils.py
├── scripts/                    # Linters and diagnostics
├── commands/                   # Custom slash commands
├── skills/                     # Auto-activating capabilities
├── configs/                    # Configuration templates
├── CLAUDE.md                   # Coding standards
└── README.md
```

## Installing MCP Servers

Three installation methods, in order of preference:

| Method | Best For | Live Code Changes? |
|--------|----------|--------------------|
| `uv tool install --editable` | Local development | Yes |
| `uv tool install` | Remote users | No (upgrade to update) |
| `uvx` | One-off / always-latest | No (fresh each run) |

### Quick Setup (Local Development)

Install all servers as editable from a local clone. Changes to source files take effect immediately.

```bash
cd ~/claude-workspace

# Install all servers (editable)
uv tool install --editable mcp/claude-session --force
uv tool install --editable mcp/document-search --force
uv tool install --editable mcp/python-interpreter --force
uv tool install --editable mcp/selenium-browser --force

# Register with Claude Code
claude mcp add --scope user claude-session -- claude-session-mcp
claude mcp add --scope user document-search -- document-search-mcp
claude mcp add --scope user python-interpreter -- python-interpreter-mcp
claude mcp add --scope user selenium-browser -- selenium-browser-mcp

# Verify
uv tool list
```

### Upgrading / Reinstalling

If an MCP server fails to start after pulling updates, force reinstall and re-register:

```bash
cd ~/claude-workspace

# Force reinstall + re-register (same commands as Quick Setup)
uv tool install --editable mcp/claude-session --force
uv tool install --editable mcp/document-search --force
uv tool install --editable mcp/python-interpreter --force
uv tool install --editable mcp/selenium-browser --force

claude mcp remove claude-session --scope user
claude mcp remove document-search --scope user
claude mcp remove python-interpreter --scope user
claude mcp remove selenium-browser --scope user

claude mcp add --scope user claude-session -- claude-session-mcp
claude mcp add --scope user document-search -- document-search-mcp
claude mcp add --scope user python-interpreter -- python-interpreter-mcp
claude mcp add --scope user selenium-browser -- selenium-browser-mcp
```

### Per-Server Details

#### [Claude Session MCP](mcp/claude-session)

Archive, restore, clone, and manage Claude Code sessions.

| Entry Point | Purpose |
|-------------|---------|
| `claude-session-mcp` | MCP server (stdio) |
| `claude-session` | CLI client |

```bash
# Remote install (from GitHub)
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/claude-session
claude mcp add --scope user claude-session -- claude-session-mcp
```

#### [Document Search MCP](mcp/document-search)

Semantic search over local documents using Gemini embeddings and Qdrant.

| Entry Point                 | Purpose                 |
|-----------------------------|-------------------------|
| `document-search-mcp`       | MCP server (stdio)      |
| `document-search`           | CLI client              |
| `document-search-dashboard` | Observability dashboard |

```bash
# Remote install (from GitHub)
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/document-search
claude mcp add --scope user document-search -- document-search-mcp

# Start Qdrant
docker compose -f ~/claude-workspace/mcp/document-search/docker-compose.yaml up -d
```

**Requires:** Gemini API key at `~/.claude-workspace/secrets/document_search_api_key`.

#### [Playwright Browser MCP](mcp/playwright-browser)

Playwright-based browser control with stealth mode.

| Entry Point | Purpose |
|-------------|---------|
| `playwright-browser-mcp` | MCP server (stdio) |

```bash
# Remote install (from GitHub)
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/playwright-browser
claude mcp add --scope user playwright-browser -- playwright-browser-mcp
```

**Requires:** Playwright browsers: `playwright install chromium`

#### [Python Interpreter MCP](mcp/python-interpreter)

Persistent Python execution environment with variable retention across calls and external interpreter support.

| Entry Point | Purpose |
|-------------|---------|
| `python-interpreter-mcp` | MCP server (stdio) |
| `python-interpreter` | CLI client for heredoc usage |

```bash
# Remote install (from GitHub)
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter
claude mcp add --scope user python-interpreter -- python-interpreter-mcp

# Check version / Upgrade
uv tool list | grep python-interpreter
uv tool upgrade python-interpreter
```

#### [Selenium Browser MCP](mcp/selenium-browser)

Selenium with CDP stealth injection to bypass Cloudflare bot detection.

| Entry Point | Purpose |
|-------------|---------|
| `selenium-browser-mcp` | MCP server (stdio) |
| `selenium-browser` | CLI client |

```bash
# Remote install (from GitHub)
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/selenium-browser
claude mcp add --scope user selenium-browser -- selenium-browser-mcp
```

**Requires:** Chrome/Chromium installed on the system.

### Entry Point Reference

Entry points follow the `<name>-mcp` convention for MCP servers and `<name>` for CLIs:

| Package | Entry Points |
|---------|-------------|
| `claude-session` | `claude-session-mcp`, `claude-session` |
| `document-search` | `document-search-mcp`, `document-search`, `document-search-dashboard` |
| `playwright-browser` | `playwright-browser-mcp` |
| `python-interpreter` | `python-interpreter-mcp`, `python-interpreter` |
| `selenium-browser` | `selenium-browser-mcp`, `selenium-browser` |

### Dependency Resolution and Reproducibility

This repo includes a `uv.lock` file, but whether it's used depends on your install method:

| Install Method | Uses uv.lock | Dependency Versions |
|----------------|--------------|---------------------|
| `uv sync` / `uv run` (local clone) | Yes | Exact (fully reproducible) |
| `uv tool install --editable` (local) | No | Resolved fresh, but editable |
| `uv tool install` (remote) | No | Resolved fresh from constraints |
| `uvx --from git+...` (remote) | No | Resolved fresh from constraints |

**Why non-sync installs ignore lock files:** This is an architectural limitation, not a bug. Tool installs and `uvx` install from wheels, and wheels cannot contain lock files. The [Astral team has acknowledged this](https://github.com/astral-sh/uv/issues/13410) and is exploring solutions like `uvx --locked`, but none exist yet.

**For full reproducibility:** Clone the repo and use the lock file:
```bash
git clone https://github.com/chrisguillory/claude-workspace.git
cd claude-workspace
uv sync
uv run --project mcp/python-interpreter python-interpreter-mcp
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