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

### [Python Interpreter MCP](mcp/python-interpreter)

```bash
claude mcp add --scope user python-interpreter -- uvx --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/python-interpreter server
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
