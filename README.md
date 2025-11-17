# Claude Workspace

A comprehensive repository for Claude Code customizations including MCP servers, hooks, slash commands, skills, and configuration templates.

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
