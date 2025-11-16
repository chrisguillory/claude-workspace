# Claude Workspace

A comprehensive repository for Claude Code customizations including MCP servers, hooks, slash commands, skills, and configuration templates.

## Repository Structure

```
claude-workspace/
├── mcp/                   # MCP-related code (servers, clients, utilities)
│   ├── browser-automation/
│   │   └── server.py
│   ├── python-interpreter/
│   │   ├── server.py
│   │   └── client.py
│   ├── selenium-browser-automation/
│   │   ├── server.py
│   │   └── src/
│   └── utils.py
├── hooks/                 # Event-driven automation scripts
│   ├── pre-tool-use/
│   └── post-tool-use/
├── commands/              # Custom slash commands
│   └── *.md
├── skills/                # Auto-activating capabilities
│   └── */SKILL.md
├── configs/               # Reusable .claude/settings.json templates
├── CLAUDE.md             # Global project memory
└── README.md
```

## What's in This Workspace

### MCP
Custom Model Context Protocol implementations that extend Claude Code's capabilities:
- **Browser Automation** - Playwright-based browser control server
- **Selenium Browser Automation** - Alternative browser automation with CDP stealth
- **Python Interpreter** - Server and client for executing Python code with auto-package installation
- **Utils** - Shared utilities for MCP implementations

### Hooks
Event-driven scripts that trigger on lifecycle events (coming soon)

### Commands
Custom slash commands for repeatable workflows (coming soon)

### Skills
Auto-activating capabilities that Claude applies contextually (coming soon)

### Configs
Reusable configuration templates for different use cases (coming soon)

## Getting Started

### Installing MCP Servers

Each MCP server can be installed by adding to your `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "browser-automation": {
      "command": "python",
      "args": ["/path/to/claude-workspace/mcp/browser-automation/server.py"]
    },
    "python-interpreter": {
      "command": "python",
      "args": ["/path/to/claude-workspace/mcp/python-interpreter/server.py"]
    },
    "selenium-browser-automation": {
      "command": "python",
      "args": ["/path/to/claude-workspace/mcp/selenium-browser-automation/server.py"]
    }
  }
}
```

## Claude Code Customization Types

This workspace supports all Claude Code customization mechanisms:

1. **MCP Servers** - External tools and data source connections
2. **Hooks** - Event-driven automation (PreToolUse, PostToolUse, etc.)
3. **Slash Commands** - Custom reusable workflows
4. **Skills** - Auto-activating capabilities
5. **Plugins** - Distributable packages bundling multiple customizations
6. **CLAUDE.md** - Project memory files for persistent context
7. **Output Styles** - Custom formatting templates
8. **Settings** - Configuration for permissions, sandbox, environment

## Contributing

This is a personal workspace for Claude Code customizations. Feel free to use as reference or fork for your own use.

## License

MIT
