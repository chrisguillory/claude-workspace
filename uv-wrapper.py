#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

"""UV wrapper that implements script-relative project discovery.

## Why This Exists

UV has a fundamental limitation: when you run `uv run /path/to/script.py`, it looks for
project configuration (pyproject.toml) starting from the CURRENT WORKING DIRECTORY, not
from the script's location. This breaks the principle of least surprise - every other
tool (git, npm, cargo) walks up from the file's location, not from where you happen to
be standing.

This wrapper implements what UV should do but doesn't:
1. Walk up from the SCRIPT's location to find pyproject.toml
2. Change to that project directory
3. Set PYTHONPATH=. so local imports work (e.g., `from src.module import foo`)
4. Run the script with proper Python path resolution

## The Problem We Solved

When running MCP servers via Claude Code's ~/.claude.json configuration, we need:
- Scripts to find their project dependencies (fastmcp, pandas, etc.)
- Local imports to work (e.g., selenium-browser-automation importing from src/)
- No hardcoded paths or complex --project flags

Without this wrapper, UV adds only the script's immediate directory to Python's path,
not the project root. So a script at `project/src/server.py` can't import from
`project/src/helpers.py` using `from src.helpers import foo`.

## Related Issues

Multiple open GitHub issues request this feature:
- https://github.com/astral-sh/uv/issues/11302 - "uv run doesn't find project from script path"
- https://github.com/astral-sh/uv/issues/14585 - "Running scripts from outside project doesn't work"
- https://github.com/astral-sh/uv/issues/12193 - "Script-relative project discovery needed"

The community consensus: UV's current behavior is counterintuitive and should be fixed.

## Usage

In Claude's ~/.claude.json (replace /path/to/claude-workspace with your actual path):
```json
{
  "mcpServers": {
    "python-interpreter": {
      "command": "/path/to/claude-workspace/uv-wrapper.py",
      "args": ["/path/to/claude-workspace/mcp/python-interpreter/server.py"]
    }
  }
}
```

This replaces the complex workaround of:
```json
"args": ["run", "--project", "/path/to/project", "python", "/path/to/script.py"]
```

## Technical Details

- Signal forwarding ensures graceful shutdown (Ctrl-C works properly)
- Exit codes are preserved from the child process
- stdin/stdout/stderr are inherited (interactive scripts work)
- PYTHONPATH=. ensures the project root is on Python's path
"""

import os
import signal
import subprocess
import sys
from pathlib import Path


def find_project_root(script_path: Path) -> Path:
    """Walk up from script path to find pyproject.toml."""
    current = script_path.parent

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError(f"No pyproject.toml found above {script_path}")


def main() -> None:
    """Run script with project-relative discovery and PYTHONPATH."""
    if len(sys.argv) < 2:
        sys.exit("Usage: uv-wrapper.py /path/to/script.py [args...]")

    script_path = Path(sys.argv[1]).resolve()
    script_args = sys.argv[2:]  # Pass through any additional arguments

    project_root = find_project_root(script_path)
    relative_script = script_path.relative_to(project_root)

    # Run with PYTHONPATH=. from project directory, with signal forwarding
    with subprocess.Popen(
        ["uv", "run", "python", str(relative_script), *script_args],
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": ".",}
    ) as proc:
        # Forward signals to child process for graceful shutdown
        def forward_signal(sig, frame):
            proc.send_signal(sig)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, forward_signal)

        # Wait and pass through exit code
        sys.exit(proc.wait())


if __name__ == "__main__":
    main()