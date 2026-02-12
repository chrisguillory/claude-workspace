"""Interpreter configuration registry with atomic file persistence.

Manages saved interpreter configurations in ~/.claude-workspace/python_interpreter/interpreters.json.
Uses file locking for cross-process safety (multiple MCP servers may share a project).
"""

from __future__ import annotations

__all__ = [
    'InterpreterRegistryManager',
]

import json
import pathlib
from collections.abc import Mapping

import filelock

from python_interpreter.models import InterpreterRegistry, SavedInterpreterConfig


class InterpreterRegistryManager:
    """Manages saved interpreter configurations with atomic file persistence."""

    STATE_DIR = pathlib.Path.home() / '.claude-workspace' / 'python_interpreter'

    def __init__(self) -> None:
        self._state_path = self.STATE_DIR / 'interpreters.json'
        self._lock = filelock.FileLock(self._state_path.with_suffix('.lock'))

    def load(self) -> InterpreterRegistry:
        """Load registry from file. Returns empty registry if not exists."""
        if not self._state_path.exists():
            return InterpreterRegistry(discover_jetbrains=True, interpreters={})
        data = json.loads(self._state_path.read_text())
        # Handle backwards compat: old files may have discover_pycharm
        if 'discover_pycharm' in data and 'discover_jetbrains' not in data:
            data['discover_jetbrains'] = data.pop('discover_pycharm')
        return InterpreterRegistry.model_validate(data)

    def save(self, registry: InterpreterRegistry) -> None:
        """Save registry atomically (write tmp + rename) under file lock."""
        with self._lock:
            self._save_unlocked(registry)

    def save_interpreter(self, name: str, config: SavedInterpreterConfig) -> None:
        """Add or update a saved interpreter config."""
        with self._lock:
            registry = self.load()
            new_interpreters = dict(registry.interpreters)
            new_interpreters[name] = config
            self._save_unlocked(
                InterpreterRegistry(
                    discover_jetbrains=registry.discover_jetbrains,
                    interpreters=new_interpreters,
                )
            )

    def remove_interpreter(self, name: str) -> bool:
        """Remove a saved interpreter. Returns True if removed, False if not found."""
        with self._lock:
            registry = self.load()
            if name not in registry.interpreters:
                return False
            new_interpreters = {k: v for k, v in registry.interpreters.items() if k != name}
            self._save_unlocked(
                InterpreterRegistry(
                    discover_jetbrains=registry.discover_jetbrains,
                    interpreters=new_interpreters,
                )
            )
            return True

    def get(self, name: str) -> SavedInterpreterConfig | None:
        """Get a saved interpreter config by name."""
        return self.load().interpreters.get(name)

    def list_saved(self) -> Mapping[str, SavedInterpreterConfig]:
        """List all saved interpreter configs."""
        return self.load().interpreters

    def _save_unlocked(self, registry: InterpreterRegistry) -> None:
        """Atomic write without acquiring lock (caller must hold lock)."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._state_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(registry.model_dump(mode='json'), indent=2) + '\n')
        temp_path.rename(self._state_path)
