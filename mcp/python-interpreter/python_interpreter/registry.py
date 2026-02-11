"""Interpreter configuration registry with atomic file persistence.

Manages saved interpreter configurations in {project_dir}/.claude/interpreters.json.
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

    def __init__(self, project_dir: pathlib.Path) -> None:
        self._state_path = project_dir / '.claude' / 'interpreters.json'
        self._lock = filelock.FileLock(self._state_path.with_suffix('.lock'))

    def load(self) -> InterpreterRegistry:
        """Load registry from file. Returns empty registry if not exists."""
        if not self._state_path.exists():
            return InterpreterRegistry(discover_pycharm=True, interpreters={})
        data = json.loads(self._state_path.read_text())
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
                    discover_pycharm=registry.discover_pycharm,
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
                    discover_pycharm=registry.discover_pycharm,
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
