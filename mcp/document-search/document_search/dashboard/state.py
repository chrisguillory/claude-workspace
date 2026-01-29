"""Dashboard state management with file locking."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import filelock

from document_search.paths import DASHBOARD_LOCK_PATH, DASHBOARD_STATE_PATH
from document_search.schemas.dashboard import DashboardState, McpServer

__all__ = [
    'DashboardStateManager',
]


class DashboardStateManager:
    """Manages dashboard state with file locking."""

    def __init__(
        self,
        state_path: Path = DASHBOARD_STATE_PATH,
        lock_path: Path = DASHBOARD_LOCK_PATH,
    ) -> None:
        self._state_path = state_path
        self._lock_path = lock_path
        self._lock = filelock.FileLock(lock_path)

    @contextmanager
    def hold_lock(self) -> Iterator[None]:
        """Context manager to hold lock for external coordination."""
        with self._lock:
            yield

    def load(self) -> DashboardState | None:
        """Load state from file. Returns None if not exists."""
        if not self._state_path.exists():
            return None
        data = json.loads(self._state_path.read_text())
        return DashboardState.model_validate(data)

    def save(self, state: DashboardState) -> None:
        """Save state atomically with lock."""
        with self._lock:
            self._save_unlocked(state)

    def delete(self) -> None:
        """Delete state file."""
        with self._lock:
            self._state_path.unlink(missing_ok=True)

    def get_dashboard_port(self) -> int | None:
        """Return dashboard port if alive, None otherwise."""
        state = self.load()
        if state is None or not _process_exists(state.server_pid):
            return None
        return state.port

    def register_mcp_server(self, pid: int) -> None:
        """Add MCP server to state. Raises ValueError if dashboard not running."""
        with self._lock:
            state = self.load()
            if state is None:
                raise ValueError('No dashboard state - dashboard not running?')

            if any(s.pid == pid for s in state.mcp_servers):
                return

            new_server = McpServer(pid=pid, started_at=datetime.now(UTC))
            new_state = DashboardState(
                port=state.port,
                server_pid=state.server_pid,
                mcp_servers=(*state.mcp_servers, new_server),
            )
            self._save_unlocked(new_state)

    def unregister_mcp_server(self, pid: int) -> None:
        """Remove MCP server from state."""
        with self._lock:
            state = self.load()
            if state is None:
                return

            new_servers = tuple(s for s in state.mcp_servers if s.pid != pid)
            if len(new_servers) != len(state.mcp_servers):
                new_state = DashboardState(
                    port=state.port,
                    server_pid=state.server_pid,
                    mcp_servers=new_servers,
                )
                self._save_unlocked(new_state)

    def get_live_mcp_servers(self) -> tuple[McpServer, ...]:
        """Return MCP servers whose processes are still running."""
        with self._lock:
            state = self.load()
            if state is None:
                return ()
            return tuple(s for s in state.mcp_servers if _process_exists(s.pid))

    def _save_unlocked(self, state: DashboardState) -> None:
        """Save without acquiring lock. Caller must hold lock."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._state_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(state.model_dump(mode='json'), indent=2) + '\n')
        temp_path.rename(self._state_path)


def _process_exists(pid: int) -> bool:
    """Check if process exists via signal 0."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Exists but no permission
