from __future__ import annotations

__all__ = [
    'ServerState',
]

from dataclasses import dataclass

from granola_kit.services.meetings import MeetingService


@dataclass(frozen=True)
class ServerState:
    """Long-lived services the MCP tools share across calls."""

    meetings: MeetingService
