"""Shared type aliases for claude-session CLI and MCP boundaries."""

from __future__ import annotations

from typing import Literal

__all__ = [
    'GistVisibility',
]

type GistVisibility = Literal['public', 'secret']
