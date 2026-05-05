"""Shared type aliases for claude-session CLI and MCP boundaries."""

from __future__ import annotations

from typing import Literal

__all__ = [
    'ArchiveFormat',
    'GistVisibility',
]

type ArchiveFormat = Literal['json', 'zst']
type GistVisibility = Literal['public', 'secret']
