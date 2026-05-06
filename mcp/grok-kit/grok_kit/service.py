"""Business logic over the Speakeasy-generated grok-kit-sdk.

Combines list/detail/tree/load-responses calls into higher-level operations:
syncing a conversation, paginating the full list, walking the message tree.
Tools layer (mcp/, cli/) calls these methods, not the SDK directly.
"""

from __future__ import annotations

__all__ = []
