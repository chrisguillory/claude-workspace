"""Pluggable HTTP transport for the grok-kit SDK.

Default: ``httpx.Client`` (sync) and ``httpx.AsyncClient`` (async). Swap to
``curl_cffi.requests`` if grok.com starts 403'ing on Cloudflare bot detection
(gpt4free's Grok provider hit this; we have not yet, but the seam is here).
"""

from __future__ import annotations

__all__ = []
