"""Cookie bootstrap and refresh for grok.com.

Bootstrap: extract X SSO cookies from the user's Chrome profile via the
selenium-browser MCP's save_profile_state path. Refresh: detect 401/403 from
SDK calls and re-bootstrap.

Five load-bearing cookies: sso, sso-rw, x-userid, cf_clearance, __cf_bm.
The SDK consumes them as a single Cookie header value (Speakeasy Python
runtime doesn't support apiKey in: cookie schemes, so the spec models them
as one apiKey in: header name: Cookie scheme; we format here).
"""

from __future__ import annotations

__all__ = []
