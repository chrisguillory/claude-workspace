"""granola-mcp auth helpers (~/granola-mcp/granola_mcp/helpers.py).

Imported via sys.path injection in scripts/sync-granola-context.py — granola-mcp
is the single source of Granola auth truth (token minting + Electron identity
headers).
"""

from collections.abc import Mapping
from typing import Any

def get_auth_headers() -> Mapping[str, str]: ...
def __getattr__(name: str) -> Any: ...
