"""Type stubs for markdownify library (v1.2.2).

Partial: only the markdownify() entry point; converter classes unstubbed.
"""

from typing import Any

def markdownify(html: str, **options: Any) -> str: ...
def __getattr__(name: str) -> Any: ...
