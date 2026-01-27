"""Type stubs for google-genai library.

Coverage: Client class and embed_content method only.
Unmapped functionality handled via __getattr__.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

class ContentEmbedding:
    """Single embedding result from API."""

    @property
    def values(self) -> Sequence[float]: ...

class EmbedContentResponse:
    """Response from embed_content API."""

    @property
    def embeddings(self) -> Sequence[ContentEmbedding]: ...

class Models:
    """Models API interface."""

    def embed_content(
        self,
        *,
        model: str,
        contents: str | Sequence[str],
        config: Mapping[str, Any] | None = None,
    ) -> EmbedContentResponse: ...
    def __getattr__(self, name: str) -> Any: ...

class Client:
    """Gemini API client."""

    models: Models

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_options: Any | None = None,
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
