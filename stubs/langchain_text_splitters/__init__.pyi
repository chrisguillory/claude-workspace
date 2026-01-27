"""Type stubs for langchain-text-splitters.

Coverage: MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

class Document:
    """LangChain document with content and metadata."""

    page_content: str
    metadata: dict[str, Any]

class RecursiveCharacterTextSplitter:
    """Splits text recursively at natural boundaries."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def split_text(self, text: str) -> list[str]: ...
    def split_documents(self, documents: Sequence[Document]) -> list[Document]: ...
    def __getattr__(self, name: str) -> Any: ...

class MarkdownHeaderTextSplitter:
    """Splits markdown by headers, preserving hierarchy in metadata."""

    def __init__(
        self,
        headers_to_split_on: Sequence[tuple[str, str]],
        strip_headers: bool = True,
        **kwargs: Any,
    ) -> None: ...
    def split_text(self, text: str) -> list[Document]: ...
    def __getattr__(self, name: str) -> Any: ...
