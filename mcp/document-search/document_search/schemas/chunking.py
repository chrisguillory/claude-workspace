"""Chunking operation schemas.

Typed models for document chunking. Defines supported file types and chunk structure.

Future support (TODOs preserved for context):
- HTML: Saved web pages with corollary assets. Parse main content, ignore JS/CSS.
- Video (.mp4, .mov): Construction walkthroughs. Would need transcription or frame extraction.
- SVG: Vector graphics. Could extract text content or describe visually.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import pydantic

from document_search.schemas.base import StrictModel

# Supported file types for document search.
# Code files (.py, .js, .ts, etc.) handled by separate code search system.
type FileType = Literal[
    # Core document types
    'markdown',  # .md - emails, meetings, research
    'text',  # .txt - simple text notes
    'pdf',  # .pdf - contracts, legal, quotes
    'image',  # .jpg, .jpeg, .png, .heic, .gif
    'json',  # .json - structured data (property deeds, configs)
    # Extended types
    'email',  # .eml - original emails with headers
    'csv',  # .csv - tabular data
]

# TODO: Future file types to consider
# - 'html': Saved web pages. ~10 files, mostly Home Depot quotes with JS/CSS assets.
#           Would need to extract main content, ignore supporting assets.
# - 'video': .mp4, .mov - ~12 files. Construction/flooring walkthroughs.
#           Would require transcription service or frame extraction for embedding.
# - 'svg': Vector graphics. ~2 files. Could extract text or use multimodal.

# Extension to FileType mapping
EXTENSION_MAP: dict[str, FileType] = {
    # Markdown
    '.md': 'markdown',
    '.markdown': 'markdown',
    # Text
    '.txt': 'text',
    # PDF
    '.pdf': 'pdf',
    # Images
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.heic': 'image',
    '.gif': 'image',
    # JSON
    '.json': 'json',
    # Email
    '.eml': 'email',
    # CSV
    '.csv': 'csv',
}


def get_file_type(path: Path) -> FileType | None:
    """Get FileType for a path, or None if unsupported."""
    return EXTENSION_MAP.get(path.suffix.lower())


class ChunkMetadata(StrictModel):
    """Position and context info for chunk provenance."""

    start_char: int
    end_char: int
    # Markdown-specific: heading hierarchy for context
    heading_context: str | None = None
    # PDF-specific: page number
    page_number: int | None = None
    # JSON-specific: JSON path to this chunk's data
    json_path: str | None = None


class Chunk(StrictModel):
    """Document chunk ready for embedding.

    Immutable after creation. Contains text content and full provenance
    for citation and retrieval.
    """

    text: Annotated[str, pydantic.Field(min_length=1)]
    source_path: str
    chunk_index: int
    file_type: FileType
    metadata: ChunkMetadata


class ChunkResult(StrictModel):
    """Chunk paired with its embedding vector."""

    chunk: Chunk
    embedding: Sequence[float]
