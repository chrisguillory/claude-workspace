"""Chunking service - splits documents into embeddable chunks.

Handles different file types with appropriate chunking strategies:
- Markdown: Structure-aware splitting by headers
- Text: Recursive character splitting
- JSON: Logical structure-based splitting
- PDF, Email, CSV, Image: Stubs for future implementation
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from document_search.schemas.chunking import (
    Chunk,
    ChunkMetadata,
    FileType,
    get_file_type,
)

# Default chunking parameters (research-backed: 400-512 tokens optimal)
DEFAULT_CHUNK_SIZE = 1500  # ~375-500 tokens at 3-4 chars/token
DEFAULT_CHUNK_OVERLAP = 300  # 20% overlap


class ChunkingService:
    """Chunks documents by file type.

    Uses appropriate strategy per file type:
    - Markdown: Header-aware splitting preserves section context
    - Text: Recursive splitting at natural boundaries
    - JSON: Structure-aware splitting for nested data
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize chunking service.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Reusable splitters
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', ' ', ''],
        )

        self._markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'h1'),
                ('##', 'h2'),
                ('###', 'h3'),
                ('####', 'h4'),
            ],
            strip_headers=False,
        )

    def chunk_file(self, path: Path) -> Sequence[Chunk]:
        """Chunk a single file.

        Args:
            path: Path to file.

        Returns:
            Sequence of chunks with metadata.

        Raises:
            ValueError: If file type not supported.
            FileNotFoundError: If file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')

        file_type = get_file_type(path)
        if file_type is None:
            raise ValueError(f'Unsupported file type: {path.suffix}')

        content = path.read_text(encoding='utf-8', errors='replace')
        return self._chunk_content(content, str(path), file_type)

    def chunk_directory(
        self,
        directory: Path,
        *,
        recursive: bool = True,
    ) -> Sequence[Chunk]:
        """Chunk all supported files in a directory.

        Args:
            directory: Directory to scan.
            recursive: If True, scan subdirectories.

        Returns:
            Sequence of all chunks from all files.
        """
        if not directory.is_dir():
            raise ValueError(f'Not a directory: {directory}')

        all_chunks: list[Chunk] = []
        pattern = '**/*' if recursive else '*'

        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            if get_file_type(path) is None:
                continue

            chunks = self.chunk_file(path)
            all_chunks.extend(chunks)

        return all_chunks

    def _chunk_content(
        self,
        content: str,
        source_path: str,
        file_type: FileType,
    ) -> Sequence[Chunk]:
        """Route content to appropriate chunker."""
        match file_type:
            case 'markdown':
                return self._chunk_markdown(content, source_path)
            case 'text':
                return self._chunk_text(content, source_path)
            case 'json':
                return self._chunk_json(content, source_path)
            case 'pdf':
                return self._chunk_pdf(content, source_path)
            case 'image':
                return self._chunk_image(source_path)
            case 'email':
                return self._chunk_email(content, source_path)
            case 'csv':
                return self._chunk_csv(content, source_path)

    def _chunk_markdown(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk markdown with header awareness."""
        # First split by headers to preserve structure
        header_splits = self._markdown_header_splitter.split_text(content)

        chunks: list[Chunk] = []
        char_offset = 0

        for i, doc in enumerate(header_splits):
            # Extract heading context from metadata
            heading_parts = [doc.metadata[level] for level in ['h1', 'h2', 'h3', 'h4'] if level in doc.metadata]
            heading_context = ' > '.join(heading_parts) if heading_parts else None

            # Further split if content is too large
            text = doc.page_content
            if len(text) > self._chunk_size:
                sub_splits = self._text_splitter.split_text(text)
                for j, sub_text in enumerate(sub_splits):
                    chunks.append(
                        Chunk(
                            text=sub_text,
                            source_path=source_path,
                            chunk_index=len(chunks),
                            file_type='markdown',
                            metadata=ChunkMetadata(
                                start_char=char_offset,
                                end_char=char_offset + len(sub_text),
                                heading_context=heading_context,
                            ),
                        )
                    )
                    char_offset += len(sub_text)
            else:
                chunks.append(
                    Chunk(
                        text=text,
                        source_path=source_path,
                        chunk_index=len(chunks),
                        file_type='markdown',
                        metadata=ChunkMetadata(
                            start_char=char_offset,
                            end_char=char_offset + len(text),
                            heading_context=heading_context,
                        ),
                    )
                )
                char_offset += len(text)

        return chunks

    def _chunk_text(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk plain text with recursive splitting."""
        splits = self._text_splitter.split_text(content)

        chunks: list[Chunk] = []
        char_offset = 0

        for i, text in enumerate(splits):
            chunks.append(
                Chunk(
                    text=text,
                    source_path=source_path,
                    chunk_index=i,
                    file_type='text',
                    metadata=ChunkMetadata(
                        start_char=char_offset,
                        end_char=char_offset + len(text),
                    ),
                )
            )
            char_offset += len(text)

        return chunks

    def _chunk_json(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk JSON by logical structure.

        For arrays: Each top-level element becomes a chunk.
        For objects: Stringify and use text chunking.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fall back to text chunking if invalid JSON
            return self._chunk_text(content, source_path)

        chunks: list[Chunk] = []

        if isinstance(data, list):
            # Array: each element is a chunk
            for i, item in enumerate(data):
                text = json.dumps(item, indent=2, ensure_ascii=False)
                if len(text) > self._chunk_size:
                    # Large item: split further
                    sub_chunks = self._chunk_text(text, source_path)
                    for sub_chunk in sub_chunks:
                        # Update with JSON path
                        chunks.append(
                            Chunk(
                                text=sub_chunk.text,
                                source_path=source_path,
                                chunk_index=len(chunks),
                                file_type='json',
                                metadata=ChunkMetadata(
                                    start_char=0,
                                    end_char=len(sub_chunk.text),
                                    json_path=f'[{i}]',
                                ),
                            )
                        )
                else:
                    chunks.append(
                        Chunk(
                            text=text,
                            source_path=source_path,
                            chunk_index=len(chunks),
                            file_type='json',
                            metadata=ChunkMetadata(
                                start_char=0,
                                end_char=len(text),
                                json_path=f'[{i}]',
                            ),
                        )
                    )
        else:
            # Object: stringify and chunk as text
            text = json.dumps(data, indent=2, ensure_ascii=False)
            sub_chunks = self._chunk_text(text, source_path)
            for sub_chunk in sub_chunks:
                chunks.append(
                    Chunk(
                        text=sub_chunk.text,
                        source_path=source_path,
                        chunk_index=len(chunks),
                        file_type='json',
                        metadata=ChunkMetadata(
                            start_char=sub_chunk.metadata.start_char,
                            end_char=sub_chunk.metadata.end_char,
                            json_path='$',
                        ),
                    )
                )

        return chunks

    def _chunk_pdf(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk PDF content.

        TODO: Implement PDF parsing with PyMuPDF or pdfplumber.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            'PDF chunking not yet implemented. Requires PyMuPDF or pdfplumber for text extraction.'
        )

    def _chunk_image(self, source_path: str) -> Sequence[Chunk]:
        """Handle image files.

        Images are not chunked - they're embedded whole via multimodal API.
        Returns a single "chunk" with the file path for reference.

        TODO: Implement multimodal embedding with Gemini vision.
        """
        return [
            Chunk(
                text=f'[Image: {Path(source_path).name}]',
                source_path=source_path,
                chunk_index=0,
                file_type='image',
                metadata=ChunkMetadata(start_char=0, end_char=0),
            )
        ]

    def _chunk_email(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk .eml email files.

        TODO: Implement email parsing with email.parser.
        Should extract headers, body, and attachments.
        """
        raise NotImplementedError(
            'Email chunking not yet implemented. Requires email.parser for header/body extraction.'
        )

    def _chunk_csv(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk CSV files.

        TODO: Implement CSV chunking strategy.
        Options: row-by-row, grouped rows, or full table with context.
        """
        raise NotImplementedError(
            'CSV chunking not yet implemented. Requires strategy decision: row-based vs table-based.'
        )
