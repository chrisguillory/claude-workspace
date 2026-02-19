"""Chunking service - splits documents into embeddable chunks.

Handles different file types with appropriate chunking strategies:
- Markdown: Structure-aware splitting by headers
- Text: Recursive character splitting
- JSON: Logical structure-based splitting
- JSONL: Streaming line-by-line with adaptive grouping
- PDF: Page-aware semantic chunking with ProcessPoolExecutor
- CSV: Context-aware row grouping with header context

PDF and JSONL processing use ProcessPoolExecutor to bypass the GIL for
CPU-bound operations (4x speedup for JSONL, significant gains for PDF).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
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
from document_search.schemas.embeddings import MAX_TEXT_CHARS
from document_search.services.chunking_worker import chunk_jsonl
from document_search.services.pdf_extraction import extract_pdf

__all__ = [
    'ChunkingService',
]

logger = logging.getLogger(__name__)

# Default chunking parameters (research-backed: 400-512 tokens optimal)
DEFAULT_CHUNK_SIZE = 1500  # ~375-500 tokens at 3-4 chars/token
DEFAULT_CHUNK_OVERLAP = 300  # 20% overlap

# CSV processing constants
CSV_MIN_GROUP_SIZE = 5  # Minimum rows per chunk
CSV_MAX_GROUP_SIZE = 15  # Maximum rows per chunk
CSV_CHUNK_SIZE_TARGET = 1000  # Target chunk size for adaptive grouping
CSV_ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']

# Chunk filtering (applied to all file types)
MIN_CHUNK_LENGTH = 50  # Characters - filters boilerplate like page footers


class ChunkingService:
    """Chunks documents by file type with async processing for I/O-bound operations.

    Uses appropriate strategy per file type:
    - Markdown: Header-aware splitting preserves section context
    - Text: Recursive splitting at natural boundaries
    - JSON: Structure-aware splitting for nested data
    - JSONL: Streaming line-by-line with adaptive grouping
    - PDF: Page-aware semantic chunking with parallel processing
    - CSV: Context-aware row grouping with header context
    """

    @classmethod
    async def create(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        process_workers: int | None = None,
    ) -> ChunkingService:
        """Create chunking service in async context.

        ProcessPoolExecutor is created lazily on first JSONL/PDF chunk
        request, so no subprocess workers are spawned until indexing runs.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            process_workers: Max workers for ProcessPoolExecutor.
                Defaults to cpu_count.
        """
        if process_workers is None:
            process_workers = os.cpu_count() or 4

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_workers=process_workers,
        )

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        max_workers: int,
    ) -> None:
        """Initialize chunking service. Use create() for async context safety.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            max_workers: Max workers for lazy ProcessPoolExecutor creation.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_workers = max_workers
        self._process_pool: ProcessPoolExecutor | None = None

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

    async def __aenter__(self) -> ChunkingService:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Async context manager exit - shutdown ProcessPoolExecutor."""
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the ProcessPoolExecutor if it was created.

        No-op if no JSONL/PDF files were ever chunked (pool never started).
        """
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None

    async def chunk_file(self, path: Path) -> Sequence[Chunk]:
        """Chunk a single file based on its type.

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

        chunks = await self._chunk_by_type(path, file_type)
        return self._filter_short_chunks(chunks)

    # --- Private ---

    @property
    def _pool(self) -> ProcessPoolExecutor:
        """Lazy ProcessPoolExecutor — created on first use."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self._max_workers)
        return self._process_pool

    def _filter_short_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Filter out chunks below minimum length threshold."""
        return [c for c in chunks if len(c.text.strip()) >= MIN_CHUNK_LENGTH]

    async def _chunk_by_type(self, path: Path, file_type: FileType) -> Sequence[Chunk]:
        """Route to appropriate chunker based on file type."""
        match file_type:
            case 'pdf':
                return await self._chunk_pdf(path)
            case 'csv':
                return await self._chunk_csv(path)
            case 'jsonl':
                return await self._chunk_jsonl(path)
            case 'markdown' | 'text' | 'json' | 'image' | 'email':
                # These are fast enough to run in thread pool
                return await asyncio.to_thread(
                    self._chunk_content_sync_from_file,
                    path,
                    file_type,
                )
            case _:
                raise ValueError(f'Unsupported file type: {file_type}')

    def _chunk_content_sync_from_file(self, path: Path, file_type: FileType) -> Sequence[Chunk]:
        """Load file and chunk synchronously."""
        content = path.read_text(encoding='utf-8', errors='replace')
        return self._chunk_content_sync(content, str(path), file_type)

    def _chunk_content_sync(self, content: str, source_path: str, file_type: FileType) -> Sequence[Chunk]:
        """Route content to appropriate synchronous chunker."""
        match file_type:
            case 'markdown':
                return self._chunk_markdown(content, source_path)
            case 'text':
                return self._chunk_text(content, source_path)
            case 'json':
                return self._chunk_json(content, source_path)
            case 'image':
                return self._chunk_image(source_path)
            case 'email':
                return self._chunk_email(content, source_path)
            case _:
                raise ValueError(f'Unsupported file type: {file_type}')

    # =========================================================================
    # PDF Chunking Implementation
    # =========================================================================

    async def _chunk_pdf(self, path: Path) -> Sequence[Chunk]:
        """Chunk PDF using page-aware semantic approach with ProcessPoolExecutor.

        Uses ProcessPoolExecutor for CPU-bound PyMuPDF/pdfplumber work.
        This bypasses the GIL for ~4x speedup on multi-core systems.

        The extraction runs in a subprocess, returning structured data.
        Chunking (text splitting) happens in the main process.
        """
        # Run CPU-heavy extraction in subprocess
        start = time.perf_counter()
        logger.debug(f'[PDF] Starting: {path.name}')
        loop = asyncio.get_running_loop()
        extraction = await loop.run_in_executor(self._pool, extract_pdf, str(path))
        elapsed = time.perf_counter() - start

        if extraction.page_count == 0:
            logger.warning(f'PDF has no pages: {path}')
            return []

        logger.debug(f'[PDF] {path.name}: {extraction.page_count} pages, type={extraction.pdf_type}, {elapsed:.2f}s')

        # Build chunks from extracted pages (lightweight, main process)
        chunks: list[Chunk] = []
        for page_data in extraction.pages:
            if not page_data.text.strip():
                continue

            # Get heading context from bookmarks
            heading_context = extraction.bookmarks.get(page_data.page_num)

            # Combine text with table markdown if present
            text = page_data.text
            if page_data.table_markdown:
                text = f'{text}\n\n{page_data.table_markdown}'

            # Sub-chunk if needed
            page_chunks = self._text_splitter.split_text(text)

            for chunk_text in page_chunks:
                metadata = ChunkMetadata(
                    start_char=0,
                    end_char=len(chunk_text),
                    page_number=page_data.page_num + 1,  # 1-indexed
                    heading_context=heading_context,
                )
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source_path=str(path),
                        chunk_index=len(chunks),
                        file_type='pdf',
                        metadata=metadata,
                    )
                )

        return chunks

    # =========================================================================
    # CSV Chunking Implementation
    # =========================================================================

    async def _chunk_csv(self, path: Path) -> Sequence[Chunk]:
        """Chunk CSV using context-aware row grouping.

        Strategy:
        - Group 5-15 rows per chunk (adaptive based on column count)
        - Include header context and detected types
        - Handle encoding variations robustly
        """
        return await asyncio.to_thread(self._chunk_csv_sync, path)

    def _chunk_csv_sync(self, path: Path) -> Sequence[Chunk]:
        """Synchronous CSV chunking implementation."""
        chunks: list[Chunk] = []

        # Try multiple encodings
        df = None
        for encoding in CSV_ENCODINGS:
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    dtype=str,
                    on_bad_lines='skip',
                )
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        if df is None:
            logger.error(f'Could not parse CSV with any encoding: {path}')
            return chunks

        if df.empty:
            logger.warning(f'Empty CSV: {path}')
            return chunks

        # Prepare data
        df = df.fillna('')
        headers = list(df.columns)

        # Detect types for context
        type_map = self._detect_csv_types(df)
        type_summary = ', '.join(f'{col} ({typ})' for col, typ in type_map.items())

        # File context from name
        file_context = path.stem.replace('_', ' ').replace('-', ' ')

        # Adaptive group size based on column count and chunk size target
        avg_row_len = sum(len(str(v)) for v in df.iloc[0]) if len(df) > 0 else 50
        estimated_row_with_headers = avg_row_len + len(headers) * 5
        group_size = max(
            CSV_MIN_GROUP_SIZE,
            min(CSV_MAX_GROUP_SIZE, CSV_CHUNK_SIZE_TARGET // max(estimated_row_with_headers, 1)),
        )

        # Process in groups
        for group_idx in range(0, len(df), group_size):
            group_df = df.iloc[group_idx : group_idx + group_size]
            if group_df.empty:
                continue

            # Build chunk text with rich context
            chunk_lines = [
                f'Source: {file_context}',
                f'Columns: {", ".join(headers)}',
                f'Types: {type_summary}',
                f'Total rows: {len(df)}, showing rows {group_idx + 1}-{group_idx + len(group_df)}:',
                '',
            ]

            for _, row in group_df.iterrows():
                row_str = ' | '.join(f'{col}: {val}' for col, val in zip(headers, row))
                chunk_lines.append(row_str)

            chunk_text = '\n'.join(chunk_lines)
            page_number = group_idx // group_size + 1

            # Sub-chunk if exceeds embedding API limit (e.g., geo_json fields)
            if len(chunk_text) > MAX_TEXT_CHARS:
                sub_texts = self._text_splitter.split_text(chunk_text)
                for sub_text in sub_texts:
                    metadata = ChunkMetadata(
                        start_char=0,
                        end_char=len(sub_text),
                        page_number=page_number,
                        heading_context=file_context,
                    )
                    chunks.append(
                        Chunk(
                            text=sub_text,
                            source_path=str(path),
                            chunk_index=len(chunks),
                            file_type='csv',
                            metadata=metadata,
                        )
                    )
            else:
                metadata = ChunkMetadata(
                    start_char=0,
                    end_char=len(chunk_text),
                    page_number=page_number,
                    heading_context=file_context,
                )
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source_path=str(path),
                        chunk_index=len(chunks),
                        file_type='csv',
                        metadata=metadata,
                    )
                )

        return chunks

    def _detect_csv_types(self, df: pd.DataFrame) -> Mapping[str, str]:
        """Detect likely data types for CSV columns."""
        type_map: dict[str, str] = {}

        for column in df.columns:
            sample = df[column].dropna().head(10)
            if sample.empty:
                type_map[column] = 'unknown'
                continue

            first_val = str(sample.iloc[0])

            # Date patterns
            if re.match(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}', first_val):
                type_map[column] = 'date'
            # Currency/price
            elif re.match(r'^\$?-?[\d,]+\.?\d*$', first_val):
                type_map[column] = 'numeric'
            # Boolean
            elif first_val.lower() in ('true', 'false', 'yes', 'no', 'y', 'n'):
                type_map[column] = 'boolean'
            else:
                type_map[column] = 'text'

        return type_map

    # =========================================================================
    # Existing Chunking Implementations (Markdown, Text, JSON, etc.)
    # =========================================================================

    def _chunk_markdown(self, content: str, source_path: str) -> Sequence[Chunk]:
        """Chunk markdown with header awareness."""
        # First split by headers to preserve structure
        header_splits = self._markdown_header_splitter.split_text(content)

        chunks: list[Chunk] = []
        char_offset = 0

        for doc in header_splits:
            # Extract heading context from metadata
            heading_parts = [doc.metadata[level] for level in ['h1', 'h2', 'h3', 'h4'] if level in doc.metadata]
            heading_context = ' > '.join(heading_parts) if heading_parts else None

            # Further split if content is too large
            text = doc.page_content
            if len(text) > self._chunk_size:
                sub_splits = self._text_splitter.split_text(text)
                for sub_text in sub_splits:
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

    def _chunk_image(self, source_path: str) -> Sequence[Chunk]:
        """Handle image files.

        Images are not chunked - they're embedded whole via multimodal API.
        Returns a single "chunk" with the file path for reference.
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
        """Chunk .eml email files by extracting meaningful content.

        Parses MIME structure to extract:
        - Subject, From, To, Date headers
        - Plain text body (preferred) or HTML body (fallback)
        - Skips: MIME boundaries, base64 attachments, Content-Type headers
        """
        import email
        from email.policy import default

        msg = email.message_from_string(content, policy=default)

        # Extract headers worth indexing
        parts: list[str] = []
        for header in ('Subject', 'From', 'To', 'Date'):
            value = msg.get(header)
            if value:
                parts.append(f'{header}: {value}')

        # Extract body content
        if msg.is_multipart():
            # Try plain text first
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_content()
                    if isinstance(body, str):
                        parts.append(body)
                        break
            else:
                # No plain text found, try HTML
                for part in msg.walk():
                    if part.get_content_type() == 'text/html':
                        body = part.get_content()
                        if isinstance(body, str):
                            # Basic HTML tag stripping
                            text = re.sub(r'<[^>]+>', ' ', body)
                            text = re.sub(r'\s+', ' ', text).strip()
                            parts.append(text)
                            break
        else:
            body = msg.get_content()
            if isinstance(body, str):
                parts.append(body)

        # Chunk the extracted content
        email_text = '\n\n'.join(parts)
        if not email_text.strip():
            return []

        return self._chunk_text(email_text, source_path)

    # =========================================================================
    # JSONL Chunking Implementation
    # =========================================================================

    async def _chunk_jsonl(self, path: Path) -> Sequence[Chunk]:
        """Chunk JSONL using ProcessPoolExecutor for 4x speedup.

        Uses subprocess worker to bypass GIL for CPU-bound JSON parsing.
        """
        loop = asyncio.get_running_loop()
        chunk_data = await loop.run_in_executor(
            self._pool, chunk_jsonl, str(path), self._chunk_size, self._chunk_overlap
        )

        # Convert ChunkData back to Chunk objects
        return [
            Chunk(
                text=cd['text'],
                source_path=cd['source_path'],
                chunk_index=cd['chunk_index'],
                file_type=cd['file_type'],
                metadata=ChunkMetadata.model_validate(cd['metadata']),
            )
            for cd in chunk_data
        ]
