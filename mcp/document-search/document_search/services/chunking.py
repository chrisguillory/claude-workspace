"""Chunking service - splits documents into embeddable chunks.

Handles different file types with appropriate chunking strategies:
- Markdown: Structure-aware splitting by headers
- Text: Recursive character splitting
- JSON: Logical structure-based splitting
- JSONL: Streaming line-by-line with adaptive grouping
- PDF: Page-aware semantic chunking with ProcessPoolExecutor
- CSV: Context-aware row grouping with header context

PDF processing uses ProcessPoolExecutor to bypass the GIL for CPU-bound
PyMuPDF/pdfplumber operations (see pdf_extraction.py for subprocess work).
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

# JSONL processing constants
JSONL_ENCODINGS = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']
JSONL_SAMPLE_SIZE = 200  # Records to sample for strategy decision
JSONL_GROUPING_THRESHOLD = 800  # Below this median size, group records
JSONL_MAX_GROUP_SIZE = 15  # Maximum records per grouped chunk
JSONL_BASE64_MIN_LENGTH = 500  # Minimum length to check for base64 content

# Chunk filtering
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
        pdf_process_workers: int | None = None,
    ) -> ChunkingService:
        """Create chunking service in async context.

        Preferred factory method - ensures proper initialization.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            pdf_process_workers: Max workers for PDF ProcessPoolExecutor.
                Defaults to cpu_count (main process only does async I/O).
        """
        if pdf_process_workers is None:
            # Use all CPU cores - main process only does async I/O, not CPU work
            pdf_process_workers = os.cpu_count() or 4

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            _process_pool=ProcessPoolExecutor(max_workers=pdf_process_workers),
        )

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        _process_pool: ProcessPoolExecutor,
    ) -> None:
        """Initialize chunking service. Use create() for async context safety.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            _process_pool: ProcessPoolExecutor for CPU-bound PDF work.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._process_pool = _process_pool

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

    def shutdown(self) -> None:
        """Shutdown the ProcessPoolExecutor and release resources.

        Should be called when the service is no longer needed.
        Also called automatically when used as a context manager.
        """
        self._process_pool.shutdown(wait=True)

    async def __aenter__(self) -> ChunkingService:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Async context manager exit - shutdown ProcessPoolExecutor."""
        self.shutdown()

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
        extraction = await loop.run_in_executor(self._process_pool, extract_pdf, str(path))
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
        import pandas as pd

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

            metadata = ChunkMetadata(
                start_char=0,
                end_char=len(chunk_text),
                # Use group index as pseudo-page for reference
                page_number=group_idx // group_size + 1,
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
        """Chunk JSONL using streaming line-by-line with adaptive grouping.

        Strategy:
        - Stream file line-by-line to handle large files efficiently
        - Sample first 200 records to determine grouping strategy
        - Small records (median < 800 chars): Group 5-15 records per chunk
        - Large records: One record per chunk, sub-chunk if > chunk_size
        - Detect schema hint from common fields for context
        """
        return await asyncio.to_thread(self._chunk_jsonl_sync, path)

    def _chunk_jsonl_sync(self, path: Path) -> Sequence[Chunk]:
        """Synchronous JSONL chunking implementation."""
        chunks: list[Chunk] = []

        # Detect encoding
        encoding = self._detect_jsonl_encoding(path)

        # Error tracking (use typed variables instead of dict)
        total_lines = 0
        malformed_lines = 0
        sample_errors: list[str] = []

        # Sampling buffers
        sample_buffer: list[tuple[int, object, int]] = []  # (line_num, obj, size)
        sample_objects: list[dict[str, object]] = []

        # Strategy decision
        strategy_decided = False
        should_group = False
        schema_hint: str | None = None

        # Grouping state
        group_buffer: list[tuple[int, object]] = []
        group_size_chars = 0

        with open(path, encoding=encoding, errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                line = line.strip()
                if not line:
                    continue

                # Parse JSON
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    malformed_lines += 1
                    if len(sample_errors) < 5:
                        sample_errors.append(f'Line {line_num}: {e}')
                    continue

                # Use raw line size for grouping decisions. Chunks contain filtered content
                # (base64 replaced with placeholders), so actual chunks may be smaller than
                # chunk_size target. This is intentionally conservative — slightly smaller
                # chunks are safer for embedding API token limits than risking overflow.
                size = len(line)

                # Phase 1: Sampling
                if not strategy_decided:
                    sample_buffer.append((line_num, obj, size))
                    if isinstance(obj, dict):
                        sample_objects.append(obj)

                    if len(sample_buffer) >= JSONL_SAMPLE_SIZE:
                        # Decide strategy based on samples
                        sizes = [s for _, _, s in sample_buffer]
                        median_size = sorted(sizes)[len(sizes) // 2]
                        should_group = median_size < JSONL_GROUPING_THRESHOLD
                        schema_hint = self._detect_jsonl_schema_hint(sample_objects[:50])
                        strategy_decided = True

                        # Process buffered samples
                        for buf_line, buf_obj, buf_size in sample_buffer:
                            if should_group:
                                group_buffer, group_size_chars = self._jsonl_update_group(
                                    buf_line,
                                    buf_obj,
                                    buf_size,
                                    path,
                                    schema_hint,
                                    group_buffer,
                                    group_size_chars,
                                    chunks,
                                )
                            else:
                                self._jsonl_add_single_record(
                                    buf_line,
                                    buf_obj,
                                    path,
                                    schema_hint,
                                    chunks,
                                )
                        sample_buffer = []
                    continue

                # Phase 2: Process with decided strategy
                if should_group:
                    group_buffer, group_size_chars = self._jsonl_update_group(
                        line_num,
                        obj,
                        size,
                        path,
                        schema_hint,
                        group_buffer,
                        group_size_chars,
                        chunks,
                    )
                else:
                    self._jsonl_add_single_record(
                        line_num,
                        obj,
                        path,
                        schema_hint,
                        chunks,
                    )

        # Handle small files (< sample size)
        if not strategy_decided and sample_buffer:
            sizes = [s for _, _, s in sample_buffer]
            median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
            should_group = median_size < JSONL_GROUPING_THRESHOLD
            schema_hint = self._detect_jsonl_schema_hint(sample_objects[:50])

            for buf_line, buf_obj, buf_size in sample_buffer:
                if should_group:
                    group_buffer, group_size_chars = self._jsonl_update_group(
                        buf_line,
                        buf_obj,
                        buf_size,
                        path,
                        schema_hint,
                        group_buffer,
                        group_size_chars,
                        chunks,
                    )
                else:
                    self._jsonl_add_single_record(
                        buf_line,
                        buf_obj,
                        path,
                        schema_hint,
                        chunks,
                    )

        # Flush remaining group
        if group_buffer:
            self._jsonl_flush_group(group_buffer, path, schema_hint, chunks)

        # Assign sequential chunk indices
        for i, chunk in enumerate(chunks):
            chunks[i] = Chunk(
                text=chunk.text,
                source_path=chunk.source_path,
                chunk_index=i,
                file_type=chunk.file_type,
                metadata=chunk.metadata,
            )

        # Log error summary with sample errors for debugging
        if malformed_lines > 0 and total_lines > 0:
            error_rate = malformed_lines / total_lines * 100
            samples_str = '; '.join(sample_errors) if sample_errors else ''
            logger.warning(
                f'JSONL {path.name}: {malformed_lines} malformed lines ({error_rate:.2f}%) of {total_lines} total'
                + (f'. Samples: {samples_str}' if samples_str else '')
            )

        return chunks

    def _detect_jsonl_encoding(self, path: Path) -> str:
        """Detect file encoding by checking BOM first, then trying encodings."""
        with open(path, 'rb') as f:
            sample = f.read(8192)

        if not sample:
            return 'utf-8'

        # Check for BOMs first - they definitively indicate encoding
        if sample.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        if sample.startswith(b'\xff\xfe'):
            return 'utf-16-le'
        if sample.startswith(b'\xfe\xff'):
            return 'utf-16-be'

        # Try encodings in order
        for encoding in JSONL_ENCODINGS:
            try:
                sample.decode(encoding)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        return 'utf-8'  # Fallback with error replacement

    def _detect_jsonl_schema_hint(
        self,
        objects: Sequence[dict[str, object]],  # strict_typing_linter.py: mutable-type (need keys())
    ) -> str | None:
        """Detect common fields across JSONL records for context."""
        if not objects:
            return None

        # Find keys present in all objects
        common_keys = set(objects[0].keys())
        for obj in objects[1:]:
            common_keys &= set(obj.keys())

        if not common_keys:
            # Fallback: keys in 70%+ of objects
            key_counts: dict[str, int] = {}
            for obj in objects:
                for key in obj:
                    key_counts[key] = key_counts.get(key, 0) + 1

            threshold = len(objects) * 0.7
            common_keys = {k for k, v in key_counts.items() if v >= threshold}

        if common_keys:
            return ', '.join(sorted(common_keys)[:5])
        return None

    def _is_likely_base64(self, value: str) -> bool:
        """Detect if a string value is likely base64-encoded binary."""
        if len(value) < JSONL_BASE64_MIN_LENGTH:
            return False

        # Base64 characteristics: length divisible by 4, specific charset
        if len(value) % 4 != 0:
            return False

        # Check for base64 alphabet (quick heuristic on sample)
        sample = value[:100]
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        return all(c in base64_chars for c in sample)

    def _filter_base64_fields(self, obj: object) -> object:
        """Recursively replace base64 content with placeholders.

        Prevents useless chunking of embedded binary data (images, files)
        while preserving record structure and non-binary content.
        """
        if isinstance(obj, dict):
            result: dict[str, object] = {}
            for k, v in obj.items():
                if isinstance(v, str) and self._is_likely_base64(v):
                    result[k] = f'[base64 content, {len(v)} bytes]'
                else:
                    result[k] = self._filter_base64_fields(v)
            return result
        elif isinstance(obj, list):
            return [self._filter_base64_fields(item) for item in obj]
        else:
            return obj

    def _jsonl_update_group(  # strict_typing_linter.py: mutable-type (returns mutated buffer)
        self,
        line_num: int,
        obj: object,
        size: int,
        path: Path,
        schema_hint: str | None,
        group_buffer: list[tuple[int, object]],  # strict_typing_linter.py: mutable-type (mutates via append)
        group_size_chars: int,
        chunks: list[Chunk],  # strict_typing_linter.py: mutable-type (mutates via append)
    ) -> tuple[list[tuple[int, object]], int]:
        """Update grouping state, flushing group when full."""
        # Check if adding this record would exceed limits
        if group_buffer and (group_size_chars + size > self._chunk_size or len(group_buffer) >= JSONL_MAX_GROUP_SIZE):
            # Flush current group
            self._jsonl_flush_group(group_buffer, path, schema_hint, chunks)
            group_buffer = []
            group_size_chars = 0

        # Add to group
        group_buffer.append((line_num, obj))
        group_size_chars += size

        return group_buffer, group_size_chars

    def _jsonl_flush_group(
        self,
        group_buffer: Sequence[tuple[int, object]],
        path: Path,
        schema_hint: str | None,
        chunks: list[Chunk],  # strict_typing_linter.py: mutable-type (appends to chunks)
    ) -> None:
        """Create a chunk from grouped JSONL records."""
        if not group_buffer:
            return

        line_start = group_buffer[0][0]
        line_end = group_buffer[-1][0]

        # Build text: one JSON object per line, filtering base64 content
        lines = [json.dumps(self._filter_base64_fields(obj), ensure_ascii=False) for _, obj in group_buffer]
        text = '\n'.join(lines)

        metadata = ChunkMetadata(
            start_char=0,
            end_char=len(text),
            jsonl_line_start=line_start,
            jsonl_line_end=line_end,
            jsonl_record_count=len(group_buffer),
            jsonl_schema_hint=schema_hint,
        )

        chunks.append(
            Chunk(
                text=text,
                source_path=str(path),
                chunk_index=0,  # Will be reassigned later
                file_type='jsonl',
                metadata=metadata,
            )
        )

    def _jsonl_add_single_record(
        self,
        line_num: int,
        obj: object,
        path: Path,
        schema_hint: str | None,
        chunks: list[Chunk],  # strict_typing_linter.py: mutable-type (appends to chunks)
    ) -> None:
        """Add a single JSONL record as chunk(s), sub-chunking if needed.

        Note: JSONL chunks use line-based provenance (jsonl_line_start/end) rather than
        character offsets. The start_char/end_char fields reflect chunk text length only,
        not file offsets - this is intentional for line-oriented formats.
        """
        # Filter base64 content before serializing to prevent useless binary chunks
        filtered_obj = self._filter_base64_fields(obj)
        text = json.dumps(filtered_obj, indent=2, ensure_ascii=False)

        if len(text) <= self._chunk_size:
            # Fits in one chunk
            metadata = ChunkMetadata(
                start_char=0,
                end_char=len(text),
                jsonl_line_start=line_num,
                jsonl_line_end=line_num,
                jsonl_record_count=1,
                jsonl_schema_hint=schema_hint,
            )
            chunks.append(
                Chunk(
                    text=text,
                    source_path=str(path),
                    chunk_index=0,
                    file_type='jsonl',
                    metadata=metadata,
                )
            )
        else:
            # Need to sub-chunk (filtered_obj preserves structure, so isinstance check works)
            if isinstance(filtered_obj, list) and len(filtered_obj) > 1:
                # Array: split by elements (already filtered)
                for i, item in enumerate(filtered_obj):
                    item_text = json.dumps(item, indent=2, ensure_ascii=False)
                    if len(item_text) > self._chunk_size:
                        # Sub-chunk the item
                        sub_texts = self._text_splitter.split_text(item_text)
                        for j, sub_text in enumerate(sub_texts):
                            metadata = ChunkMetadata(
                                start_char=0,
                                end_char=len(sub_text),
                                jsonl_line_start=line_num,
                                jsonl_line_end=line_num,
                                jsonl_record_count=1,
                                jsonl_schema_hint=schema_hint,
                                json_path=f'[{i}]#{j}',
                            )
                            chunks.append(
                                Chunk(
                                    text=sub_text,
                                    source_path=str(path),
                                    chunk_index=0,
                                    file_type='jsonl',
                                    metadata=metadata,
                                )
                            )
                    else:
                        metadata = ChunkMetadata(
                            start_char=0,
                            end_char=len(item_text),
                            jsonl_line_start=line_num,
                            jsonl_line_end=line_num,
                            jsonl_record_count=1,
                            jsonl_schema_hint=schema_hint,
                            json_path=f'[{i}]',
                        )
                        chunks.append(
                            Chunk(
                                text=item_text,
                                source_path=str(path),
                                chunk_index=0,
                                file_type='jsonl',
                                metadata=metadata,
                            )
                        )
            else:
                # Object or single-element array: text split
                sub_texts = self._text_splitter.split_text(text)
                for i, sub_text in enumerate(sub_texts):
                    metadata = ChunkMetadata(
                        start_char=0,
                        end_char=len(sub_text),
                        jsonl_line_start=line_num,
                        jsonl_line_end=line_num,
                        jsonl_record_count=1,
                        jsonl_schema_hint=schema_hint,
                        json_path=f'#{i}',
                    )
                    chunks.append(
                        Chunk(
                            text=sub_text,
                            source_path=str(path),
                            chunk_index=0,
                            file_type='jsonl',
                            metadata=metadata,
                        )
                    )
