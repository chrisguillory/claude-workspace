"""Chunking service - splits documents into embeddable chunks.

Handles different file types with appropriate chunking strategies:
- Markdown: Structure-aware splitting by headers
- Text: Recursive character splitting
- JSON: Logical structure-based splitting
- PDF: Page-aware semantic chunking with parallel processing
- CSV: Context-aware row grouping with header context

PDF processing uses PyMuPDF for extraction with pdfplumber for tables.
Parallel page processing controlled via semaphore for memory efficiency.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fitz  # PyMuPDF

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

logger = logging.getLogger(__name__)

# Default chunking parameters (research-backed: 400-512 tokens optimal)
DEFAULT_CHUNK_SIZE = 1500  # ~375-500 tokens at 3-4 chars/token
DEFAULT_CHUNK_OVERLAP = 300  # 20% overlap

# PDF processing constants
PDF_MAX_CONCURRENT_PAGES = 8  # Parallel page processing
PDF_OCR_CHAR_THRESHOLD = 100  # If extracted text < this, try OCR
PDF_HEADER_FOOTER_SAMPLE_PAGES = 5  # Pages to analyze for header/footer
PDF_HEADER_REGION_RATIO = 0.12  # Top 12% of page is header region
PDF_FOOTER_REGION_RATIO = 0.12  # Bottom 12% of page is footer region
PDF_MIN_REPEAT_COUNT = 2  # Minimum repeats to consider as header/footer

# CSV processing constants
CSV_MIN_GROUP_SIZE = 5  # Minimum rows per chunk
CSV_MAX_GROUP_SIZE = 15  # Maximum rows per chunk
CSV_CHUNK_SIZE_TARGET = 1000  # Target chunk size for adaptive grouping
CSV_ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']

# PDF type detection
type PDFType = Literal['text', 'scanned', 'mixed', 'image_heavy']


@dataclasses.dataclass
class _PDFAnalysis:
    """Analysis results for a PDF document."""

    pdf_type: PDFType
    header_pattern: str
    footer_pattern: str
    bookmarks: dict[int, str]  # page_num -> bookmark title
    page_count: int


@dataclasses.dataclass
class _PDFPageResult:
    """Result of processing a single PDF page."""

    page_num: int
    text: str
    table_markdown: str | None = None
    required_ocr: bool = False
    error: str | None = None


class ChunkingService:
    """Chunks documents by file type with async processing for I/O-bound operations.

    Uses appropriate strategy per file type:
    - Markdown: Header-aware splitting preserves section context
    - Text: Recursive splitting at natural boundaries
    - JSON: Structure-aware splitting for nested data
    - PDF: Page-aware semantic chunking with parallel processing
    - CSV: Context-aware row grouping with header context
    """

    @classmethod
    async def create(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        pdf_max_concurrent: int = PDF_MAX_CONCURRENT_PAGES,
    ) -> ChunkingService:
        """Create chunking service in async context.

        Preferred factory method - ensures semaphore is bound to correct event loop.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            pdf_max_concurrent: Max concurrent PDF page processing.
        """
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            _pdf_semaphore=asyncio.Semaphore(pdf_max_concurrent),
        )

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        *,
        _pdf_semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize chunking service. Use create() for async context safety.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks for context continuity.
            _pdf_semaphore: Internal - use create() instead.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._pdf_semaphore = _pdf_semaphore

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

        return await self._chunk_by_type(path, file_type)

    async def chunk_directory(
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

            chunks = await self.chunk_file(path)
            all_chunks.extend(chunks)

        return all_chunks

    async def _chunk_by_type(self, path: Path, file_type: FileType) -> Sequence[Chunk]:
        """Route to appropriate chunker based on file type."""
        match file_type:
            case 'pdf':
                return await self._chunk_pdf(path)
            case 'csv':
                return await self._chunk_csv(path)
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
        """Chunk PDF using page-aware semantic approach with parallel processing.

        Phases:
        1. Analyze PDF (type detection, header/footer, bookmarks)
        2. Process pages in parallel (extract text, OCR if needed, tables)
        3. Remove headers/footers, apply semantic chunking
        4. Build chunks with page number metadata

        Thread safety: PyMuPDF Document is not thread-safe. We use a threading.Lock
        to serialize access when processing pages in parallel via asyncio.to_thread.
        """
        chunks: list[Chunk] = []

        # Lock for thread-safe access to PyMuPDF Document (not thread-safe)
        doc_lock = threading.Lock()

        # Open PDF in main thread for analysis
        doc = await asyncio.to_thread(fitz.open, str(path))
        try:
            if doc.page_count == 0:
                logger.warning(f'PDF has no pages: {path}')
                return chunks

            # Phase 1: Analyze PDF (single-threaded, no lock needed)
            analysis = await asyncio.to_thread(self._analyze_pdf, doc)
            logger.debug(
                f'PDF analysis: type={analysis.pdf_type}, pages={analysis.page_count}, '
                f'header="{analysis.header_pattern[:30]}..." footer="{analysis.footer_pattern[:30]}..."'
            )

            # Phase 2: Process pages in parallel (lock serializes doc access)
            page_results = await self._process_pdf_pages_parallel(doc, analysis, doc_lock)

            # Phase 3 & 4: Build chunks from page results
            for result in page_results:
                if result.error:
                    logger.warning(f'Page {result.page_num + 1} error: {result.error}')
                    continue

                if not result.text.strip():
                    continue

                # Get heading context from bookmarks
                heading_context = analysis.bookmarks.get(result.page_num)

                # Combine text with table markdown if present
                text = result.text
                if result.table_markdown:
                    text = f'{text}\n\n{result.table_markdown}'

                # Sub-chunk if needed
                page_chunks = self._text_splitter.split_text(text)

                for chunk_text in page_chunks:
                    metadata = ChunkMetadata(
                        start_char=0,
                        end_char=len(chunk_text),
                        page_number=result.page_num + 1,  # 1-indexed
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

        finally:
            await asyncio.to_thread(doc.close)

    def _analyze_pdf(self, doc: fitz.Document) -> _PDFAnalysis:
        """Analyze PDF for type, headers/footers, and bookmarks.

        Runs synchronously - called from thread pool.
        """
        # Detect PDF type by sampling pages
        pdf_type = self._detect_pdf_type(doc)

        # Detect repeating headers and footers
        header_pattern, footer_pattern = self._detect_header_footer(doc)

        # Extract bookmarks (table of contents)
        bookmarks = self._extract_bookmarks(doc)

        return _PDFAnalysis(
            pdf_type=pdf_type,
            header_pattern=header_pattern,
            footer_pattern=footer_pattern,
            bookmarks=bookmarks,
            page_count=doc.page_count,
        )

    def _detect_pdf_type(self, doc: fitz.Document) -> PDFType:
        """Detect PDF type by sampling pages."""
        sample_size = min(PDF_HEADER_FOOTER_SAMPLE_PAGES, doc.page_count)
        text_pages = 0
        image_pages = 0

        for page_num in range(sample_size):
            page = doc[page_num]
            text = page.get_text().strip()
            images = page.get_images()

            text_len = len(text)
            image_count = len(images)

            if text_len > 500:
                text_pages += 1
            elif text_len < PDF_OCR_CHAR_THRESHOLD and image_count > 0:
                image_pages += 1

        if text_pages == sample_size:
            return 'text'
        if image_pages == sample_size:
            return 'scanned'
        if image_pages > 0 and text_pages > 0:
            return 'mixed'
        return 'image_heavy'

    def _detect_header_footer(self, doc: fitz.Document) -> tuple[str, str]:
        """Detect repeating header and footer text patterns."""
        sample_size = min(PDF_HEADER_FOOTER_SAMPLE_PAGES, doc.page_count)
        header_candidates: dict[str, int] = {}
        footer_candidates: dict[str, int] = {}

        for page_num in range(sample_size):
            page = doc[page_num]
            page_height = page.rect.height
            header_region = page_height * PDF_HEADER_REGION_RATIO
            footer_region = page_height * (1 - PDF_FOOTER_REGION_RATIO)

            blocks = page.get_text('dict')['blocks']

            for block in blocks:
                if 'lines' not in block:
                    continue

                bbox = block['bbox']
                text = ''.join(span['text'] for line in block['lines'] for span in line['spans']).strip()

                if not text or len(text) < 3:
                    continue

                # Normalize for comparison (remove page numbers)
                normalized = re.sub(r'\b\d+\b', '#', text)

                if bbox[1] < header_region:
                    header_candidates[normalized] = header_candidates.get(normalized, 0) + 1
                elif bbox[3] > footer_region:
                    footer_candidates[normalized] = footer_candidates.get(normalized, 0) + 1

        # Find patterns that repeat enough times
        header_pattern = ''
        footer_pattern = ''

        for text, count in header_candidates.items():
            if count >= PDF_MIN_REPEAT_COUNT and len(text) > len(header_pattern):
                header_pattern = text

        for text, count in footer_candidates.items():
            if count >= PDF_MIN_REPEAT_COUNT and len(text) > len(footer_pattern):
                footer_pattern = text

        return header_pattern, footer_pattern

    def _extract_bookmarks(self, doc: fitz.Document) -> dict[int, str]:
        """Extract PDF bookmarks/outline mapped to page numbers."""
        bookmarks: dict[int, str] = {}

        try:
            toc = doc.get_toc()  # [level, title, page_number, ...]
            for entry in toc:
                if len(entry) >= 3:
                    _, title, page_num = entry[0], entry[1], entry[2]
                    # Only keep first bookmark per page (most specific heading)
                    if page_num - 1 not in bookmarks:
                        bookmarks[page_num - 1] = title
        except Exception:
            pass  # Some PDFs have malformed TOC

        return bookmarks

    async def _process_pdf_pages_parallel(
        self,
        doc: fitz.Document,
        analysis: _PDFAnalysis,
        doc_lock: threading.Lock,
    ) -> list[_PDFPageResult]:
        """Process all PDF pages in parallel with semaphore for memory control.

        Args:
            doc: PyMuPDF Document (not thread-safe).
            analysis: PDF analysis results.
            doc_lock: Lock to serialize access to doc object.
        """

        async def process_page(page_num: int) -> _PDFPageResult:
            async with self._pdf_semaphore:
                return await asyncio.to_thread(
                    self._process_single_page,
                    doc,
                    page_num,
                    analysis,
                    doc_lock,
                )

        # Fire all pages concurrently
        results = await asyncio.gather(
            *[process_page(i) for i in range(doc.page_count)],
            return_exceptions=True,
        )

        # Handle exceptions
        processed: list[_PDFPageResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                processed.append(
                    _PDFPageResult(
                        page_num=i,
                        text='',
                        error=str(result),
                    )
                )
            else:
                processed.append(result)

        return processed

    def _process_single_page(
        self,
        doc: fitz.Document,
        page_num: int,
        analysis: _PDFAnalysis,
        doc_lock: threading.Lock,
    ) -> _PDFPageResult:
        """Process a single PDF page (synchronous, runs in thread pool).

        Uses doc_lock to serialize access to PyMuPDF Document which is not thread-safe.
        """
        required_ocr = False

        # Lock required for all PyMuPDF doc/page operations
        with doc_lock:
            page = doc[page_num]
            text = page.get_text()

            # Try OCR for scanned pages
            if len(text.strip()) < PDF_OCR_CHAR_THRESHOLD and analysis.pdf_type in ('scanned', 'mixed'):
                try:
                    # PyMuPDF OCR (requires Tesseract)
                    text = page.get_textpage_ocr(full=True).get_text()
                    required_ocr = True
                except Exception as e:
                    logger.debug(f'OCR failed for page {page_num}: {e}')
                    # Continue with whatever text we have

            # Check for tables while holding lock (needs page object)
            might_have_table = self._page_might_have_table(page, text)

        # These operations don't need the lock (no doc access)
        text = self._remove_header_footer(
            text,
            analysis.header_pattern,
            analysis.footer_pattern,
        )

        # Table extraction uses pdfplumber (separate file handle, thread-safe)
        table_markdown = None
        if might_have_table:
            table_markdown = self._extract_tables_from_page(doc, page_num, doc_lock)

        return _PDFPageResult(
            page_num=page_num,
            text=text,
            table_markdown=table_markdown,
            required_ocr=required_ocr,
        )

    def _remove_header_footer(self, text: str, header: str, footer: str) -> str:
        """Remove detected header/footer patterns from text."""
        if not header and not footer:
            return text

        lines = text.split('\n')
        result_lines: list[str] = []

        for line in lines:
            line_stripped = line.strip()
            normalized = re.sub(r'\b\d+\b', '#', line_stripped)

            # Skip if matches header or footer pattern
            if header and normalized == header:
                continue
            if footer and normalized == footer:
                continue

            # Also skip common page number patterns
            if re.match(r'^(Page\s+)?\d+(\s+of\s+\d+)?$', line_stripped, re.IGNORECASE):
                continue

            result_lines.append(line)

        return '\n'.join(result_lines)

    def _page_might_have_table(self, page: fitz.Page, text: str) -> bool:
        """Heuristic to detect if page might contain a table."""
        # Check for table-like patterns in text
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False

        # Count lines with consistent delimiter patterns
        tab_lines = sum(1 for line in lines if '\t' in line or '  ' in line)
        pipe_lines = sum(1 for line in lines if '|' in line)

        # If many lines have consistent patterns, might be table
        return tab_lines > len(lines) * 0.3 or pipe_lines > len(lines) * 0.3

    def _extract_tables_from_page(
        self,
        doc: fitz.Document,
        page_num: int,
        doc_lock: threading.Lock,
    ) -> str | None:
        """Extract tables from PDF page using pdfplumber.

        Note: pdfplumber opens its own file handle, so it's thread-safe.
        We only need doc_lock to access doc.name.

        TODO: Performance - this reopens the PDF for each page with tables.
        Could batch table extraction by opening pdfplumber once per PDF.
        """
        import pdfplumber

        try:
            # Get the PDF path from doc (needs lock for thread safety)
            with doc_lock:
                pdf_path = doc.name
            if not pdf_path:
                return None

            # pdfplumber uses its own file handle - no lock needed
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return None

                page = pdf.pages[page_num]
                tables = page.extract_tables()

                if not tables:
                    return None

                markdown_parts: list[str] = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # First row as headers
                    headers = [str(cell or '') for cell in table[0]]
                    rows = table[1:]

                    # Build markdown table
                    md_lines = ['| ' + ' | '.join(headers) + ' |']
                    md_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

                    for row in rows:
                        cells = [str(cell or '').replace('\n', ' ') for cell in row]
                        # Pad or truncate to match header count
                        while len(cells) < len(headers):
                            cells.append('')
                        cells = cells[: len(headers)]
                        md_lines.append('| ' + ' | '.join(cells) + ' |')

                    markdown_parts.append('\n'.join(md_lines))

                return '\n\n'.join(markdown_parts) if markdown_parts else None

        except Exception as e:
            logger.debug(f'Table extraction failed for page {page_num}: {e}')
            return None

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

    def _detect_csv_types(self, df: pd.DataFrame) -> dict[str, str]:
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
        """Chunk .eml email files.

        Basic implementation: treat as text for now.
        TODO: Parse headers, body, and attachments separately.
        """
        # For now, treat email as plain text
        # Future: use email.parser to extract structured data
        return self._chunk_text(content, source_path)
