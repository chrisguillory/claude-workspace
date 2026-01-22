"""Standalone PDF extraction for ProcessPoolExecutor.

This module contains functions that can be pickled and run in subprocesses.
All functions must be module-level (not methods) and use only primitives.

The heavy CPU work (PyMuPDF operations) happens here in subprocess.
Chunking logic stays in the main process for simplicity.

Dependencies:
    - PyMuPDF (fitz): Required for text extraction
    - pdfplumber: Required for table extraction
    - Tesseract: Optional, required for OCR on scanned/mixed PDFs
      Install: brew install tesseract (macOS) or apt-get install tesseract-ocr
      Set TESSDATA_PREFIX if needed for language data location
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import fitz  # PyMuPDF
import pdfplumber

# Constants (duplicated from chunking.py to avoid import issues in subprocess)
PDF_OCR_CHAR_THRESHOLD = 100
PDF_HEADER_FOOTER_SAMPLE_PAGES = 5
PDF_HEADER_REGION_RATIO = 0.12
PDF_FOOTER_REGION_RATIO = 0.12
PDF_MIN_REPEAT_COUNT = 2

type PDFType = Literal['text', 'scanned', 'mixed', 'image_heavy']


@dataclass
class PDFPageData:
    """Data extracted from a single PDF page."""

    page_num: int
    text: str
    might_have_table: bool = False
    table_markdown: str | None = None


@dataclass
class PDFExtractionResult:
    """Complete extraction result for a PDF file."""

    path: str
    page_count: int
    pdf_type: PDFType
    pages: list[PDFPageData]
    bookmarks: dict[int, str]  # page_num -> bookmark title
    header_pattern: str
    footer_pattern: str


def extract_pdf(path: str) -> PDFExtractionResult:
    """Extract all content from a PDF file.

    This function runs in a subprocess via ProcessPoolExecutor.
    It does all the CPU-heavy PyMuPDF work and returns structured data.

    Args:
        path: Absolute path to PDF file.

    Returns:
        PDFExtractionResult with all extracted content.

    Raises:
        Any exception from PyMuPDF or pdfplumber - not swallowed.
    """
    doc = fitz.open(path)
    try:
        if doc.page_count == 0:
            return PDFExtractionResult(
                path=path,
                page_count=0,
                pdf_type='text',
                pages=[],
                bookmarks={},
                header_pattern='',
                footer_pattern='',
            )

        # Phase 1: Analyze PDF type
        pdf_type = _detect_pdf_type(doc)

        # Phase 2: Detect header/footer patterns
        header_pattern, footer_pattern = _detect_header_footer(doc)

        # Phase 3: Extract bookmarks
        bookmarks = _extract_bookmarks(doc)

        # Phase 4: Extract tables (open pdfplumber once for all pages)
        table_data = _extract_all_tables(path)

        # Phase 5: Extract all pages
        pages: list[PDFPageData] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()

            # Try OCR for scanned pages
            if len(text.strip()) < PDF_OCR_CHAR_THRESHOLD and pdf_type in ('scanned', 'mixed'):
                text = page.get_textpage_ocr(full=True).get_text()

            # Remove header/footer
            text = _remove_header_footer(text, header_pattern, footer_pattern)

            # Check for tables
            might_have_table = _page_might_have_table(text)

            pages.append(
                PDFPageData(
                    page_num=page_num,
                    text=text,
                    might_have_table=might_have_table,
                    table_markdown=table_data.get(page_num),
                )
            )

        return PDFExtractionResult(
            path=path,
            page_count=doc.page_count,
            pdf_type=pdf_type,
            pages=pages,
            bookmarks=bookmarks,
            header_pattern=header_pattern,
            footer_pattern=footer_pattern,
        )

    finally:
        doc.close()


def _detect_pdf_type(doc: fitz.Document) -> PDFType:
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


def _detect_header_footer(doc: fitz.Document) -> tuple[str, str]:
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


def _extract_bookmarks(doc: fitz.Document) -> dict[int, str]:
    """Extract PDF bookmarks/outline mapped to page numbers."""
    bookmarks: dict[int, str] = {}

    toc = doc.get_toc()  # [level, title, page_number, ...]
    for entry in toc:
        if len(entry) >= 3:
            _, title, page_num = entry[0], entry[1], entry[2]
            # Only keep first bookmark per page (most specific heading)
            if page_num - 1 not in bookmarks:
                bookmarks[page_num - 1] = title

    return bookmarks


def _remove_header_footer(text: str, header: str, footer: str) -> str:
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


def _page_might_have_table(text: str) -> bool:
    """Heuristic to detect if page might contain a table."""
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return False

    # Count lines with consistent delimiter patterns
    tab_lines = sum(1 for line in lines if '\t' in line or '  ' in line)
    pipe_lines = sum(1 for line in lines if '|' in line)

    # If many lines have consistent patterns, might be table
    return tab_lines > len(lines) * 0.3 or pipe_lines > len(lines) * 0.3


def _extract_all_tables(pdf_path: str) -> dict[int, str]:
    """Extract tables from all pages of a PDF using pdfplumber.

    Opens the PDF once and extracts tables from all pages.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Dict mapping page_num to markdown table string.
    """
    result: dict[int, str] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            if not tables:
                continue

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

            if markdown_parts:
                result[page_num] = '\n\n'.join(markdown_parts)

    return result
