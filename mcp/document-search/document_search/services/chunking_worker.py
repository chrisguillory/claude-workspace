"""JSONL chunking worker for ProcessPoolExecutor.

Runs in subprocesses to bypass the GIL for CPU-bound JSON parsing and
text processing. This is the main optimization target - benchmarks show
4x speedup for JSONL files.

Other file types (text, markdown, pdf, csv) remain in the main process
for now.

IMPORTANT: Exceptions propagate to caller - fail fast principle.
Only JSONDecodeError for individual malformed lines is handled.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_search.schemas.chunking import Chunk, ChunkMetadata, FileType

__all__ = [
    'chunk_jsonl',
    'ChunkData',
    'ChunkMetadataDict',
]

# Default chunking parameters (used if caller doesn't specify)
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
MIN_CHUNK_LENGTH = 50

# JSONL constants
JSONL_ENCODINGS = ('utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252')
JSONL_SAMPLE_SIZE = 200
JSONL_GROUPING_THRESHOLD = 800
JSONL_MAX_GROUP_SIZE = 15
JSONL_BASE64_MIN_LENGTH = 500


class ChunkMetadataDict(TypedDict, total=False):
    """Serializable metadata for IPC - matches ChunkMetadata fields."""

    start_char: int
    end_char: int
    heading_context: str | None
    page_number: int | None
    json_path: str | None
    jsonl_line_start: int | None
    jsonl_line_end: int | None
    jsonl_record_count: int | None
    jsonl_schema_hint: str | None


class ChunkData(TypedDict):
    """Serializable chunk data for IPC."""

    text: str
    source_path: str
    chunk_index: int
    file_type: FileType
    metadata: ChunkMetadataDict


def chunk_jsonl(
    path_str: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Sequence[ChunkData]:
    """Chunk JSONL file in subprocess.

    This is the main ProcessPool optimization target. JSON parsing and
    text processing are CPU-bound and benefit from GIL bypass.

    Args:
        path_str: Path to JSONL file as string.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks for context continuity.

    Returns:
        Sequence of ChunkData for IPC back to main process.

    Raises:
        FileNotFoundError: If file doesn't exist.
        OSError: If file cannot be read.
    """
    path = Path(path_str)
    chunks = _chunk_jsonl_impl(path, chunk_size, chunk_overlap)

    # Filter short chunks
    chunks = [c for c in chunks if len(c.text.strip()) >= MIN_CHUNK_LENGTH]

    # Convert to serializable TypedDict
    return tuple(
        ChunkData(
            text=c.text,
            source_path=c.source_path,
            chunk_index=c.chunk_index,
            file_type=c.file_type,
            metadata=cast(ChunkMetadataDict, c.metadata.model_dump()),
        )
        for c in chunks
    )


def _chunk_jsonl_impl(path: Path, chunk_size: int, chunk_overlap: int) -> Sequence[Chunk]:
    """JSONL chunking implementation with adaptive grouping."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    chunks: list[Chunk] = []

    encoding = _detect_encoding(path)

    sample_buffer: list[tuple[int, object, int]] = []
    sample_objects: list[dict[str, object]] = []
    strategy_decided = False
    should_group = False
    schema_hint: str | None = None

    group_buffer: list[tuple[int, object]] = []
    group_size_chars = 0

    with open(path, encoding=encoding, errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines (matches original - JSONL may have occasional bad lines)
                continue

            size = len(line)

            if not strategy_decided:
                sample_buffer.append((line_num, obj, size))
                if isinstance(obj, dict):
                    sample_objects.append(obj)

                if len(sample_buffer) >= JSONL_SAMPLE_SIZE:
                    sizes = [s for _, _, s in sample_buffer]
                    median_size = sorted(sizes)[len(sizes) // 2]
                    should_group = median_size < JSONL_GROUPING_THRESHOLD
                    schema_hint = _detect_schema_hint(sample_objects[:50])
                    strategy_decided = True

                    for buf_line, buf_obj, buf_size in sample_buffer:
                        if buf_size > chunk_size:
                            if group_buffer:
                                _flush_group(group_buffer, path, schema_hint, chunks)
                                group_buffer = []
                                group_size_chars = 0
                            _add_single_record(buf_line, buf_obj, path, schema_hint, chunks, chunk_size, text_splitter)
                        elif should_group:
                            group_buffer, group_size_chars = _update_group(
                                buf_line,
                                buf_obj,
                                buf_size,
                                path,
                                schema_hint,
                                group_buffer,
                                group_size_chars,
                                chunk_size,
                                chunks,
                            )
                        else:
                            _add_single_record(buf_line, buf_obj, path, schema_hint, chunks, chunk_size, text_splitter)
                    sample_buffer = []
                continue

            if size > chunk_size:
                if group_buffer:
                    _flush_group(group_buffer, path, schema_hint, chunks)
                    group_buffer = []
                    group_size_chars = 0
                _add_single_record(line_num, obj, path, schema_hint, chunks, chunk_size, text_splitter)
            elif should_group:
                group_buffer, group_size_chars = _update_group(
                    line_num,
                    obj,
                    size,
                    path,
                    schema_hint,
                    group_buffer,
                    group_size_chars,
                    chunk_size,
                    chunks,
                )
            else:
                _add_single_record(line_num, obj, path, schema_hint, chunks, chunk_size, text_splitter)

    # Handle small files (< sample size)
    if not strategy_decided and sample_buffer:
        sizes = [s for _, _, s in sample_buffer]
        median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
        should_group = median_size < JSONL_GROUPING_THRESHOLD
        schema_hint = _detect_schema_hint(sample_objects[:50])

        for buf_line, buf_obj, buf_size in sample_buffer:
            if buf_size > chunk_size:
                if group_buffer:
                    _flush_group(group_buffer, path, schema_hint, chunks)
                    group_buffer = []
                    group_size_chars = 0
                _add_single_record(buf_line, buf_obj, path, schema_hint, chunks, chunk_size, text_splitter)
            elif should_group:
                group_buffer, group_size_chars = _update_group(
                    buf_line,
                    buf_obj,
                    buf_size,
                    path,
                    schema_hint,
                    group_buffer,
                    group_size_chars,
                    chunk_size,
                    chunks,
                )
            else:
                _add_single_record(buf_line, buf_obj, path, schema_hint, chunks, chunk_size, text_splitter)

    if group_buffer:
        _flush_group(group_buffer, path, schema_hint, chunks)

    # Assign sequential indices
    for i, chunk in enumerate(chunks):
        chunks[i] = Chunk(
            text=chunk.text,
            source_path=chunk.source_path,
            chunk_index=i,
            file_type=chunk.file_type,
            metadata=chunk.metadata,
        )

    return chunks


def _detect_encoding(path: Path) -> str:
    """Detect file encoding from BOM or content."""
    with open(path, 'rb') as f:
        sample = f.read(8192)

    if not sample:
        return 'utf-8'

    if sample.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    if sample.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    if sample.startswith(b'\xfe\xff'):
        return 'utf-16-be'

    for encoding in JSONL_ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    return 'utf-8'


def _detect_schema_hint(objects: Sequence[Mapping[str, object]]) -> str | None:
    """Detect common fields across records for context."""
    if not objects:
        return None

    common_keys = set(objects[0].keys())
    for obj in objects[1:]:
        common_keys &= set(obj.keys())

    if not common_keys:
        key_counts: dict[str, int] = {}
        for obj in objects:
            for key in obj:
                key_counts[key] = key_counts.get(key, 0) + 1
        threshold = len(objects) * 0.7
        common_keys = {k for k, v in key_counts.items() if v >= threshold}

    if common_keys:
        return ', '.join(sorted(common_keys)[:5])
    return None


def _is_likely_base64(value: str) -> bool:
    """Check if string is likely base64-encoded binary."""
    if len(value) < JSONL_BASE64_MIN_LENGTH or len(value) % 4 != 0:
        return False
    sample = value[:100]
    base64_chars = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
    return all(c in base64_chars for c in sample)


def _filter_base64(obj: object) -> object:
    """Replace base64 content with placeholders."""
    if isinstance(obj, dict):
        result: dict[str, object] = {}
        for k, v in obj.items():
            if isinstance(v, str) and _is_likely_base64(v):
                result[k] = f'[base64 content, {len(v)} bytes]'
            else:
                result[k] = _filter_base64(v)
        return result
    elif isinstance(obj, list):
        return [_filter_base64(item) for item in obj]
    return obj


def _update_group(  # strict_typing_linter.py: mutable-type (returns buffer for reuse)
    line_num: int,
    obj: object,
    size: int,
    path: Path,
    schema_hint: str | None,
    group_buffer: list[tuple[int, object]],  # strict_typing_linter.py: mutable-type (mutates)
    group_size_chars: int,
    chunk_size: int,
    chunks: list[Chunk],  # strict_typing_linter.py: mutable-type (appends)
) -> tuple[list[tuple[int, object]], int]:
    """Update group buffer, flushing when full."""
    if group_buffer and (group_size_chars + size > chunk_size or len(group_buffer) >= JSONL_MAX_GROUP_SIZE):
        _flush_group(group_buffer, path, schema_hint, chunks)
        group_buffer = []
        group_size_chars = 0

    group_buffer.append((line_num, obj))
    return group_buffer, group_size_chars + size


def _flush_group(
    group_buffer: Sequence[tuple[int, object]],
    path: Path,
    schema_hint: str | None,
    chunks: list[Chunk],  # strict_typing_linter.py: mutable-type (appends)
) -> None:
    """Create chunk from grouped records."""
    if not group_buffer:
        return

    line_start = group_buffer[0][0]
    line_end = group_buffer[-1][0]

    lines = [json.dumps(_filter_base64(obj), ensure_ascii=False) for _, obj in group_buffer]
    text = '\n'.join(lines)

    chunks.append(
        Chunk(
            text=text,
            source_path=str(path),
            chunk_index=0,
            file_type='jsonl',
            metadata=ChunkMetadata(
                start_char=0,
                end_char=len(text),
                jsonl_line_start=line_start,
                jsonl_line_end=line_end,
                jsonl_record_count=len(group_buffer),
                jsonl_schema_hint=schema_hint,
            ),
        )
    )


def _add_single_record(
    line_num: int,
    obj: object,
    path: Path,
    schema_hint: str | None,
    chunks: list[Chunk],  # strict_typing_linter.py: mutable-type (appends)
    chunk_size: int,
    text_splitter: RecursiveCharacterTextSplitter,
) -> None:
    """Add single record as chunk(s), sub-chunking if needed."""
    filtered = _filter_base64(obj)
    text = json.dumps(filtered, indent=2, ensure_ascii=False)

    if len(text) <= chunk_size:
        chunks.append(
            Chunk(
                text=text,
                source_path=str(path),
                chunk_index=0,
                file_type='jsonl',
                metadata=ChunkMetadata(
                    start_char=0,
                    end_char=len(text),
                    jsonl_line_start=line_num,
                    jsonl_line_end=line_num,
                    jsonl_record_count=1,
                    jsonl_schema_hint=schema_hint,
                ),
            )
        )
    else:
        sub_texts = text_splitter.split_text(text)
        for i, sub_text in enumerate(sub_texts):
            chunks.append(
                Chunk(
                    text=sub_text,
                    source_path=str(path),
                    chunk_index=0,
                    file_type='jsonl',
                    metadata=ChunkMetadata(
                        start_char=0,
                        end_char=len(sub_text),
                        jsonl_line_start=line_num,
                        jsonl_line_end=line_num,
                        jsonl_record_count=1,
                        jsonl_schema_hint=schema_hint,
                        json_path=f'#{i}',
                    ),
                )
            )
