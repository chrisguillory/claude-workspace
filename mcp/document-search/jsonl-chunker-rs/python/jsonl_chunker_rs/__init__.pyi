from collections.abc import Sequence

def chunk_jsonl(
    path: str,
    chunk_size: int = ...,
    overlap: int = ...,
    min_chunk_length: int = ...,
) -> tuple[Sequence[tuple[str, int]], int]:
    """Chunk JSONL file using Rust + rayon parallelism.

    Reads a JSONL file, parses each line, filters base64 content,
    pretty-prints oversized records, and splits into chunks on newline
    boundaries. Processing is parallelized across CPU cores with the
    GIL released.

    Args:
        path: Path to JSONL file.
        chunk_size: Target chunk size in characters (default 1500).
        overlap: Overlap between consecutive chunks in characters (default 200).
        min_chunk_length: Minimum chunk length to include (default 50).

    Returns:
        Tuple of (chunks, skipped_lines). Chunks is a list of (chunk_text,
        line_number) tuples with 1-based line numbers. skipped_lines is the
        count of oversized lines that failed JSON parsing and were dropped.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If chunk_size is 0 or overlap >= chunk_size.
    """
    ...
