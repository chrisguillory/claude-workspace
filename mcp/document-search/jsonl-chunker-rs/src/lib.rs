use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Minimum length for a string to be considered potential base64 content.
const BASE64_MIN_LENGTH: usize = 500;

/// Characters valid in base64 encoding (standard alphabet + padding).
const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

/// Check if a string is likely base64-encoded binary content.
fn is_likely_base64(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() >= BASE64_MIN_LENGTH
        && bytes.len() % 4 == 0
        && bytes[..100.min(bytes.len())]
            .iter()
            .all(|b| BASE64_CHARS.contains(b))
}

/// Replace base64 content with placeholders in a JSON value (recursive).
fn filter_base64(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let filtered = map
                .into_iter()
                .map(|(k, v)| {
                    if let Value::String(ref s) = v {
                        if is_likely_base64(s) {
                            return (
                                k,
                                Value::String(format!("[base64 content, {} bytes]", s.len())),
                            );
                        }
                    }
                    (k, filter_base64(v))
                })
                .collect();
            Value::Object(filtered)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(filter_base64).collect()),
        other => other,
    }
}

/// Split a single long string into chunks at character boundaries.
/// Advances by `step` bytes each iteration to provide overlap.
fn split_at_char_boundaries(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < text.len() {
        // Find end boundary (snap backward to valid UTF-8)
        let mut end = (pos + chunk_size).min(text.len());
        while end > pos && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == pos {
            // Advance past the current multi-byte character
            end = pos + 1;
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
        }
        chunks.push(text[pos..end].to_string());

        // Advance by step, snap forward to valid UTF-8
        let mut next = pos + step;
        while next < text.len() && !text.is_char_boundary(next) {
            next += 1;
        }
        pos = next;
    }

    chunks
}

/// Split text on newline boundaries into chunks of approximately `chunk_size`
/// characters with `overlap` characters carried forward between chunks.
/// Falls back to character-boundary splitting for lines exceeding `chunk_size`.
fn split_text_on_newlines(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let lines: Vec<&str> = text.split('\n').collect();
    let mut chunks = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_len: usize = 0;

    for line in &lines {
        let line_len = line.len() + 1; // +1 for newline

        // Single line exceeds chunk_size — split on character boundaries
        if line.len() > chunk_size {
            if !current_lines.is_empty() {
                chunks.push(current_lines.join("\n"));
                current_lines = Vec::new();
                current_len = 0;
            }
            chunks.extend(split_at_char_boundaries(line, chunk_size, overlap));
            continue;
        }

        if current_len + line_len > chunk_size && !current_lines.is_empty() {
            chunks.push(current_lines.join("\n"));

            // Compute overlap: keep trailing lines fitting in overlap budget
            let mut overlap_lines: Vec<&str> = Vec::new();
            let mut overlap_len: usize = 0;
            for prev_line in current_lines.iter().rev() {
                if overlap_len + prev_line.len() + 1 > overlap {
                    break;
                }
                overlap_lines.insert(0, prev_line);
                overlap_len += prev_line.len() + 1;
            }
            current_lines = overlap_lines;
            current_len = overlap_len;

            // If overlap + new line still exceeds budget, flush overlap first
            if current_len + line_len > chunk_size && !current_lines.is_empty() {
                chunks.push(current_lines.join("\n"));
                current_lines = Vec::new();
                current_len = 0;
            }
        }
        current_lines.push(line);
        current_len += line_len;
    }

    if !current_lines.is_empty() {
        chunks.push(current_lines.join("\n"));
    }

    chunks
}

/// Chunk a JSONL file using rayon parallelism.
///
/// Reads the file, parses each JSON line, filters base64 content,
/// pretty-prints oversized records, splits into chunks on newline
/// boundaries, and returns (chunk_text, line_number) tuples.
///
/// Line numbers are 1-based for consistency with Python conventions.
#[pyfunction]
#[pyo3(signature = (path, chunk_size=1500, overlap=200, min_chunk_length=50))]
fn chunk_jsonl(
    py: Python<'_>,
    path: &str,
    chunk_size: usize,
    overlap: usize,
    min_chunk_length: usize,
) -> PyResult<(Vec<(String, usize)>, usize)> {
    if chunk_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_size must be > 0",
        ));
    }
    if overlap >= chunk_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlap must be < chunk_size",
        ));
    }

    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!("{}: {}", path, e))
        } else {
            pyo3::exceptions::PyIOError::new_err(format!("{}: {}", path, e))
        }
    })?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    // Read all lines with 1-based line numbers
    let lines: Vec<(usize, String)> = reader
        .lines()
        .enumerate()
        .filter_map(|(idx, r)| r.ok().map(|l| (idx + 1, l))) // 1-based
        .filter(|(_, l)| !l.is_empty())
        .collect();

    // Process in parallel, releasing the GIL.
    // Each worker returns (chunks, skipped) where skipped is 0 or 1.
    let results: Vec<(Vec<(String, usize)>, usize)> = py.allow_threads(|| {
        lines
            .par_iter()
            .map(|(line_num, line)| {
                let mut local_chunks = Vec::new();

                if line.len() <= chunk_size {
                    if line.len() >= min_chunk_length {
                        local_chunks.push((line.clone(), *line_num));
                    }
                    (local_chunks, 0)
                } else if let Ok(parsed) = serde_json::from_str::<Value>(line) {
                    let filtered = filter_base64(parsed);
                    if let Ok(indented) = serde_json::to_string_pretty(&filtered) {
                        if indented.len() <= chunk_size {
                            if indented.len() >= min_chunk_length {
                                local_chunks.push((indented, *line_num));
                            }
                        } else {
                            for chunk in
                                split_text_on_newlines(&indented, chunk_size, overlap)
                            {
                                if chunk.len() >= min_chunk_length {
                                    local_chunks.push((chunk, *line_num));
                                }
                            }
                        }
                    }
                    (local_chunks, 0)
                } else {
                    // Oversized line that failed JSON parse — can't split without structure
                    (local_chunks, 1)
                }
            })
            .collect()
    });

    let mut all_chunks = Vec::new();
    let mut total_skipped: usize = 0;
    for (chunks, skipped) in results {
        all_chunks.extend(chunks);
        total_skipped += skipped;
    }
    Ok((all_chunks, total_skipped))
}

#[pymodule]
fn jsonl_chunker_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_jsonl, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_text_basic() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let chunks = split_text_on_newlines(text, 12, 0);
        assert!(chunks.len() > 1);
        // All original content should be present across chunks
        let all_lines: Vec<&str> = chunks.iter().flat_map(|c| c.split('\n')).collect();
        assert!(all_lines.contains(&"line1"));
        assert!(all_lines.contains(&"line5"));
    }

    #[test]
    fn test_split_text_single_chunk() {
        let text = "short text";
        let chunks = split_text_on_newlines(text, 1500, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "short text");
    }

    #[test]
    fn test_split_text_overlap() {
        let text = "aaaa\nbbbb\ncccc\ndddd\neeee";
        let chunks = split_text_on_newlines(text, 10, 5);
        assert!(chunks.len() >= 2);
        // With overlap, last line(s) of chunk N should appear at start of chunk N+1
        if chunks.len() >= 2 {
            let first_last_line = chunks[0].split('\n').last().unwrap();
            let second_lines: Vec<&str> = chunks[1].split('\n').collect();
            assert!(
                second_lines.contains(&first_last_line),
                "Overlap not preserved: '{}' not found in chunk 2: {:?}",
                first_last_line,
                second_lines
            );
        }
    }

    #[test]
    fn test_split_long_line_no_newlines() {
        // A single line exceeding chunk_size — must split on char boundaries
        let long_line = "x".repeat(5000);
        let chunks = split_text_on_newlines(&long_line, 1500, 200);
        assert!(chunks.len() >= 3, "Expected 3+ chunks, got {}", chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= 1500,
                "Chunk {} has {} chars, exceeds 1500",
                i,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_split_long_line_multibyte_utf8() {
        // Multi-byte chars (→ = 3 bytes each). 2000 arrows = 6000 bytes
        let arrows = "→".repeat(2000);
        let chunks = split_text_on_newlines(&arrows, 1500, 200);
        assert!(chunks.len() >= 2);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= 1500,
                "Chunk {} has {} bytes, exceeds 1500",
                i,
                chunk.len()
            );
            // Verify valid UTF-8 (would panic on slice if not)
            assert!(chunk.chars().all(|c| c == '→'));
        }
    }

    #[test]
    fn test_split_mixed_long_and_short() {
        let text = format!("short1\nshort2\n{}\nshort3", "y".repeat(3000));
        let chunks = split_text_on_newlines(&text, 1500, 200);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= 1500,
                "Chunk {} has {} chars, exceeds 1500",
                i,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_split_overlap_doesnt_exceed_budget() {
        // Overlap carryover should not produce oversized chunks
        let text = "aaaaaaaaa\nbbbbbbbbb\ncccccccccccc";
        let chunks = split_text_on_newlines(&text, 20, 10);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= 20,
                "Chunk {} has {} chars, exceeds chunk_size 20: {:?}",
                i,
                chunk.len(),
                chunk
            );
        }
    }

    #[test]
    fn test_char_boundary_split_tiny_chunk_size() {
        // chunk_size smaller than a multi-byte char — should not panic
        let text = "→→→"; // 9 bytes, 3 chars
        let chunks = split_at_char_boundaries(&text, 2, 0);
        assert!(!chunks.is_empty());
        // Each chunk should be valid UTF-8
        for chunk in &chunks {
            assert!(chunk.len() > 0);
            let _ = chunk.chars().count(); // validates UTF-8
        }
    }

    #[test]
    fn test_split_text_empty() {
        let chunks = split_text_on_newlines("", 1500, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn test_base64_detection() {
        // Valid base64: >= 500 chars, mod 4, all base64 alphabet
        let valid_b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            .repeat(8); // 512 chars
        assert!(is_likely_base64(&valid_b64));

        // Too short (< 500)
        assert!(!is_likely_base64("QUFB"));

        // Not mod 4
        assert!(!is_likely_base64(&"A".repeat(501)));

        // Has spaces (not base64 alphabet)
        let with_spaces = format!("{}  {}", "A".repeat(250), "B".repeat(250));
        assert!(!is_likely_base64(&with_spaces));
    }

    #[test]
    fn test_base64_filter_recursive() {
        let b64_content = "A".repeat(504); // 504 = divisible by 4, > 500
        let json: Value = serde_json::from_str(&format!(
            r#"{{"outer": {{"inner": "{}"}}, "keep": "normal"}}"#,
            b64_content
        ))
        .unwrap();

        let filtered = filter_base64(json);
        let outer = filtered.get("outer").unwrap().get("inner").unwrap();
        assert!(outer.as_str().unwrap().starts_with("[base64 content,"));
        assert_eq!(filtered.get("keep").unwrap().as_str().unwrap(), "normal");
    }

    #[test]
    fn test_base64_preserves_non_base64() {
        let json: Value =
            serde_json::from_str(r#"{"text": "hello world", "count": 42}"#).unwrap();
        let filtered = filter_base64(json.clone());
        assert_eq!(json, filtered);
    }

    // ── Text splitter: additional edge cases ──────────────────────

    #[test]
    fn test_split_exact_chunk_size() {
        let text = "a".repeat(1500);
        let chunks = split_text_on_newlines(&text, 1500, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1500);
    }

    #[test]
    fn test_split_one_over_chunk_size() {
        let text = "a".repeat(1501);
        let chunks = split_text_on_newlines(&text, 1500, 200);
        assert!(chunks.len() >= 2, "Should split 1501 chars into 2+ chunks");
        for (i, c) in chunks.iter().enumerate() {
            assert!(c.len() <= 1500, "Chunk {} is {} chars", i, c.len());
        }
    }

    #[test]
    fn test_split_under_chunk_size() {
        let text = "hello\nworld\nfoo";
        let chunks = split_text_on_newlines(text, 1500, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello\nworld\nfoo");
    }

    #[test]
    fn test_split_chunk_size_one() {
        let text = "abc";
        let chunks = split_text_on_newlines(text, 1, 0);
        assert!(chunks.len() >= 3, "Each char should be a chunk, got {}", chunks.len());
    }

    #[test]
    fn test_split_no_overlap() {
        let text = "aaaa\nbbbb\ncccc\ndddd";
        let chunks = split_text_on_newlines(text, 10, 0);
        assert!(chunks.len() >= 2);
        // With zero overlap, chunks should be disjoint
        if chunks.len() >= 2 {
            let first_last = chunks[0].split('\n').last().unwrap();
            let second_first = chunks[1].split('\n').next().unwrap();
            assert_ne!(first_last, second_first, "Zero overlap should not repeat lines");
        }
    }

    #[test]
    fn test_split_strict_size_on_realistic_input() {
        // Simulate a realistic pretty-printed JSON with long string values
        let mut lines = Vec::new();
        lines.push(r#"{"#.to_string());
        lines.push(format!(r#"  "short_key": "short_value","#));
        lines.push(format!(r#"  "long_key": "{}","#, "x".repeat(3000)));
        lines.push(format!(r#"  "medium_key": "{}","#, "y".repeat(800)));
        lines.push(r#"  "final": true"#.to_string());
        lines.push(r#"}"#.to_string());
        let text = lines.join("\n");

        let chunks = split_text_on_newlines(&text, 1500, 200);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= 1500,
                "Chunk {} has {} chars (max 1500): {:?}",
                i,
                chunk.len(),
                &chunk[..80.min(chunk.len())]
            );
        }
    }

    // ── Character boundary splitter ──────────────────────────────

    #[test]
    fn test_char_boundary_basic() {
        let text = "a".repeat(3000);
        let chunks = split_at_char_boundaries(&text, 1000, 100);
        assert!(chunks.len() >= 3);
        for (i, c) in chunks.iter().enumerate() {
            assert!(c.len() <= 1000, "Chunk {} is {} chars", i, c.len());
        }
    }

    #[test]
    fn test_char_boundary_overlap_content() {
        let text = "abcdefghij"; // 10 chars
        let chunks = split_at_char_boundaries(&text, 5, 2);
        // step = 5 - 2 = 3. Positions: 0..5, 3..8, 6..10
        assert!(chunks.len() >= 2);
        // Overlap: end of chunk 0 should appear at start of chunk 1
        if chunks.len() >= 2 {
            let overlap_from_first = &chunks[0][chunks[0].len() - 2..];
            let start_of_second = &chunks[1][..2.min(chunks[1].len())];
            assert_eq!(
                overlap_from_first, start_of_second,
                "Overlap content should match"
            );
        }
    }

    #[test]
    fn test_char_boundary_emoji() {
        // Emoji: 🎉 = 4 bytes (U+1F389)
        let text = "🎉".repeat(500); // 2000 bytes
        let chunks = split_at_char_boundaries(&text, 100, 20);
        for (i, c) in chunks.iter().enumerate() {
            assert!(c.len() <= 100, "Chunk {} is {} bytes", i, c.len());
            // Verify all chars are valid emoji
            assert!(c.chars().all(|ch| ch == '🎉'), "Chunk {} has invalid chars", i);
        }
    }

    #[test]
    fn test_char_boundary_mixed_width() {
        // Mix of 1-byte (ASCII), 2-byte (é), 3-byte (→), 4-byte (🎉)
        let text = "hello→world🎉café→→🎉🎉end";
        let chunks = split_at_char_boundaries(&text, 10, 3);
        for (i, c) in chunks.iter().enumerate() {
            assert!(c.len() <= 10, "Chunk {} is {} bytes", i, c.len());
            let _ = c.chars().count(); // validates UTF-8
        }
    }

    #[test]
    fn test_char_boundary_empty() {
        let chunks = split_at_char_boundaries("", 100, 10);
        assert!(chunks.is_empty() || chunks == vec![""]);
    }

    #[test]
    fn test_char_boundary_under_chunk_size() {
        let chunks = split_at_char_boundaries("hello", 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello");
    }

    #[test]
    fn test_char_boundary_forward_progress() {
        // Ensure no infinite loop with large overlap
        let text = "x".repeat(100);
        let chunks = split_at_char_boundaries(&text, 10, 9);
        // step = 10 - 9 = 1, so ~100 chunks
        assert!(chunks.len() >= 90);
        assert!(chunks.len() <= 110);
    }

    // ── Base64: additional edge cases ────────────────────────────

    #[test]
    fn test_base64_at_threshold() {
        // Exactly 500 chars, mod 4, all valid base64
        let b64 = "ABCD".repeat(125); // 500 chars
        assert!(is_likely_base64(&b64));
    }

    #[test]
    fn test_base64_below_threshold() {
        let b64 = "ABCD".repeat(124); // 496 chars < 500
        assert!(!is_likely_base64(&b64));
    }

    #[test]
    fn test_base64_in_array() {
        let b64 = "A".repeat(504);
        let json: Value = serde_json::from_str(&format!(
            r#"[{{"key": "{}"}}, {{"key": "normal"}}]"#,
            b64
        ))
        .unwrap();
        let filtered = filter_base64(json);
        let arr = filtered.as_array().unwrap();
        assert!(arr[0]["key"].as_str().unwrap().starts_with("[base64 content,"));
        assert_eq!(arr[1]["key"].as_str().unwrap(), "normal");
    }

    #[test]
    fn test_base64_placeholder_format() {
        let b64 = "A".repeat(504);
        let json: Value =
            serde_json::from_str(&format!(r#"{{"data": "{}"}}"#, b64)).unwrap();
        let filtered = filter_base64(json);
        let placeholder = filtered["data"].as_str().unwrap();
        assert_eq!(placeholder, "[base64 content, 504 bytes]");
    }

    #[test]
    fn test_base64_non_base64_long_string() {
        // 600 chars with spaces — not base64
        let text = format!("This is a long text. {}", "word ".repeat(116));
        assert!(!is_likely_base64(&text));
    }

    #[test]
    fn test_base64_deeply_nested() {
        let b64 = "A".repeat(504);
        let json: Value = serde_json::from_str(&format!(
            r#"{{"a": {{"b": {{"c": {{"d": "{}"}}}}}}}}"#,
            b64
        ))
        .unwrap();
        let filtered = filter_base64(json);
        let inner = &filtered["a"]["b"]["c"]["d"];
        assert!(inner.as_str().unwrap().starts_with("[base64 content,"));
    }
}
