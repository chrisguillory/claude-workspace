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

/// Split text on newline boundaries into chunks of approximately `chunk_size`
/// characters with `overlap` characters carried forward between chunks.
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
}
