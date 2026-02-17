//! BM25 sparse embedding module compatible with fastembed's Qdrant/bm25 model.
//!
//! Produces identical (indices, values) sparse vectors to fastembed's
//! `SparseTextEmbedding('Qdrant/bm25')`. Uses the same Rust crate (`rust-stemmers`)
//! that fastembed wraps via `py_rust_stemmers`, and the same MurmurHash3 algorithm
//! that fastembed calls via `mmh3`.
//!
//! Architecture:
//!   Python MCP server -> import bm25_rs (.so, loaded once)
//!     -> bm25_rs.embed_batch(texts) releases GIL, rayon parallelism
//!       -> rust-stemmers (direct) + murmur3 (inline)
//!         -> returns sparse vectors directly to Python
//!
//! Performance optimizations:
//! - mimalloc allocator (optional feature) for reduced contention
//! - Inline MurmurHash3 (no io::Cursor overhead)
//! - Flat string buffers with offset ranges (no per-token String allocation)
//! - Thread-local state (stemmer + buffers) for zero-contention parallelism
//! - `embed_batch_bytes` returns raw bytes for near-zero PyO3 conversion cost

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use rayon::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};

/// Default BM25 parameters matching fastembed Qdrant/bm25.
const DEFAULT_K: f64 = 1.2;
const DEFAULT_B: f64 = 0.75;
const DEFAULT_AVG_LEN: f64 = 256.0;
const TOKEN_MAX_LENGTH: usize = 40;

/// English stopwords from fastembed's Qdrant/bm25 model.
const ENGLISH_STOPWORDS: &str = include_str!("stopwords_en.txt");

/// Compute MurmurHash3 x86 32-bit (seed 0) matching Python's `abs(mmh3.hash(token))`.
///
/// fastembed uses `mmh3.hash(token)` which is MurmurHash3_x86_32 returning a signed
/// 32-bit int, then takes `abs()`. We replicate this exactly.
#[inline(always)]
fn token_id(token: &str) -> u32 {
    let hash_32 = murmur3_x86_32(token.as_bytes(), 0);
    (hash_32 as i32).unsigned_abs()
}

/// Inline MurmurHash3 x86 32-bit. Avoids Cursor + io::Read overhead from the
/// murmur3 crate, operating directly on byte slices.
#[inline(always)]
fn murmur3_x86_32(data: &[u8], seed: u32) -> u32 {
    let c1: u32 = 0xcc9e_2d51;
    let c2: u32 = 0x1b87_3593;
    let len = data.len();

    let mut h1 = seed;
    let n_blocks = len / 4;

    for i in 0..n_blocks {
        let start = i * 4;
        let k = u32::from_le_bytes([
            data[start],
            data[start + 1],
            data[start + 2],
            data[start + 3],
        ]);
        let mut k1 = k;
        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(c2);
        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    let tail = &data[n_blocks * 4..];
    let mut k1: u32 = 0;
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        _ => {}
    }

    h1 ^= len as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2_ae35);
    h1 ^= h1 >> 16;
    h1
}

/// Check if a character matches Python's `\w` (word character) per Unicode rules.
/// Python's `\w` with re.UNICODE matches: [a-zA-Z0-9_] plus Unicode letters/digits.
#[inline(always)]
fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Check if a character is Unicode punctuation (category "P").
/// fastembed filters tokens where ALL characters are punctuation.
fn is_punctuation(c: char) -> bool {
    if c.is_ascii_punctuation() {
        return true;
    }
    let cp = c as u32;
    matches!(cp,
        0x00A1..=0x00BF  // Latin-1 punctuation
        | 0x2010..=0x2027 // General punctuation
        | 0x2030..=0x205E // General punctuation continued
        | 0x2E00..=0x2E52 // Supplemental punctuation
        | 0x3001..=0x3003 // CJK punctuation
        | 0x3008..=0x3011 // CJK brackets
        | 0x3014..=0x301F // CJK brackets continued
        | 0xFE50..=0xFE6B // Small form variants
        | 0xFF01..=0xFF0F // Fullwidth punctuation
        | 0xFF1A..=0xFF20 // Fullwidth punctuation
        | 0xFF3B..=0xFF40 // Fullwidth brackets
        | 0xFF5B..=0xFF65 // Fullwidth punctuation
    )
}

/// Thread-safe BM25 model. Holds stopwords and scoring parameters.
///
/// Text processing uses inline char classification instead of regex to avoid
/// Mutex contention on regex's internal DFA cache in parallel workloads.
struct BM25 {
    stopwords: HashSet<String>,
    k: f64,
    b: f64,
    avg_len: f64,
}

impl BM25 {
    fn new(stopwords: HashSet<String>, k: f64, b: f64, avg_len: f64) -> Self {
        Self {
            stopwords,
            k,
            b,
            avg_len,
        }
    }

    /// Embed a single document using thread-local reusable state.
    ///
    /// Uses flat string buffers with offset ranges to avoid per-token
    /// String heap allocation. All tokens and stems are written contiguously
    /// into pre-allocated buffers that grow to high-water mark and are reused.
    fn embed_one_reuse(
        &self,
        text: &str,
        state: &mut ThreadState,
    ) -> (Vec<u32>, Vec<f64>) {
        state.token_buf.clear();
        state.token_ranges.clear();
        state.stem_buf.clear();
        state.stem_ranges.clear();

        // Step 1+2: Tokenize into flat buffer. Lowercase + split on non-word boundaries.
        let mut token_start = state.token_buf.len();
        let mut in_token = false;

        for c in text.chars() {
            if is_word_char(c) {
                if !in_token {
                    token_start = state.token_buf.len();
                    in_token = true;
                }
                for lc in c.to_lowercase() {
                    state.token_buf.push(lc);
                }
            } else if in_token {
                state.token_ranges.push((token_start, state.token_buf.len()));
                in_token = false;
            }
        }
        if in_token {
            state.token_ranges.push((token_start, state.token_buf.len()));
        }

        // Step 3: Stem -- filter stopwords, length, punctuation, then Snowball stem.
        for &(start, end) in &state.token_ranges {
            let token = &state.token_buf[start..end];
            if token.len() > TOKEN_MAX_LENGTH {
                continue;
            }
            if self.stopwords.contains(token) {
                continue;
            }
            // fastembed checks if token is a single punctuation CHARACTER.
            // Multi-char tokens like "___" are NOT filtered.
            let mut chars = token.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                // Single character token -- check if it's punctuation
                if c.is_ascii_punctuation() || is_punctuation(c) {
                    continue;
                }
            }
            let stem = state.stemmer.stem(token);
            if !stem.is_empty() {
                let stem_start = state.stem_buf.len();
                state.stem_buf.push_str(&stem);
                state.stem_ranges.push((stem_start, state.stem_buf.len()));
            }
        }

        // Step 4: BM25 term frequency scoring.
        let doc_len = state.stem_ranges.len() as f64;
        let mut counts: HashMap<&str, u32> = HashMap::with_capacity(state.stem_ranges.len() / 2);
        for &(start, end) in &state.stem_ranges {
            let stem = &state.stem_buf[start..end];
            *counts.entry(stem).or_insert(0) += 1;
        }

        let mut indices = Vec::with_capacity(counts.len());
        let mut values = Vec::with_capacity(counts.len());

        for (token, count) in &counts {
            let tf = (*count as f64 * (self.k + 1.0))
                / (*count as f64 + self.k * (1.0 - self.b + self.b * doc_len / self.avg_len));
            indices.push(token_id(token));
            values.push(tf);
        }

        (indices, values)
    }
}

/// Thread-local state: stemmer + reusable flat buffers for embed_one.
///
/// Uses flat string buffers with offset ranges to avoid per-token String allocation.
/// Buffers grow to high-water mark and are reused across texts without reallocating.
struct ThreadState {
    stemmer: Stemmer,
    /// Flat buffer holding all lowercased tokens contiguously.
    token_buf: String,
    /// (start, end) byte offsets into token_buf for each token.
    token_ranges: Vec<(usize, usize)>,
    /// Flat buffer holding all stemmed tokens contiguously.
    stem_buf: String,
    /// (start, end) byte offsets into stem_buf for each stemmed token.
    stem_ranges: Vec<(usize, usize)>,
}

impl ThreadState {
    fn new() -> Self {
        Self {
            stemmer: Stemmer::create(Algorithm::English),
            token_buf: String::with_capacity(8192),
            token_ranges: Vec::with_capacity(512),
            stem_buf: String::with_capacity(8192),
            stem_ranges: Vec::with_capacity(512),
        }
    }
}

thread_local! {
    static TL_STATE: RefCell<ThreadState> = RefCell::new(ThreadState::new());
}

/// Convert a typed slice to bytes for zero-copy PyO3 transmission.
///
/// SAFETY: T must be a plain-old-data type with no padding bytes (u32, f64, etc.).
/// The resulting bytes are only valid on the same endianness (native).
#[inline]
fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

/// Python-exposed BM25 model.
#[pyclass]
struct BM25Model {
    inner: BM25,
}

#[pymethods]
impl BM25Model {
    /// Create a new BM25 model with parameters matching fastembed Qdrant/bm25.
    ///
    /// Args:
    ///     k: Term frequency saturation parameter (default: 1.2)
    ///     b: Document length normalization (default: 0.75)
    ///     avg_doc_len: Assumed average document length in tokens (default: 256.0)
    #[new]
    #[pyo3(signature = (k=DEFAULT_K, b=DEFAULT_B, avg_doc_len=DEFAULT_AVG_LEN))]
    fn new(k: f64, b: f64, avg_doc_len: f64) -> Self {
        let stopwords: HashSet<String> = ENGLISH_STOPWORDS
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        Self {
            inner: BM25::new(stopwords, k, b, avg_doc_len),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BM25Model(k={}, b={}, avg_doc_len={})",
            self.inner.k, self.inner.b, self.inner.avg_len
        )
    }

    /// Embed a batch of texts in parallel using rayon with timing instrumentation.
    ///
    /// Releases the GIL during computation so Python threads are not blocked.
    /// Uses thread-local stemmers and regex-free text processing for
    /// contention-free parallelism.
    ///
    /// Timing is measured inside `py.allow_threads` where only BM25 work runs:
    /// - wall_secs: Instant around the entire par_iter (latency)
    /// - cpu_secs: Sum of per-task Instant durations (total compute across cores)
    /// - Ratio cpu_secs/wall_secs = effective parallelism factor
    ///
    /// Args:
    ///     texts: List of document strings to embed.
    ///
    /// Returns:
    ///     (results, wall_secs, cpu_secs) where results is a list of
    ///     (indices, values) tuples.
    #[pyo3(text_signature = "($self, texts)")]
    fn embed_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> (Vec<(Vec<u32>, Vec<f64>)>, f64, f64) {
        py.allow_threads(|| {
            let wall_t0 = std::time::Instant::now();

            let results_with_timing: Vec<((Vec<u32>, Vec<f64>), std::time::Duration)> = texts
                .par_iter()
                .map(|text| {
                    let task_t0 = std::time::Instant::now();
                    let result = TL_STATE.with(|s| {
                        let mut state = s.borrow_mut();
                        self.inner.embed_one_reuse(text, &mut state)
                    });
                    (result, task_t0.elapsed())
                })
                .collect();

            let wall_secs = wall_t0.elapsed().as_secs_f64();
            let cpu_secs: f64 = results_with_timing
                .iter()
                .map(|(_, d)| d.as_secs_f64())
                .sum();
            let results = results_with_timing
                .into_iter()
                .map(|(r, _)| r)
                .collect();

            (results, wall_secs, cpu_secs)
        })
    }

    /// Number of rayon worker threads in the global thread pool.
    #[staticmethod]
    fn thread_count() -> usize {
        rayon::current_num_threads()
    }

    /// Embed a batch of texts, returning packed bytes for near-zero conversion cost.
    ///
    /// Returns (offsets_bytes, indices_bytes, values_bytes) where:
    ///   - offsets_bytes: u32 native-endian, len(texts)+1 entries
    ///   - indices_bytes: u32 native-endian, flat packed token IDs
    ///   - values_bytes: f64 native-endian, flat packed BM25 TF scores
    ///
    /// Python unpacking:
    ///   ```python
    ///   import array
    ///   offsets = array.array('I'); offsets.frombytes(offsets_bytes)
    ///   indices = array.array('I'); indices.frombytes(indices_bytes)
    ///   values = array.array('d');  values.frombytes(values_bytes)
    ///   # text i: indices[offsets[i]:offsets[i+1]], values[offsets[i]:offsets[i+1]]
    ///   ```
    #[pyo3(text_signature = "($self, texts)")]
    fn embed_batch_bytes<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> (Bound<'py, pyo3::types::PyBytes>, Bound<'py, pyo3::types::PyBytes>, Bound<'py, pyo3::types::PyBytes>) {
        // Phase 1: Parallel compute.
        let results: Vec<(Vec<u32>, Vec<f64>)> = py.allow_threads(|| {
            texts
                .par_iter()
                .map(|text| {
                    TL_STATE.with(|s| {
                        let mut state = s.borrow_mut();
                        self.inner.embed_one_reuse(text, &mut state)
                    })
                })
                .collect()
        });

        // Phase 2: Pack into flat arrays, then transmute to bytes.
        let total_len: usize = results.iter().map(|(idx, _)| idx.len()).sum();
        let n = results.len();

        let mut offsets: Vec<u32> = Vec::with_capacity(n + 1);
        let mut offset: u32 = 0;
        for (indices, _) in &results {
            offsets.push(offset);
            offset = offset
                .checked_add(u32::try_from(indices.len()).expect("token count exceeds u32::MAX"))
                .expect("total token count exceeds u32::MAX");
        }
        offsets.push(offset);

        let mut all_indices: Vec<u32> = Vec::with_capacity(total_len);
        let mut all_values: Vec<f64> = Vec::with_capacity(total_len);
        for (indices, values) in &results {
            all_indices.extend_from_slice(indices);
            all_values.extend_from_slice(values);
        }

        // Convert to bytes for zero-copy PyO3 transmission.
        // SAFETY: u32 and f64 are POD types with no padding. Bytes are native-endian,
        // matching Python's array.array.
        let offsets_bytes = as_bytes(&offsets);
        let indices_bytes = as_bytes(&all_indices);
        let values_bytes = as_bytes(&all_values);

        (
            pyo3::types::PyBytes::new(py, offsets_bytes),
            pyo3::types::PyBytes::new(py, indices_bytes),
            pyo3::types::PyBytes::new(py, values_bytes),
        )
    }

    /// Embed texts for query (values are all 1.0, deduplicated tokens).
    ///
    /// Matches fastembed's query_embed behavior where TF weighting is not applied.
    #[pyo3(text_signature = "($self, text)")]
    fn query_embed(&self, py: Python<'_>, text: String) -> (Vec<u32>, Vec<f64>) {
        py.allow_threads(|| {
            TL_STATE.with(|s| {
                let mut state = s.borrow_mut();
                let (indices, _) = self.inner.embed_one_reuse(&text, &mut state);
                // Deduplicate indices, all values 1.0
                let unique: HashSet<u32> = indices.into_iter().collect();
                let deduped: Vec<u32> = unique.into_iter().collect();
                let values = vec![1.0; deduped.len()];
                (deduped, values)
            })
        })
    }
}

/// Python module definition.
#[pymodule]
fn bm25_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25Model>()?;
    Ok(())
}
