# mypy: ignore-errors
"""Equivalence tests: bm25_rs must produce identical output to fastembed Qdrant/bm25.

These tests compare sparse vectors from both implementations across a variety of
inputs to ensure the Rust extension is a drop-in replacement.
"""

from __future__ import annotations

from collections.abc import Mapping

import bm25_rs
import pytest
from fastembed import SparseTextEmbedding

# --- Fixtures ---


@pytest.fixture(scope='module')
def fastembed_model() -> SparseTextEmbedding:
    """fastembed BM25 model (reference implementation)."""
    return SparseTextEmbedding('Qdrant/bm25')


@pytest.fixture(scope='module')
def rust_model() -> bm25_rs.BM25Model:
    """Rust BM25 model (must match fastembed)."""
    return bm25_rs.BM25Model()


# --- Test Cases ---

TEXTS = [
    # Basic
    'hello world',
    'The quick brown fox jumps over the lazy dog',
    # Stopword-heavy (most tokens filtered)
    'this is a the an and but or',
    # Single token
    'test',
    # Empty after filtering
    '',
    # Punctuation
    "Hello, world! How are you? I'm fine.",
    # Numbers and mixed
    'Python 3.13 released in 2024 with free-threading support',
    # Unicode
    'café résumé naïve über straße',
    # Repeated tokens (TF > 1)
    'test test test test test document',
    # Long document
    ' '.join(['word'] * 500),
    # Special characters
    'C++ is not C# but both are programming languages',
    # Mixed case
    'UPPERCASE lowercase MiXeD CaSe',
    # Underscores (kept by \\w)
    'snake_case variable_name __init__',
    # Multi-underscore separators (critical: NOT filtered by fastembed)
    '___ section separator ___',
    '__ double __ underscore __',
    # Only underscores
    '_ __ ___ ____',
    # Very long token (>40 chars, should be filtered)
    'supercalifragilisticexpialidociousandmore characters here',
    # Tab and newline whitespace
    'hello\tworld\nnew line\there',
    # JSON-like content (common in our indexing)
    '{"key": "value", "count": 42, "nested": {"a": "b"}}',
    # Markdown-like content
    '# Heading\n\n**bold** and *italic* text with [links](http://example.com)',
    # Code snippet
    'def embed_batch(self, texts: list[str]) -> list[tuple]:',
    # Real-world chunk from our corpus
    'The BM25 algorithm computes term frequency with document length normalization. '
    'Parameters k=1.2 and b=0.75 control saturation and length normalization respectively.',
]


def _fastembed_embed(model: SparseTextEmbedding, text: str) -> Mapping[int, float]:
    """Embed with fastembed, return {token_id: score} dict."""
    results = list(model.embed([text]))
    if not results:
        return {}
    sparse = results[0]
    return dict(zip(sparse.indices.tolist(), sparse.values.tolist()))


def _rust_embed(model: bm25_rs.BM25Model, text: str) -> Mapping[int, float]:
    """Embed with bm25_rs, return {token_id: score} dict."""
    embeddings, _, _ = model.embed_batch([text])
    if not embeddings:
        return {}
    indices, values = embeddings[0]
    return dict(zip(indices, values))


@pytest.mark.parametrize('text', TEXTS, ids=[t[:40] for t in TEXTS])
def test_sparse_vectors_match(
    fastembed_model: SparseTextEmbedding,
    rust_model: bm25_rs.BM25Model,
    text: str,
) -> None:
    """Each text must produce identical sparse vectors from both implementations."""
    fe = _fastembed_embed(fastembed_model, text)
    rs = _rust_embed(rust_model, text)

    # Same token IDs
    assert set(fe.keys()) == set(rs.keys()), (
        f'Token ID mismatch.\n'
        f'  fastembed only: {set(fe.keys()) - set(rs.keys())}\n'
        f'  rust only: {set(rs.keys()) - set(fe.keys())}'
    )

    # Same scores (within floating-point tolerance)
    for token_id in fe:
        assert abs(fe[token_id] - rs[token_id]) < 1e-10, (
            f'Score mismatch for token {token_id}: fastembed={fe[token_id]}, rust={rs[token_id]}'
        )


def test_batch_matches_individual(
    fastembed_model: SparseTextEmbedding,
    rust_model: bm25_rs.BM25Model,
) -> None:
    """Batch embedding must produce same results as individual embedding."""
    batch_results, _, _ = rust_model.embed_batch(TEXTS)
    for i, text in enumerate(TEXTS):
        individual, _, _ = rust_model.embed_batch([text])
        batch_indices = sorted(batch_results[i][0])
        individual_indices = sorted(individual[0][0])
        assert batch_indices == individual_indices, f'Batch vs individual mismatch at index {i}'


def test_empty_input(rust_model: bm25_rs.BM25Model) -> None:
    """Empty batch returns empty list."""
    embeddings, wall, cpu = rust_model.embed_batch([])
    assert embeddings == []
    assert wall >= 0.0
    assert cpu >= 0.0


def test_parallel_deterministic(rust_model: bm25_rs.BM25Model) -> None:
    """Multiple runs of the same batch produce identical results (rayon determinism)."""
    texts = TEXTS * 3  # 57 texts
    r1, _, _ = rust_model.embed_batch(texts)
    r2, _, _ = rust_model.embed_batch(texts)
    for i in range(len(texts)):
        assert sorted(r1[i][0]) == sorted(r2[i][0])
        d1 = dict(zip(r1[i][0], r1[i][1]))
        d2 = dict(zip(r2[i][0], r2[i][1]))
        for k in d1:
            assert d1[k] == d2[k]
