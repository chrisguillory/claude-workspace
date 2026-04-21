# mypy: ignore-errors
"""Regression tests for hash-collision handling in sparse-vector output.

Qdrant rejects sparse vectors with duplicate indices ("must be unique").
When two distinct stems hash to the same token_id under abs(mmh3.hash()),
the embedder must collapse them into a single (index, value) entry — not
emit both.

The reference implementation (fastembed) collapses via `dict[int, float]`
semantics: each write to the same token_id overwrites the previous value
(last-write-wins). bm25-rs must match this behavior.

Collision data
--------------
These stem pairs were found by brute-forcing /usr/share/dict/words through
Snowball English and `abs(mmh3.hash(stem))`. Both stems in each pair are
their own stem (stemmer is a no-op), alphabetic, and not stopwords, so
they pass the pipeline unmodified and hash to the shared token_id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import bm25_rs
import mmh3
import pytest
from fastembed import SparseTextEmbedding

# (word_1, word_2, shared_token_id) — verified to collide via abs(mmh3.hash()).
COLLISION_PAIRS: Sequence[tuple[str, str, int]] = (
    ('haught', 'leeway', 1649890854),
    ('killick', 'noctidi', 1614794467),
    ('nawt', 'nympheal', 187790963),
    ('burhinus', 'phytin', 791486376),
)


@pytest.fixture(scope='module')
def rust_model() -> bm25_rs.BM25Model:
    return bm25_rs.BM25Model()


@pytest.fixture(scope='module')
def fastembed_model() -> SparseTextEmbedding:
    return SparseTextEmbedding('Qdrant/bm25')


@pytest.mark.parametrize(('word_a', 'word_b', 'expected_token_id'), COLLISION_PAIRS)
def test_collision_pair_hashes_are_equal(word_a: str, word_b: str, expected_token_id: int) -> None:
    """Sanity check: the chosen test inputs actually collide under abs(mmh3.hash())."""
    assert abs(mmh3.hash(word_a)) == expected_token_id
    assert abs(mmh3.hash(word_b)) == expected_token_id


@pytest.mark.parametrize(('word_a', 'word_b', 'expected_token_id'), COLLISION_PAIRS)
def test_colliding_stems_produce_unique_indices(
    rust_model: bm25_rs.BM25Model,
    word_a: str,
    word_b: str,
    expected_token_id: int,
) -> None:
    """Regression: sparse-vector indices must be unique even when stems collide.

    Qdrant rejects upserts containing duplicate indices with the error
    ``Validation error in body: [points[N].vectors.[].indices: must be unique]``.
    bm25-rs v0.1.0 emitted `[token_id, token_id]` for colliding stems, which
    broke every indexing run touching a corpus large enough to trigger
    birthday-paradox collisions (~/.claude/projects JSONLs, in practice).
    """
    text = f'{word_a} {word_b}'
    embeddings, _, _ = rust_model.embed_batch([text])
    indices, _values = embeddings[0]

    indices_list = list(indices)
    assert expected_token_id in indices_list, (
        f'Expected the collision token_id {expected_token_id} in output, got {indices_list}'
    )
    assert len(indices_list) == len(set(indices_list)), (
        f'bm25-rs emitted duplicate indices for colliding stems {word_a!r}/{word_b!r}: '
        f'indices={indices_list}. Qdrant rejects this as "indices must be unique".'
    )


def test_no_duplicate_indices_when_stems_match_fastembed_output_length(
    rust_model: bm25_rs.BM25Model,
    fastembed_model: SparseTextEmbedding,
) -> None:
    """Cross-check: for any colliding input, bm25-rs and fastembed emit the
    same *number* of indices.

    Existing tests compare via `dict(zip(indices, values))` which silently
    collapses duplicate keys, so `len(dict) == len(dict)` passes even when
    one side emits duplicates and the other doesn't. This test compares raw
    lengths before the collapse to catch the specific failure mode.
    """
    for word_a, word_b, _ in COLLISION_PAIRS:
        text = f'{word_a} {word_b}'
        rs_embeddings, _, _ = rust_model.embed_batch([text])
        rs_indices_len = len(list(rs_embeddings[0][0]))

        fe_indices_len = len(next(iter(fastembed_model.embed([text]))).indices.tolist())

        assert rs_indices_len == fe_indices_len, (
            f'Index count mismatch for {text!r}: rust emitted {rs_indices_len}, '
            f'fastembed emitted {fe_indices_len}. bm25-rs is not collapsing colliding stems.'
        )


def _rust_dict(model: bm25_rs.BM25Model, text: str) -> Mapping[int, float]:
    embeddings, _, _ = model.embed_batch([text])
    if not embeddings:
        return {}
    indices, values = embeddings[0]
    return dict(zip(indices, values, strict=True))


def _fastembed_dict(model: SparseTextEmbedding, text: str) -> Mapping[int, float]:
    results = list(model.embed([text]))
    if not results:
        return {}
    sparse = results[0]
    return dict(zip(sparse.indices.tolist(), sparse.values.tolist(), strict=True))
