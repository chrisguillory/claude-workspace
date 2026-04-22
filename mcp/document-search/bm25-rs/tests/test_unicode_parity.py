# mypy: ignore-errors
"""Regression tests for Unicode-handling parity with fastembed.

Covers four divergences found by an audit after v0.1.1 (collision fix) shipped:

- A1: ``TOKEN_MAX_LENGTH`` was compared against ``token.len()`` (UTF-8 bytes);
  fastembed compares against ``len(token)`` (Python char count). CJK and
  accented tokens that fit in the 40-char limit but exceed 40 bytes were
  silently dropped.
- A2: ``is_word_char`` returned true for any ``char::is_alphanumeric()``,
  which includes Unicode ``Mn``/``Mc`` (combining mark) general categories.
  Python's ``\\w`` only matches ``L*`` / ``N*`` / ``_`` — marks are NOT
  word chars. Scripts that use combining marks as integral word components
  (Devanagari, Thai, Arabic diacritics) tokenized to one long token in
  bm25-rs and to multiple shorter tokens in fastembed.
- A3: ``counts: HashMap`` iteration is nondeterministic; fastembed's
  Python ``dict`` preserves insertion order (Python 3.7+). The ``(indices,
  values)`` array ordering differed across bm25-rs processes and differed
  from fastembed's stable order.
- A4: ``is_punctuation`` used hand-rolled codepoint ranges that missed
  real punctuation (Devanagari danda, Arabic/Hebrew punctuation) and
  wrongly included non-punctuation codepoints in 0x00A1-0x00BF (e.g.
  ``²`` U+00B2 is ``No`` — Other Number, not ``P*``).
"""

from __future__ import annotations

from collections.abc import Mapping

import bm25_rs
import pytest
from fastembed import SparseTextEmbedding


@pytest.fixture(scope='module')
def rust_model() -> bm25_rs.BM25Model:
    return bm25_rs.BM25Model()


@pytest.fixture(scope='module')
def fastembed_model() -> SparseTextEmbedding:
    return SparseTextEmbedding('Qdrant/bm25')


# ---------------------------------------------------------------------------
# A1: TOKEN_MAX_LENGTH — byte count vs char count
# ---------------------------------------------------------------------------


# Japanese compound word: 17 characters, 51 UTF-8 bytes.
# fastembed keeps it (<=40 chars); bm25-rs drops it (>40 bytes).
CJK_LONG_TOKEN = '日本語プログラミング言語処理系開発'


def test_a1_length_filter_uses_char_count_not_bytes(
    rust_model: bm25_rs.BM25Model,
    fastembed_model: SparseTextEmbedding,
) -> None:
    """Token ≤40 chars but >40 UTF-8 bytes must be kept (matches fastembed)."""
    assert len(CJK_LONG_TOKEN) <= 40, 'Precondition: token fits fastembed char limit'
    assert len(CJK_LONG_TOKEN.encode('utf-8')) > 40, 'Precondition: token exceeds bm25-rs byte limit'

    fe = _fastembed_dict(fastembed_model, CJK_LONG_TOKEN)
    rs = _rust_dict(rust_model, CJK_LONG_TOKEN)

    assert fe, 'fastembed should emit at least one token for a valid-length CJK word'
    assert set(rs.keys()) == set(fe.keys()), (
        f'CJK token length filter divergence: fastembed emitted {set(fe.keys())}, bm25-rs emitted {set(rs.keys())}. '
        f'bm25-rs is comparing UTF-8 bytes instead of char count.'
    )


# ---------------------------------------------------------------------------
# A2: is_word_char — Python \w semantics (no combining marks)
# ---------------------------------------------------------------------------


# Devanagari word "hindi": ह (Lo) + ि (Mc) + ं (Mn) + द (Lo) + ी (Mc).
# Python \w rejects Mn/Mc — word splits at those codepoints.
# bm25-rs's char::is_alphanumeric() includes Mn/Mc, so it treats the whole
# sequence as one token.
DEVANAGARI_WORD = 'हिंदी'


def test_a2_word_char_rejects_combining_marks(
    rust_model: bm25_rs.BM25Model,
    fastembed_model: SparseTextEmbedding,
) -> None:
    """Combining-mark codepoints must split tokens the same way Python does."""
    fe = _fastembed_dict(fastembed_model, DEVANAGARI_WORD)
    rs = _rust_dict(rust_model, DEVANAGARI_WORD)

    assert set(rs.keys()) == set(fe.keys()), (
        f'Devanagari tokenization divergence for {DEVANAGARI_WORD!r}: '
        f'fastembed emitted {len(fe)} token(s) {set(fe.keys())}, '
        f'bm25-rs emitted {len(rs)} token(s) {set(rs.keys())}. '
        f'bm25-rs includes Mn/Mc combining marks in is_word_char; Python \\w excludes them.'
    )


# ---------------------------------------------------------------------------
# A3: Iteration order — match fastembed's insertion order
# ---------------------------------------------------------------------------


# Five distinct stems, no expected hash collisions; each appears once.
# Fastembed emits indices in first-occurrence order (dict insertion order).
ORDERING_TEXT = 'apple banana cherry date elderberry'


def test_a3_index_ordering_matches_fastembed(
    rust_model: bm25_rs.BM25Model,
    fastembed_model: SparseTextEmbedding,
) -> None:
    """(indices, values) must be returned in the same order as fastembed."""
    rs_embeddings, _, _ = rust_model.embed_batch([ORDERING_TEXT])
    rs_indices = list(rs_embeddings[0][0])

    fe_result = next(iter(fastembed_model.embed([ORDERING_TEXT])))
    fe_indices = fe_result.indices.tolist()

    assert rs_indices == fe_indices, (
        f'Index ordering divergence for {ORDERING_TEXT!r}: '
        f'rust={rs_indices}, fastembed={fe_indices}. '
        f'bm25-rs HashMap iteration is nondeterministic; fastembed preserves '
        f'first-occurrence order via Python dict semantics.'
    )


# ---------------------------------------------------------------------------
# A4: is_punctuation — Unicode P* category, not hand-rolled ranges
# ---------------------------------------------------------------------------


# U+00B2 SUPERSCRIPT TWO: general category 'No' (Other Number), NOT punctuation.
# bm25-rs wrongly includes U+00B2 in its 0x00A1-0x00BF range → filters as punctuation.
# fastembed's is_punctuation checks unicodedata.category(c).startswith('P') → keeps it.
FALSE_POSITIVE_PUNCTUATION_TOKEN = '²'


def test_a4_false_positive_punctuation_kept(
    rust_model: bm25_rs.BM25Model,
    fastembed_model: SparseTextEmbedding,
) -> None:
    """Non-P* codepoints in 0x00A1-0x00BF (e.g. ², £, ©) must NOT be filtered."""
    fe = _fastembed_dict(fastembed_model, FALSE_POSITIVE_PUNCTUATION_TOKEN)
    rs = _rust_dict(rust_model, FALSE_POSITIVE_PUNCTUATION_TOKEN)

    assert set(rs.keys()) == set(fe.keys()), (
        f'Punctuation filter false positive on {FALSE_POSITIVE_PUNCTUATION_TOKEN!r}: '
        f'fastembed emitted {set(fe.keys())}, bm25-rs emitted {set(rs.keys())}. '
        f'U+00B2 is general category "No", not "P*"; bm25-rs hand-rolled range '
        f'0x00A1-0x00BF wrongly includes it.'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
