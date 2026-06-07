from __future__ import annotations

from document_search.search_config import neighbor_chunk_targets, resolve_context_window


class TestResolveContextWindow:
    def test_context_sets_both_sides(self) -> None:
        assert resolve_context_window(context=2, before=None, after=None) == (2, 2)

    def test_explicit_before_overrides_context(self) -> None:
        assert resolve_context_window(context=2, before=1, after=None) == (1, 2)

    def test_explicit_after_overrides_context(self) -> None:
        assert resolve_context_window(context=2, before=None, after=1) == (2, 1)

    def test_zero_after_overrides_context(self) -> None:
        # The docstring's worked example: -C 2 -A 0 means two before, none after.
        # 0 must win over context, so `is None` (not falsiness) gates the override.
        assert resolve_context_window(context=2, before=None, after=0) == (2, 0)

    def test_all_zero_default(self) -> None:
        assert resolve_context_window(context=0, before=None, after=None) == (0, 0)


class TestNeighborChunkTargets:
    def test_symmetric_window(self) -> None:
        assert neighbor_chunk_targets([('a', 5)], before=1, after=1) == {'a': frozenset({4, 6})}

    def test_lower_boundary_drops_negatives(self) -> None:
        # Nothing precedes chunk 0 — negative indices are dropped, not clamped to 0.
        assert neighbor_chunk_targets([('a', 0)], before=2, after=1) == {'a': frozenset({1})}

    def test_hit_is_excluded_from_context(self) -> None:
        # Adjacent hits: each one's would-be neighbor is the other hit, so neither is
        # emitted as context (it's already returned as a hit).
        assert neighbor_chunk_targets([('a', 5), ('a', 6)], before=1, after=1) == {'a': frozenset({4, 7})}

    def test_overlapping_windows_collapse(self) -> None:
        # Hits at 5 and 7 both want 6; frozenset dedups it to a single fetch target.
        assert neighbor_chunk_targets([('a', 5), ('a', 7)], before=1, after=1) == {'a': frozenset({4, 6, 8})}

    def test_same_index_distinct_documents(self) -> None:
        # Identical chunk_index in different documents stays separate (keyed by source_path).
        assert neighbor_chunk_targets([('a', 5), ('b', 5)], before=1, after=0) == {
            'a': frozenset({4}),
            'b': frozenset({4}),
        }

    def test_no_context_yields_empty(self) -> None:
        assert neighbor_chunk_targets([('a', 5)], before=0, after=0) == {}
