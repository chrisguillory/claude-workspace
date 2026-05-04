"""Tests for search-path resolution and repo-filter translation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from document_search.search_path import (
    resolve_index_paths,
    resolve_search_path,
    resolve_search_paths,
    to_repo_filter,
)


class TestSentinels:
    def test_double_star_returns_global_sentinel(self) -> None:
        assert resolve_search_path('**') == '**'

    def test_dot_resolves_to_cwd(self) -> None:
        assert resolve_search_path('.') == str(Path.cwd())

    def test_empty_string_resolves_to_cwd(self) -> None:
        # Python's pathlib treats Path('') as Path('.'), so empty string
        # silently resolves to cwd. Pinned so a future "reject empty"
        # change is an explicit decision, not a regression.
        assert resolve_search_path('') == str(Path.cwd())


class TestResolution:
    def test_existing_absolute_path(self, tmp_path: Path) -> None:
        assert resolve_search_path(str(tmp_path)) == str(tmp_path.resolve())

    def test_existing_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'sub').mkdir()
        assert resolve_search_path('sub') == str((tmp_path / 'sub').resolve())

    def test_tilde_expansion(self) -> None:
        assert resolve_search_path('~').startswith(str(Path.home()))

    def test_file_path_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'hello.txt'
        f.write_text('hi')
        assert resolve_search_path(str(f)) == str(f.resolve())


class TestRejection:
    def test_trailing_double_star_glob(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='Glob characters not supported'):
            resolve_search_path(f'{tmp_path}/**')

    def test_single_star_glob(self) -> None:
        with pytest.raises(ValueError, match='Glob characters not supported'):
            resolve_search_path('/tmp/*.py')

    def test_question_mark_glob(self) -> None:
        with pytest.raises(ValueError, match='Glob characters not supported'):
            resolve_search_path('/tmp/?.py')

    def test_bracket_glob(self) -> None:
        with pytest.raises(ValueError, match='Glob characters not supported'):
            resolve_search_path('/tmp/[ab].py')

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='Path does not exist'):
            resolve_search_path(str(tmp_path / 'does-not-exist'))

    def test_nonexistent_path_with_tilde(self) -> None:
        with pytest.raises(ValueError, match='Path does not exist'):
            resolve_search_path('~/this-path-should-not-exist-on-any-machine-12345')


class TestResolveSearchPaths:
    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match='path cannot be empty'):
            resolve_search_paths([])

    def test_global_sentinel_alone_passes(self) -> None:
        assert resolve_search_paths(['**']) == ['**']

    def test_global_sentinel_mixed_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='cannot be mixed with other paths'):
            resolve_search_paths(['**', str(tmp_path)])

    def test_resolves_each_element(self, tmp_path: Path) -> None:
        sub = tmp_path / 'sub'
        sub.mkdir()
        result = resolve_search_paths([str(tmp_path), str(sub)])
        assert result == [str(tmp_path.resolve()), str(sub.resolve())]

    def test_scope_hint_appears_in_empty_message(self) -> None:
        with pytest.raises(ValueError, match='entire collection'):
            resolve_search_paths([], scope_hint='entire collection')

    def test_scope_hint_appears_in_mix_message(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='entire collection'):
            resolve_search_paths(['**', str(tmp_path)], scope_hint='entire collection')

    def test_allow_global_false_rejects_lone_global(self) -> None:
        with pytest.raises(ValueError, match='not supported here'):
            resolve_search_paths(['**'], allow_global=False)

    def test_allow_global_false_rejects_mixed_global(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='not supported here'):
            resolve_search_paths(['**', str(tmp_path)], allow_global=False)

    def test_allow_global_false_empty_message_omits_global(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            resolve_search_paths([], allow_global=False)
        assert '**' not in str(exc_info.value)

    def test_allow_global_false_passes_through_concrete_paths(self, tmp_path: Path) -> None:
        result = resolve_search_paths([str(tmp_path)], allow_global=False)
        assert result == [str(tmp_path.resolve())]


class TestResolveIndexPaths:
    def test_directory_accepted(self, tmp_path: Path) -> None:
        assert resolve_index_paths([str(tmp_path)]) == [tmp_path.resolve()]

    def test_file_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'doc.md'
        f.write_text('hi')
        assert resolve_index_paths([str(f)]) == [f.resolve()]

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        # Caller doesn't have to redo Path-wrapping.
        result = resolve_index_paths([str(tmp_path)])
        assert all(isinstance(rp, Path) for rp in result)

    def test_global_sentinel_rejected(self) -> None:
        with pytest.raises(ValueError, match='not supported here'):
            resolve_index_paths(['**'])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match='path cannot be empty'):
            resolve_index_paths([])

    def test_glob_rejected(self) -> None:
        with pytest.raises(ValueError, match='Glob characters not supported'):
            resolve_index_paths(['/tmp/*.py'])

    def test_fifo_rejected(self, tmp_path: Path) -> None:
        # FIFOs exist (resolve_search_path's .exists() passes) but are
        # not regular files or directories — indexing can't consume them.
        fifo_path = tmp_path / 'fifo'
        os.mkfifo(fifo_path)
        with pytest.raises(ValueError, match='not a file or directory'):
            resolve_index_paths([str(fifo_path)])


class TestToRepoFilter:
    def test_str_global_sentinel_to_empty(self) -> None:
        assert to_repo_filter('**') == ''

    def test_str_passes_through(self) -> None:
        assert to_repo_filter('/Users/chris') == '/Users/chris'

    def test_str_empty_passes_through(self) -> None:
        # Translator trusts validated input — empty string is the
        # already-validated cwd-equivalent; not its job to reject.
        assert to_repo_filter('') == ''

    def test_list_with_global_to_empty(self) -> None:
        assert to_repo_filter(['**']) == []

    def test_list_global_anywhere_to_empty(self) -> None:
        # If the validator allowed "**" in a multi-element list, the
        # translator still collapses to the empty form. (The validator
        # rejects this case, so this branch is defensive.)
        assert to_repo_filter(['/foo', '**']) == []

    def test_list_passes_through(self) -> None:
        assert to_repo_filter(['/foo', '/bar']) == ['/foo', '/bar']

    def test_empty_list_passes_through(self) -> None:
        # Translator does not validate; empty list comes from validator
        # rejection, but is harmless to translate.
        assert to_repo_filter([]) == []

    def test_list_returns_new_list(self) -> None:
        # Defensive copy so callers can't mutate the input via the result.
        original = ['/foo']
        result = to_repo_filter(original)
        assert result == original
        assert result is not original
