"""Tests for the search-path validator."""

from __future__ import annotations

from pathlib import Path

import pytest
from document_search.search_path import resolve_search_path


class TestSentinels:
    def test_double_star_returns_global_sentinel(self) -> None:
        assert resolve_search_path('**') == '**'

    def test_none_defaults_to_cwd(self) -> None:
        assert resolve_search_path(None) == str(Path.cwd())


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
