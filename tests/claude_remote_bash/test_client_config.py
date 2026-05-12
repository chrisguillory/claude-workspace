"""Tests for `claude_remote_bash.client_config.load_groups`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from claude_remote_bash import client_config
from claude_remote_bash.client_config import (
    GroupOfGroupsError,
    MalformedClientConfigError,
    load_groups,
)


class TestLoadGroups:
    """End-to-end tests of the client_config.json loader against a tmp_path file."""

    def test_file_absent_returns_empty(self, tmp_config_file: Path) -> None:
        assert not tmp_config_file.exists()
        assert load_groups() == {}

    def test_file_empty_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('')
        assert load_groups() == {}

    def test_file_whitespace_only_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('   \n\n  ')
        assert load_groups() == {}

    def test_empty_object_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{}')
        assert load_groups() == {}

    def test_empty_groups_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {}}')
        assert load_groups() == {}

    def test_null_groups_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": null}')
        assert load_groups() == {}

    def test_missing_groups_key_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"other_setting": 42}')
        assert load_groups() == {}

    def test_happy_path(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"fleet": ["M2", "M3", "M4"], "workers": ["M3", "M4"]}}')
        assert load_groups() == {'fleet': ['M2', 'M3', 'M4'], 'workers': ['M3', 'M4']}

    def test_group_keys_are_lowercased(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"Fleet": ["M2"], "WORKERS": ["M3"]}}')
        result = load_groups()
        assert 'fleet' in result
        assert 'workers' in result
        assert 'Fleet' not in result

    def test_groups_as_list_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": ["fleet", "workers"]}')
        with pytest.raises(MalformedClientConfigError, match=r"'groups' must be a JSON object, got list"):
            load_groups()

    def test_groups_as_string_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": "not a dict"}')
        with pytest.raises(MalformedClientConfigError, match=r"'groups' must be a JSON object, got str"):
            load_groups()

    def test_groups_as_number_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": 42}')
        with pytest.raises(MalformedClientConfigError, match=r"'groups' must be a JSON object, got int"):
            load_groups()

    def test_self_reference_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"fleet": ["M2", "fleet"]}}')
        with pytest.raises(GroupOfGroupsError, match=r"group 'fleet' references group 'fleet'"):
            load_groups()

    def test_cross_reference_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"fleet": ["M2"], "all": ["fleet", "M3"]}}')
        with pytest.raises(GroupOfGroupsError, match=r"group 'all' references group 'fleet'"):
            load_groups()

    def test_cross_reference_detected_after_case_normalization(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"Fleet": ["M2"], "all": ["FLEET"]}}')
        with pytest.raises(GroupOfGroupsError, match=r"group 'all' references group 'FLEET'"):
            load_groups()

    def test_malformed_json_raises(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{not json')
        with pytest.raises(json.JSONDecodeError):
            load_groups()


@pytest.fixture
def tmp_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect CLIENT_CONFIG_FILE to a tmp_path location for the duration of the test."""
    target = tmp_path / 'client_config.json'
    monkeypatch.setattr(client_config, 'CLIENT_CONFIG_FILE', target)
    return target
