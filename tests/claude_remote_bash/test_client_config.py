"""Tests for `claude_remote_bash.client_config.ClientConfig.load`."""

from __future__ import annotations

from pathlib import Path

import pytest
from claude_remote_bash import client_config
from claude_remote_bash.client_config import ClientConfig
from claude_remote_bash.exceptions import ConfigError


class TestLoad:
    def test_file_absent_returns_empty(self, tmp_config_file: Path) -> None:
        assert not tmp_config_file.exists()
        assert ClientConfig.load().groups == {}

    def test_empty_object_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{}')
        assert ClientConfig.load().groups == {}

    def test_empty_groups_returns_empty(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {}}')
        assert ClientConfig.load().groups == {}

    def test_happy_path(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"fleet": ["M2", "M3", "M4"], "workers": ["M3", "M4"]}}')
        cfg = ClientConfig.load()
        assert dict(cfg.groups) == {'fleet': ['M2', 'M3', 'M4'], 'workers': ['M3', 'M4']}

    def test_group_keys_are_lowercased(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"Fleet": ["M2"], "WORKERS": ["M3"]}}')
        cfg = ClientConfig.load()
        assert 'fleet' in cfg.groups
        assert 'workers' in cfg.groups
        assert 'Fleet' not in cfg.groups

    def test_groups_as_list_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": ["fleet", "workers"]}')
        with pytest.raises(ConfigError):
            ClientConfig.load()

    def test_groups_as_string_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": "not a dict"}')
        with pytest.raises(ConfigError):
            ClientConfig.load()

    def test_self_reference_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"fleet": ["M2", "fleet"]}}')
        with pytest.raises(ConfigError, match=r"group 'fleet' references group 'fleet'"):
            ClientConfig.load()

    def test_cross_reference_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"fleet": ["M2"], "all": ["fleet", "M3"]}}')
        with pytest.raises(ConfigError, match=r"group 'all' references group 'fleet'"):
            ClientConfig.load()

    def test_cross_reference_detected_after_case_normalization(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {"Fleet": ["M2"], "all": ["FLEET"]}}')
        with pytest.raises(ConfigError, match=r"group 'all' references group 'FLEET'"):
            ClientConfig.load()

    def test_unknown_top_level_key_rejected(self, tmp_config_file: Path) -> None:
        tmp_config_file.write_text('{"groups": {}, "future_setting": 42}')
        with pytest.raises(ConfigError):
            ClientConfig.load()


@pytest.fixture
def tmp_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``CLIENT_CONFIG`` to a tmp_path location for the duration of the test."""
    target = tmp_path / 'client_config.json'
    monkeypatch.setattr(client_config, 'CLIENT_CONFIG', target)
    return target
