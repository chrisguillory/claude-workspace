"""Tests for the ``claude-remote-bash discover`` command output."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from claude_remote_bash import cache, client_config
from claude_remote_bash.cli import main as cli_main
from claude_remote_bash.discovery import DiscoveredHost
from typer.testing import CliRunner


def _host(alias: str, *, is_self: bool = False) -> DiscoveredHost:
    """Construct a minimal DiscoveredHost for assertions."""
    return DiscoveredHost(
        alias=alias,
        hostname=f'{alias}.local.',
        ips=[f'192.168.4.{ord(alias[-1])}'],
        port=60000 + ord(alias[-1]),
        version='0.3.1',
        is_self=is_self,
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def discover_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect ``CLIENT_CONFIG`` and ``HOSTS_CACHE`` to tmp paths.

    Returns the tmp ``client_config.json`` path so tests can write group
    fixtures to it.
    """
    config_path = tmp_path / 'client_config.json'
    monkeypatch.setattr(client_config, 'CLIENT_CONFIG', config_path)
    monkeypatch.setattr(cli_main, 'CLIENT_CONFIG', config_path)
    monkeypatch.setattr(cache, 'HOSTS_CACHE', tmp_path / 'hosts-cache.json')
    return config_path


@pytest.fixture
def fake_hosts(monkeypatch: pytest.MonkeyPatch) -> list[DiscoveredHost]:
    """Stub ``browse_hosts`` with a configurable list and return the list for mutation."""
    hosts: list[DiscoveredHost] = []

    async def _stub(timeout: float) -> Sequence[DiscoveredHost]:
        return hosts

    monkeypatch.setattr(cli_main, 'browse_hosts', _stub)
    return hosts


class TestText:
    def test_daemons_only_no_config(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3')])
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'Found 2 daemon(s)' in result.stdout
        assert 'M2' in result.stdout
        assert '(self)' in result.stdout
        assert 'No groups configured.' in result.stdout
        assert str(discover_env) in result.stdout

    def test_daemons_and_groups(self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3'), _host('M4')])
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3', 'M4'], 'workers': ['M3', 'M4']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'Groups (2):' in result.stdout
        # Fleet line has self-marker on M2.
        fleet_line = next(line for line in result.stdout.splitlines() if line.lstrip().startswith('fleet'))
        assert 'M2 (self)' in fleet_line
        assert 'M3' in fleet_line
        assert 'M4' in fleet_line
        assert '(no daemon)' not in fleet_line
        assert 'Use with `execute --target <group>`.' in result.stdout

    def test_group_member_missing_daemon_marked(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3')])
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3', 'M99']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'M99 (no daemon)' in result.stdout
        # Present members are not falsely marked.
        assert 'M2 (no daemon)' not in result.stdout
        assert 'M3 (no daemon)' not in result.stdout

    def test_malformed_config_warns_and_continues(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.append(_host('M2', is_self=True))
        discover_env.write_text('{"groups": "not-a-mapping"}')
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'Found 1 daemon(s)' in result.stdout
        assert 'Warning:' in result.stderr
        assert 'Groups (' not in result.stdout

    def test_no_daemons_still_renders_groups(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        # fake_hosts intentionally empty.
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'No daemons found on the network.' in result.stdout
        assert 'Groups (1):' in result.stdout
        # Every member should be marked unreachable since no daemons were found.
        assert 'M2 (no daemon)' in result.stdout
        assert 'M3 (no daemon)' in result.stdout


class TestJson:
    def test_groups_included_in_json(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3')])
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3']}}))
        result = runner.invoke(cli_main.app, ['discover', '--format', 'json'])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert [d['alias'] for d in payload['daemons']] == ['M2', 'M3']
        assert payload['groups'] == [{'name': 'fleet', 'members': ['M2', 'M3']}]

    def test_empty_groups_field_present(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.append(_host('M2', is_self=True))
        result = runner.invoke(cli_main.app, ['discover', '--format', 'json'])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload['groups'] == []
