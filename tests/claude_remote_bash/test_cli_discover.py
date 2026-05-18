"""Tests for the ``claude-remote-bash discover`` command output."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from claude_remote_bash import cache, client_config
from claude_remote_bash.cli import main as cli_main
from claude_remote_bash.discovery import BrowseResult, DiscoveredAddress, DiscoveredHost, InterfaceKind
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def discover_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect ``CLIENT_CONFIG`` and ``HOSTS_CACHE`` to tmp paths."""
    config_path = tmp_path / 'client_config.json'
    monkeypatch.setattr(client_config, 'CLIENT_CONFIG', config_path)
    monkeypatch.setattr(cli_main, 'CLIENT_CONFIG', config_path)
    monkeypatch.setattr(cache, 'HOSTS_CACHE', tmp_path / 'hosts-cache.json')
    return config_path


@pytest.fixture
def fake_browse(monkeypatch: pytest.MonkeyPatch) -> _BrowseState:
    """Stub ``browse_hosts`` with a mutable state object the test populates."""
    state = _BrowseState()

    async def _stub(timeout: float) -> BrowseResult:
        return BrowseResult(remote_daemons=state.remote, local_daemon=state.local)

    monkeypatch.setattr(cli_main, 'browse_hosts', _stub)
    return state


class TestText:
    def test_local_and_remote_daemons(
        self,
        runner: CliRunner,
        discover_env: Path,
        fake_browse: _BrowseState,
    ) -> None:
        fake_browse.local = _host('M2')
        fake_browse.remote = [_host('M3')]
        result = runner.invoke(cli_main.app, ['discover'])
        expected = textwrap.dedent(f"""\
            Found 2 daemon(s):

              M2           192.168.4.50(ethernet):60050  (M2.local.)  v0.4.0  (self)
              M3           192.168.4.51(ethernet):60051  (M3.local.)  v0.4.0

            No groups configured.
            Define groups in {discover_env} to target multiple hosts at once.
        """)
        assert result.exit_code == 0
        assert result.stdout == expected

    def test_groups_rendered(
        self,
        runner: CliRunner,
        discover_env: Path,
        fake_browse: _BrowseState,
    ) -> None:
        fake_browse.local = _host('M2')
        fake_browse.remote = [_host('M3'), _host('M4')]
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3', 'M4'], 'workers': ['M3', 'M4']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'fleet    M2,M3,M4' in result.stdout
        assert 'workers  M3,M4' in result.stdout

    def test_group_member_missing_daemon_marked(
        self,
        runner: CliRunner,
        discover_env: Path,
        fake_browse: _BrowseState,
    ) -> None:
        fake_browse.local = _host('M2')
        fake_browse.remote = [_host('M3')]
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3', 'M99']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        fleet_line = next(line for line in result.stdout.splitlines() if line.lstrip().startswith('fleet'))
        assert result.exit_code == 0
        assert fleet_line == '  fleet  M2,M3,M99 (no daemon)'

    def test_no_daemons(
        self,
        runner: CliRunner,
        discover_env: Path,
        fake_browse: _BrowseState,
    ) -> None:
        # fake_browse intentionally empty.
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        assert result.exit_code == 0
        assert 'No daemons found on the network.' in result.stdout
        assert 'fleet  M2 (no daemon),M3 (no daemon)' in result.stdout


class TestJson:
    def test_envelope_shape(
        self,
        runner: CliRunner,
        discover_env: Path,
        fake_browse: _BrowseState,
    ) -> None:
        fake_browse.local = _host('M2')
        fake_browse.remote = [_host('M3')]
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3']}}))
        result = runner.invoke(cli_main.app, ['discover', '--format', 'json'])
        payload = json.loads(result.stdout)
        assert result.exit_code == 0
        assert payload['local_daemon']['alias'] == 'M2'
        assert [d['alias'] for d in payload['remote_daemons']] == ['M3']
        assert payload['groups'] == [{'name': 'fleet', 'members': ['M2', 'M3']}]

    def test_empty_local_and_remote(
        self,
        runner: CliRunner,
        discover_env: Path,
        fake_browse: _BrowseState,
    ) -> None:
        result = runner.invoke(cli_main.app, ['discover', '--format', 'json'])
        payload = json.loads(result.stdout)
        assert result.exit_code == 0
        assert payload['local_daemon'] is None
        assert payload['remote_daemons'] == []
        assert payload['groups'] == []


@dataclass
class _BrowseState:
    """Mutable state for the fake browse fixture — tests assign before invoking the CLI."""

    remote: list[DiscoveredHost] = field(default_factory=list)
    local: DiscoveredHost | None = None


def _host(alias: str, kind: InterfaceKind = 'ethernet') -> DiscoveredHost:
    """Construct a minimal DiscoveredHost for assertions."""
    return DiscoveredHost(
        alias=alias,
        hostname=f'{alias}.local.',
        addresses=[DiscoveredAddress(ip=f'192.168.4.{ord(alias[-1])}', kind=kind)],
        port=60000 + ord(alias[-1]),
        version='0.4.0',
        legacy=False,
    )
