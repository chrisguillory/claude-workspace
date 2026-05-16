"""Tests for the ``claude-remote-bash discover`` command output."""

from __future__ import annotations

import json
import textwrap
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
        expected = textwrap.dedent(f"""\
            Found 2 daemon(s):

              M2           192.168.4.50:60050  (M2.local.)  v0.3.1  (self)
              M3           192.168.4.51:60051  (M3.local.)  v0.3.1

            No groups configured.
            Define groups in {discover_env} to target multiple hosts at once.
        """)
        assert result.exit_code == 0
        assert result.stdout == expected

    def test_daemons_and_groups(self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3'), _host('M4')])
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3', 'M4'], 'workers': ['M3', 'M4']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        expected = textwrap.dedent("""\
            Found 3 daemon(s):

              M2           192.168.4.50:60050  (M2.local.)  v0.3.1  (self)
              M3           192.168.4.51:60051  (M3.local.)  v0.3.1
              M4           192.168.4.52:60052  (M4.local.)  v0.3.1

            Groups (2):

              fleet    M2,M3,M4
              workers  M3,M4

            Use with `execute --target <group>`.
        """)
        assert result.exit_code == 0
        assert result.stdout == expected

    def test_group_member_missing_daemon_marked(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3')])
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3', 'M99']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        fleet_line = next(line for line in result.stdout.splitlines() if line.lstrip().startswith('fleet'))
        assert result.exit_code == 0
        assert fleet_line == '  fleet  M2,M3,M99 (no daemon)'

    def test_malformed_config_warns_and_continues(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.append(_host('M2', is_self=True))
        discover_env.write_text('{"groups": "not-a-mapping"}')
        result = runner.invoke(cli_main.app, ['discover'])
        expected_stdout = textwrap.dedent(f"""\
            Found 1 daemon(s):

              M2           192.168.4.50:60050  (M2.local.)  v0.3.1  (self)

            No groups configured.
            Define groups in {discover_env} to target multiple hosts at once.
        """)
        expected_stderr = f'Warning: {discover_env}: Input should be an object\n'
        assert result.exit_code == 0
        assert result.stdout == expected_stdout
        assert result.stderr == expected_stderr

    def test_no_daemons_still_renders_groups(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        # fake_hosts intentionally empty.
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3']}}))
        result = runner.invoke(cli_main.app, ['discover'])
        expected = textwrap.dedent("""\
            No daemons found on the network.

            If a daemon should be visible:
              - Verify the daemon is running on the target:
                  `pgrep -f claude-remote-bash-daemon`
              - Verify network reachability: `ping <target>.local`
              - Ensure client and target are on the same LAN segment (mDNS doesn't cross subnets).

            Groups (1):

              fleet  M2 (no daemon),M3 (no daemon)

            Use with `execute --target <group>`.
        """)
        assert result.exit_code == 0
        assert result.stdout == expected


class TestJson:
    def test_groups_included_in_json(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.extend([_host('M2', is_self=True), _host('M3')])
        discover_env.write_text(json.dumps({'groups': {'fleet': ['M2', 'M3']}}))
        result = runner.invoke(cli_main.app, ['discover', '--format', 'json'])
        payload = json.loads(result.stdout)
        assert result.exit_code == 0
        assert [d['alias'] for d in payload['daemons']] == ['M2', 'M3']
        assert payload['groups'] == [{'name': 'fleet', 'members': ['M2', 'M3']}]

    def test_empty_groups_field_present(
        self, runner: CliRunner, discover_env: Path, fake_hosts: list[DiscoveredHost]
    ) -> None:
        fake_hosts.append(_host('M2', is_self=True))
        result = runner.invoke(cli_main.app, ['discover', '--format', 'json'])
        payload = json.loads(result.stdout)
        assert result.exit_code == 0
        assert payload['groups'] == []
