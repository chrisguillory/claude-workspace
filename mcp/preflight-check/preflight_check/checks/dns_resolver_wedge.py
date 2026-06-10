"""Detect and repair a wedged macOS DNS resolver — mDNSResponder stuck after sleep/wake."""

from __future__ import annotations

__all__ = [
    'DnsResolverWedge',
]

import socket
import subprocess
import sys
import time
from typing import ClassVar

from preflight_check.checks.base import Finding, FixFailedError

PROBE_HOST = 'one.one.one.one'
RAW_IP = '1.1.1.1'
PROBE_PORT = 443
CONNECT_TIMEOUT = 3.0
RESTART = 'sudo killall -9 mDNSResponder'
VERIFY_DEADLINE = 15.0
VERIFY_POLL = 0.5


class DnsResolverWedge:
    """macOS mDNSResponder wedges on sleep/wake: getaddrinfo fails while the network is up.

    The signature: name resolution (the Python getaddrinfo path every tool uses) fails, yet a
    raw-IP connection succeeds — so it is the local resolver, not the network or the DNS server.
    """

    id: ClassVar[str] = 'dns_resolver_wedge'
    summary: ClassVar[str] = 'macOS DNS resolver (mDNSResponder) wedged after sleep/wake'

    def detect(self) -> Finding:
        if sys.platform != 'darwin':
            return Finding(
                check_id=self.id,
                severity='ok',
                title='not applicable (non-macOS)',
                detail='dns_resolver_wedge checks the macOS mDNSResponder resolver.',
                remedy=None,
            )
        if _resolves(PROBE_HOST):
            return Finding(
                check_id=self.id,
                severity='ok',
                title='DNS resolver healthy',
                detail=f'getaddrinfo({PROBE_HOST}) resolves.',
                remedy=None,
            )
        if _reachable(RAW_IP, PROBE_PORT):
            return Finding(
                check_id=self.id,
                severity='critical',
                title='DNS resolver wedged',
                detail=(
                    f'getaddrinfo({PROBE_HOST}) fails but {RAW_IP}:{PROBE_PORT} is reachable — '
                    'mDNSResponder is wedged (common after sleep/wake). Every getaddrinfo-based tool will fail.'
                ),
                remedy=f'{RESTART}   (or: preflight-check fix dns_resolver_wedge)',
            )
        return Finding(
            check_id=self.id,
            severity='warning',
            title='network unreachable',
            detail=(
                f'Neither {PROBE_HOST} nor {RAW_IP}:{PROBE_PORT} is reachable — '
                'looks like a network outage, not a resolver wedge.'
            ),
            remedy=None,
        )

    def fix(self) -> None:
        """SIGKILL mDNSResponder so launchd respawns it fresh; verify recovery by re-detection.

        The only restart SIP permits: ``launchctl kickstart``/``kill`` are refused even as root,
        and SIGTERM returns 0 yet is silently ignored (PID unchanged) — exit codes lie here, so
        success is only ever the re-detect coming back healthy.
        """
        before = _resolver_pid()
        subprocess.run(['sudo', 'killall', '-9', 'mDNSResponder'], check=True)
        deadline = time.monotonic() + VERIFY_DEADLINE
        while time.monotonic() < deadline:
            time.sleep(VERIFY_POLL)
            if self.detect().severity == 'ok':
                return
        raise FixFailedError(
            f'resolver still wedged {VERIFY_DEADLINE:.0f}s after {RESTART!r} '
            f'(mDNSResponder pid {before} -> {_resolver_pid()}) — reboot the host'
        )


def _resolves(host: str) -> bool:
    # getaddrinfo failure IS the signal we detect (expected normal flow), so we classify it, not raise.
    try:
        socket.getaddrinfo(host, PROBE_PORT, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


def _reachable(ip: str, port: int) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def _resolver_pid() -> int | None:
    # pgrep exit 1 means no such process — an expected state mid-respawn, not an error.
    result = subprocess.run(['pgrep', '-x', 'mDNSResponder'], capture_output=True, text=True, check=False)
    pids = result.stdout.split()
    return int(pids[0]) if pids else None
