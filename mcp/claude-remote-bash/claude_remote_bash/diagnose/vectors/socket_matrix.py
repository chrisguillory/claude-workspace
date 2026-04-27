from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Sequence
from pathlib import Path

from claude_remote_bash.diagnose.system import resolve_interpreter, which_all
from claude_remote_bash.diagnose.types import Status, VectorResult
from claude_remote_bash.discovery import DiscoveredHost, browse_hosts

__all__ = [
    'check_socket_matrix',
]


_INTERNET_CONTROL_HOST = '8.8.8.8'
_INTERNET_CONTROL_PORT = 443
_CONNECT_TIMEOUT_SECONDS = 2.0
_SUBPROCESS_TIMEOUT_SECONDS = 5.0


def check_socket_matrix() -> VectorResult:
    """Probe TCP connect from each Python interpreter to internet + each known peer."""
    interpreters = _unique_interpreters()
    if not interpreters:
        return VectorResult(
            name='socket-matrix',
            status=Status.INFO,
            summary='no resolvable Python interpreter; matrix not run',
            detail='',
            fix_suggestion='',
        )

    peers = asyncio.run(browse_hosts(timeout=3.0))
    targets = _build_targets(peers)

    rows: list[str] = []
    failures = 0
    for interp in interpreters:
        cells = [_probe(interp, host, port) for _, host, port in targets]
        if any(c != 'CONNECTED' for c in cells):
            failures += 1
        rows.append(_format_row(interp, targets, cells))

    detail = '\n\n'.join(rows)
    if failures:
        return VectorResult(
            name='socket-matrix',
            status=Status.WARN,
            summary=f'{failures}/{len(interpreters)} interpreter(s) had failed probes',
            detail=detail,
            fix_suggestion=(
                'If failures match private-range targets only, the cause is likely macOS Local\n'
                'Network Privacy. Run `claude-remote-bash diagnose --vector local-network`.'
            ),
        )

    return VectorResult(
        name='socket-matrix',
        status=Status.OK,
        summary=f'all {len(interpreters) * len(targets)} probes connected',
        detail=detail,
        fix_suggestion='',
    )


def _unique_interpreters() -> Sequence[Path]:
    """Resolve every claude-remote-bash shim's Python interpreter, deduped, in PATH order."""
    shims = which_all('claude-remote-bash')
    seen: dict[Path, None] = {}
    for shim in shims:
        interp = resolve_interpreter(Path(shim))
        if interp is not None:
            seen.setdefault(interp, None)
    return list(seen.keys())


def _build_targets(peers: Sequence[DiscoveredHost]) -> Sequence[tuple[str, str, int]]:
    """Return [(label, host, port), ...] starting with the internet control."""
    out: list[tuple[str, str, int]] = [
        (
            f'internet ({_INTERNET_CONTROL_HOST}:{_INTERNET_CONTROL_PORT})',
            _INTERNET_CONTROL_HOST,
            _INTERNET_CONTROL_PORT,
        ),
    ]
    for peer in peers:
        if peer.ips:
            ip = peer.ips[0]
            out.append((f'{peer.alias} ({ip}:{peer.port})', ip, peer.port))
    return out


def _format_row(interp: Path, targets: Sequence[tuple[str, str, int]], cells: Sequence[str]) -> str:
    lines = [str(interp)]
    label_width = max((len(label) for label, _, _ in targets), default=0)
    for (label, _, _), outcome in zip(targets, cells, strict=True):
        lines.append(f'  {label:<{label_width}}  {outcome}')
    return '\n'.join(lines)


def _probe(python: Path, host: str, port: int) -> str:
    """Run a tiny socket-connect script under `python`; return outcome string."""
    script = (
        f'import socket\n'
        f's = socket.socket()\n'
        f's.settimeout({_CONNECT_TIMEOUT_SECONDS})\n'
        f'try:\n'
        f'    s.connect(({host!r}, {port}))\n'
        f'    print("CONNECTED")\n'
        f'except OSError as e:\n'
        f'    print(f"{{type(e).__name__}}: {{e}}")\n'
    )
    result = subprocess.run(  # noqa: S603 — python is the resolved interpreter
        [str(python), '-c', script],
        capture_output=True,
        text=True,
        check=False,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    return result.stdout.strip() or f'<no output; rc={result.returncode}>'
