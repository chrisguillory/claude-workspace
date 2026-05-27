"""Service-layer entry point: resolve a target selector, execute on every host, aggregate results."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

from cc_lib.schemas import ClosedModel

from claude_remote_bash.cache import HostsCache
from claude_remote_bash.client import execute_at
from claude_remote_bash.client_config import ClientConfig
from claude_remote_bash.exceptions import RemoteBashError
from claude_remote_bash.resolve import (
    browse_and_cache,
    browse_fresh_and_cache,
    lookup_alias,
)
from claude_remote_bash.selector import SelectorError, UnknownAtomError
from claude_remote_bash.selector import parse as parse_selector

__all__ = [
    'DispatchResult',
    'DispatchService',
    'HostRunResult',
]


class HostRunResult(ClosedModel):
    """Outcome of running one command on one host."""

    host: str
    """The selector atom this host was resolved from — lowercased alias (e.g. ``"m2"``) or literal ``ip:port``."""

    exit_code: int
    """Exit code from the remote command; ``-1`` when the command never ran (connection/auth/protocol failure)."""

    duration_s: float
    """Wall-clock time for the connect+auth+execute round-trip."""

    stdout: str
    stderr: str
    error: str | None
    """``"{ExceptionType}: {message}"`` when ``execute_at`` raised; ``None`` even on non-zero exit."""


class DispatchResult(ClosedModel):
    """Aggregate outcome of a single ``run_target`` invocation across one or more hosts."""

    results: Sequence[HostRunResult]
    overall_exit_code: int
    """``0`` iff every host reports ``exit_code == 0`` and ``error is None``."""

    total_duration_s: float


class DispatchService:
    """Resolve a target selector and execute the command across every resolved host."""

    def __init__(
        self,
        *,
        client_config: ClientConfig | None = None,
        hosts_resolver: Callable[[], Awaitable[HostsCache]] | None = None,
    ) -> None:
        self._client_config = client_config if client_config is not None else ClientConfig.load()
        self._hosts_resolver = hosts_resolver if hosts_resolver is not None else browse_and_cache

    async def run_target(
        self,
        target: str,
        command: str,
        *,
        session_id: str,
        agent_id: str | None,
        timeout: float | None,
    ) -> DispatchResult:
        """Parse the selector, fan out across hosts in parallel, return aggregated results."""
        cache = await self._hosts_resolver()
        try:
            atoms = parse_selector(
                target,
                groups=self._client_config.groups,
                discovered_aliases=frozenset(entry.alias.lower() for entry in cache.all_hosts()),
            )
        except UnknownAtomError:
            # Cache may be fresh-by-TTL but stale-by-content — a daemon may have
            # come up since the last browse. Retry once with a fresh browse.
            cache = await browse_fresh_and_cache()
            try:
                atoms = parse_selector(
                    target,
                    groups=self._client_config.groups,
                    discovered_aliases=frozenset(entry.alias.lower() for entry in cache.all_hosts()),
                )
            except SelectorError as exc:
                raise RemoteBashError(str(exc)) from exc
        except SelectorError as exc:
            raise RemoteBashError(str(exc)) from exc

        resolved: list[tuple[str, tuple[Sequence[str], int]]] = []
        for atom in atoms:
            addr = lookup_alias(cache, atom)
            if addr is None:
                raise AssertionError(f'parser validated {atom!r} but cache lookup missed it')
            resolved.append((atom, addr))

        started = time.monotonic()
        host_results = await asyncio.gather(
            *(
                _run_one(
                    host,
                    ips,
                    port,
                    command,
                    session_id=session_id,
                    agent_id=agent_id,
                    timeout=timeout,
                )
                for host, (ips, port) in resolved
            )
        )
        total = time.monotonic() - started
        overall = 0 if all(r.exit_code == 0 and r.error is None for r in host_results) else 1
        return DispatchResult(
            results=host_results,
            overall_exit_code=overall,
            total_duration_s=total,
        )


async def _run_one(
    host: str,
    ips: Sequence[str],
    port: int,
    command: str,
    *,
    session_id: str,
    agent_id: str | None,
    timeout: float | None,
) -> HostRunResult:
    """Run on one host; capture wire-level failures into ``HostRunResult.error`` instead of aborting the batch."""
    started = time.monotonic()
    try:
        result = await execute_at(
            ips=ips,
            port=port,
            command=command,
            session_id=session_id,
            agent_id=agent_id,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001  # exception_safety_linter.py: swallowed-exception — one bad daemon must not abort the batch; per-host failure is captured into the summary table
        error_text = str(exc) if isinstance(exc, RemoteBashError) else f'{type(exc).__name__}: {exc}'
        return HostRunResult(
            host=host,
            exit_code=-1,
            duration_s=time.monotonic() - started,
            stdout='',
            stderr='',
            error=error_text,
        )
    return HostRunResult(
        host=host,
        exit_code=result.exit_code,
        duration_s=time.monotonic() - started,
        stdout=result.stdout,
        stderr=result.stderr,
        error=None,
    )
