"""macOS system introspection utilities."""

from __future__ import annotations

__all__ = [
    'detect_root_app',
]

import os
import subprocess


def detect_root_app() -> str:
    """Walk the process tree to find the root application that needs FDA.

    Returns the .app path (e.g., '/Applications/iTerm.app') or the
    root process name if no .app is found in the chain.

    Raises on failure — callers handle graceful degradation.
    """
    result = subprocess.run(
        ['ps', '-o', 'pid=,ppid=,comm=', '-ax'],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    procs: dict[int, tuple[int, str]] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            procs[int(parts[0])] = (int(parts[1]), parts[2])

    pid = os.getpid()
    chain: list[str] = []
    while pid in procs and pid != 1:
        ppid, comm = procs[pid]
        chain.append(comm)
        pid = ppid

    # Find the .app in the chain
    for entry in reversed(chain):
        if '.app/' in entry:
            return entry[: entry.index('.app/') + 4]

    if chain:
        return chain[-1]

    return 'unknown'
