#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../../../cc-lib/", editable = true }
# ///
"""Hoist a built quest-map from its ephemeral staging dir to the long-lived browsable store.

The build writes to a throwaway temp dir, so it never touches Anthropic's `~/.claude` session dir. This
promotes that output to `<main-repo>/claude-sessions/{id8}/where-am-i/` — diffing against any existing copy
first (a re-run surfaces what changed instead of silently overwriting) and keeping the prior copy as
`where-am-i.prev` for comparison. The promote is atomic: copy onto the store's filesystem, then rename.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from cc_lib import ErrorBoundary


@ErrorBoundary(exit_code=1)
def main() -> None:
    if len(sys.argv) != 3:
        sys.exit('usage: hoist-where-am-i.py <staging-dir> <session-id>')
    staging = Path(sys.argv[1])
    if not (staging / 'quest-map.md').exists():
        sys.exit(f'nothing to hoist: {staging / "quest-map.md"} not found (did the build run?)')
    store = _store_dir(sys.argv[2])

    if store.exists():
        _print_diff(store, staging)
    _promote(staging, store)
    shutil.rmtree(staging, ignore_errors=True)
    print(f'hoisted → {store}')


def _store_dir(session_id: str) -> Path:
    """`<main-repo>/claude-sessions/{id8}/where-am-i/` — the long-lived, gitignored, browsable home.

    Resolved via the shared git-common-dir so it points at the MAIN repo even from a worktree (the build
    runs in a worktree; the store persists across branches in the main checkout).
    """
    common = subprocess.run(
        ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    main_repo = Path(common).parent  # parent of the shared .git dir = the main repo root
    return main_repo / 'claude-sessions' / session_id[:8] / 'where-am-i'


def _print_diff(old: Path, new: Path) -> None:
    old_nodes = {p.name for p in (old / 'nodes').glob('*.md')}
    new_nodes = {p.name for p in (new / 'nodes').glob('*.md')}
    print(f'diff vs existing store ({old}):')
    print(f'  nodes: {len(old_nodes)} → {len(new_nodes)}')
    if added := sorted(new_nodes - old_nodes):
        print(f'  + {", ".join(added)}')
    if removed := sorted(old_nodes - new_nodes):
        print(f'  - {", ".join(removed)}')


def _promote(staging: Path, store: Path) -> None:
    prev = store.with_name('where-am-i.prev')
    incoming = store.with_name('.where-am-i.incoming')
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, incoming)  # copy onto the store's filesystem first
    if store.exists():  # keep the prior copy for comparison
        shutil.rmtree(prev, ignore_errors=True)
        store.rename(prev)
    incoming.rename(store)  # atomic swap into place


if __name__ == '__main__':
    main()
