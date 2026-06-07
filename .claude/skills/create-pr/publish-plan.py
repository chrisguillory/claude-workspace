#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
#     "diskcache>=5.6",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../../../cc-lib/", editable = true }
# ///
"""Publish a session plan as a secret gist for the create-pr skill — idempotently.

First run creates the gist (GitHub mints the id) and associates the plan's slug
with that id in a local store; later runs reuse the id (``markdown-kit --gist-id``)
so the same gist is updated in place instead of duplicated. Prints the gisthost
viewer URL for the PR body.

Usage: publish-plan.py <plan.md>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import diskcache
from cc_lib.error_boundary import ErrorBoundary

MARKDOWN_KIT = Path(__file__).parents[3] / 'tools' / 'markdown-kit' / 'markdown-kit.js'
STORE = Path.home() / '.cache' / 'create-pr' / 'plan-gists'
VIEW_URL = re.compile(r'View:\s*(https://gisthost\.github\.io/\?([0-9a-f]+)/\S+)')


@ErrorBoundary(exit_code=1)
def main() -> None:
    plan = Path(sys.argv[1]).expanduser().resolve()
    if not plan.is_file():
        raise FileNotFoundError(f'plan not found: {plan}')

    slug = plan.stem
    with diskcache.Cache(str(STORE)) as store:
        gist_id: str | None = store.get(slug)
        view_url, minted_id = _publish(plan, gist_id)
        if gist_id is None:
            store[slug] = minted_id  # associate the freshly-minted id with this plan
    print(view_url)


def _publish(plan: Path, gist_id: str | None) -> tuple[str, str]:
    """Run markdown-kit (update if we have an id, else create). Returns (view_url, gist_id)."""
    cmd = ['node', str(MARKDOWN_KIT), str(plan), '--secret-gist', '--embed-images', '--no-show-filepath']
    if gist_id is not None:
        cmd += ['--gist-id', gist_id]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    match = VIEW_URL.search(out)
    if match is None:
        raise RuntimeError(f'could not parse gist URL from markdown-kit output:\n{out}')
    return match.group(1), match.group(2)


if __name__ == '__main__':
    main()
