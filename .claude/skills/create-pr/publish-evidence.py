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
"""Publish PR evidence images to a secret gist — idempotently.

PR bodies render images only from URLs — base64 data URIs are stripped, and the
web editor's drag-drop upload endpoint is not reachable with `gh` credentials —
so screenshots captured during a session need a host. A gist is a git repo:
pushing image files to one yields stable ``gist.githubusercontent.com/{user}/
{id}/raw/{file}`` URLs that render in PR bodies (Camo-proxied, auth-free).

First run creates the gist and associates the slug with its id in a local
store; later runs push into the same gist (additive; same-name files update in
place, and raw URLs always serve the latest revision). Prints one ready-to-embed
markdown image line per file.

Usage: publish-evidence.py <slug> <image>...
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import diskcache
from cc_lib.error_boundary import ErrorBoundary

STORE = Path.home() / '.cache' / 'create-pr' / 'evidence-gists'
# Route git auth through gh (clearing inherited helpers first) so clone/push
# work against gist.github.com without a global `gh auth setup-git`.
GIT_AUTH = ('-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential')


@ErrorBoundary(exit_code=1)
def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f'usage: {Path(sys.argv[0]).name} <slug> <image>...')
    slug = sys.argv[1]
    images = [Path(arg).expanduser().resolve(strict=True) for arg in sys.argv[2:]]
    names = [image.name for image in images]
    if len(set(names)) != len(names):
        raise ValueError(f'image basenames must be unique within the gist, got: {names}')

    with diskcache.Cache(STORE.as_posix()) as store:
        gist_id = store.get(slug) or _create_gist(slug)
        store[slug] = gist_id  # remember the id so re-runs reuse this gist
    _push(gist_id, images)

    login = _gh('api', 'user', '-q', '.login')
    for name in names:
        print(f'![{Path(name).stem}](https://gist.githubusercontent.com/{login}/{gist_id}/raw/{name})')


def _create_gist(slug: str) -> str:
    """Create the secret gist with a stub README (gists cannot be empty). Returns the id."""
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / 'README.md'
        readme.write_text(f'Evidence images for `{slug}`, referenced from the PR body.\n')
        url = _gh('gist', 'create', '--desc', f'PR evidence: {slug}', str(readme))
    return url.rstrip('/').rsplit('/', 1)[-1]


def _push(gist_id: str, images: Sequence[Path]) -> None:
    """Clone the gist, copy images in, commit + push only when content changed."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / 'gist'
        _git('clone', f'https://gist.github.com/{gist_id}.git', str(repo))
        for image in images:
            shutil.copy2(image, repo / image.name)
        if _git('status', '--porcelain', cwd=repo):
            _git('add', '--all', cwd=repo)
            _git('commit', '-m', 'Update evidence images', cwd=repo)
            _git('push', cwd=repo)


def _gh(*args: str) -> str:
    return subprocess.run(['gh', *args], capture_output=True, text=True, check=True).stdout.strip()


def _git(*args: str, cwd: Path | None = None) -> str:
    cmd = ['git', *GIT_AUTH, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()


if __name__ == '__main__':
    main()
