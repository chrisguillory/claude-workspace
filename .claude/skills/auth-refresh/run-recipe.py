#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../../../cc-lib/", editable = true }
# ///
"""Resolve a recipe argument to a path and exec it.

Used by the ``auth-refresh`` skill's ``!`` directive to bridge between the
user's CLI input and a tool-specific auth recipe. Accepts either a full
path to a recipe or a tool-name shorthand:

  ./run-recipe.py ~/claude-workspace/scripts/gh-upload-auth-recipe.py
  ./run-recipe.py gh-upload                 # → scripts/gh-upload-auth-recipe.py
  ./run-recipe.py grok-kit                  # → mcp/grok-kit/auth-recipe.py

No-arg or unresolvable-arg invocations land in the AuthRefreshError handler
which prints usage and lists all discoverable recipes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary

WORKSPACE = Path.home() / 'claude-workspace'

boundary = ErrorBoundary(exit_code=1)


@boundary
def main() -> None:
    """Resolve the recipe argument and replace this process with it."""
    if len(sys.argv) < 2:
        raise AuthRefreshError('argument required')
    recipe = _resolve(sys.argv[1])
    os.execv(str(recipe), [str(recipe), *sys.argv[2:]])


# -- Helpers (private) -------------------------------------------------------


def _resolve(arg: str) -> Path:
    """Map a path-or-tool-name to a recipe path; raise on miss."""
    direct = Path(arg).expanduser()
    if direct.is_file():
        return direct
    by_script = WORKSPACE / 'scripts' / f'{arg}-auth-recipe.py'
    if by_script.is_file():
        return by_script
    by_mcp = WORKSPACE / 'mcp' / arg / 'auth-recipe.py'
    if by_mcp.is_file():
        return by_mcp
    raise AuthRefreshError(f'cannot resolve recipe: {arg}')


# -- Exceptions + error boundary handlers ------------------------------------


class AuthRefreshError(Exception):
    """Wrapper could not resolve a recipe argument."""


@boundary.handler(AuthRefreshError)
def _handle_refresh_error(exc: AuthRefreshError) -> None:
    """Print the failure reason, then usage and the discovered recipe list."""
    print(f'auth-refresh: {exc}', file=sys.stderr)
    print('', file=sys.stderr)
    print('Usage: /auth-refresh <recipe-path-or-tool-name> [recipe-args...]', file=sys.stderr)
    print('', file=sys.stderr)
    print('Available recipes:', file=sys.stderr)
    recipes = sorted(
        {
            *(WORKSPACE / 'scripts').glob('*-auth-recipe.py'),
            *(WORKSPACE / 'mcp').glob('*/auth-recipe.py'),
        }
    )
    if recipes:
        for r in recipes:
            print(f'  {r}', file=sys.stderr)
    else:
        print('  (none found)', file=sys.stderr)


@boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'auth-refresh: {type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
