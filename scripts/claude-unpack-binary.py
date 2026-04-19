#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
#     "typer",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

"""Unpack the active Claude Code binary to extract the embedded JS bundle.

Performs the unpack on a COPY of the binary — never touches the installed
executable. ``npx tweakcc unpack`` rewrites its input in place; running it
against the installed binary bricks ``claude``. This wrapper makes that
physically impossible by copying first.

Usage:
    claude-unpack-binary                    Unpack to a fresh /tmp/claude-unpack-XXXXXX
    claude-unpack-binary -o /tmp/mydir      Unpack to a specific dir
    claude-unpack-binary --path /path/to    Unpack a specific binary

Typical follow-on analysis:
    strings <copy> > /tmp/strings.txt
    grep -n hideHelp <unpacked.js>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import typer
from cc_lib.error_boundary import ErrorBoundary

DEFAULT_SYMLINK = Path.home() / '.local' / 'bin' / 'claude'

error_boundary = ErrorBoundary(exit_code=1)


class UnpackError(Exception):
    """Raised when the unpack workflow cannot complete."""


@error_boundary
def main(
    path: Path | None = typer.Option(
        None,
        '--path',
        help='Path to a specific claude binary (default: resolve from ~/.local/bin/claude symlink)',
    ),
    output_dir: Path | None = typer.Option(
        None,
        '-o',
        '--output-dir',
        help='Directory to place the copy and unpacked JS (default: a fresh /tmp/claude-unpack-XXXXXX)',
    ),
) -> None:
    """Copy the active claude binary to a workspace and unpack it safely."""
    binary = _resolve_binary(path)
    version = binary.name  # e.g., ``2.1.114`` (the installed path is .../versions/<version>)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='claude-unpack-'))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    copy_path = output_dir / f'claude-{version}'
    if copy_path.exists():
        raise UnpackError(f'Refusing to overwrite existing file: {copy_path}')
    shutil.copy2(binary, copy_path)
    print(f'Copied installed binary -> {copy_path}', file=sys.stderr)
    print('Running `npx tweakcc unpack` on the COPY (never the installed binary)...', file=sys.stderr)

    subprocess.run(
        ['npx', 'tweakcc@latest', 'unpack', str(copy_path)],
        cwd=output_dir,
        check=True,
    )

    # tweakcc overwrites the input file in place with the unpacked JS —
    # so ``copy_path`` now contains JS, not Mach-O. That's exactly why
    # we copied first: the rewrite is confined to our temp directory.
    print('', file=sys.stderr)
    print(f'Unpacked JS: {copy_path}', file=sys.stderr)
    print(f'  size: {copy_path.stat().st_size / 1_048_576:.1f} MB', file=sys.stderr)
    print(file=sys.stderr)
    print('Suggested next steps:', file=sys.stderr)
    print(f'  grep -n hideHelp {copy_path}', file=sys.stderr)
    print(f'  strings {copy_path} | less', file=sys.stderr)

    # stdout gets just the unpacked file path so the wrapper can be piped:
    #   claude-unpack-binary | xargs grep -n hideHelp
    print(copy_path)


def _resolve_binary(explicit: Path | None) -> Path:
    if explicit is not None:
        resolved = explicit.resolve()
        if not resolved.exists():
            raise UnpackError(f'Binary not found at {resolved}')
        return resolved
    if not DEFAULT_SYMLINK.exists():
        raise UnpackError(f'Claude symlink not found at {DEFAULT_SYMLINK}')
    return DEFAULT_SYMLINK.resolve()


@error_boundary.handler(UnpackError)
def _handle_unpack_error(exc: UnpackError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(subprocess.CalledProcessError)
def _handle_tweakcc_failure(exc: subprocess.CalledProcessError) -> None:
    print(f'tweakcc exited with status {exc.returncode}', file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_crash(exc: Exception) -> None:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    for frame in traceback.format_tb(exc.__traceback__)[-2:]:
        print(frame.rstrip(), file=sys.stderr)


if __name__ == '__main__':
    typer.run(main)
