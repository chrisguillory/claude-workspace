"""Shared subprocess driver for the linter test suites."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = [
    'run_linter',
]


def run_linter(test_file: Path, linter: Path) -> str:
    """Run the linter and return combined stdout+stderr."""
    result = subprocess.run(
        [sys.executable, str(linter), '--no-skip-file', '--no-config', str(test_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.stdout + result.stderr
