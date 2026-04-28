from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    'codesign_identifier',
    'python_version',
    'resolve_interpreter',
    'which_all',
]


def which_all(name: str) -> Sequence[str]:
    """All paths matching `name` on PATH, in resolution order."""
    if not shutil.which(name):
        return []
    result = subprocess.run(  # noqa: S603, S607 — `which` is a fixed system utility
        ['/usr/bin/which', '-a', name],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def resolve_interpreter(shim: Path) -> Path | None:
    """Read the shim's shebang; return the Python interpreter path."""
    first_line = shim.read_text(errors='replace').splitlines()[0]
    if not first_line.startswith('#!'):
        return None
    interp_str = first_line[2:].split()[0]
    return Path(interp_str) if 'python' in interp_str.lower() else None


def codesign_identifier(binary: Path) -> str:
    """Run `codesign -dv` and return the Identifier value (the LN grant key)."""
    result = subprocess.run(  # noqa: S603, S607 — `codesign` is a fixed system utility
        ['/usr/bin/codesign', '-dv', str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith('Identifier='):
            return line.split('=', 1)[1]
    return ''


def python_version(binary: Path) -> str:
    """Run `<binary> --version`."""
    result = subprocess.run(  # noqa: S603 — binary is the resolved interpreter
        [str(binary), '--version'],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or result.stderr).strip()
