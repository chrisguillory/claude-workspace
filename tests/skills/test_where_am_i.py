"""Golden-fixture + negative tests for the where-am-i map validator."""

from __future__ import annotations

import subprocess
from pathlib import Path

SKILL = Path(__file__).parents[2] / '.claude' / 'skills' / 'where-am-i'
VALIDATOR = SKILL / 'validate-map.py'
EXAMPLE = SKILL / 'example.md'


def test_committed_example_conforms() -> None:
    result = _validate(EXAMPLE)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_broken_map(tmp_path: Path) -> None:
    broken = tmp_path / 'broken.md'
    broken.write_text('not a quest-map\n')
    assert _validate(broken).returncode != 0


def _validate(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['uv', 'run', '--no-project', '--script', str(VALIDATOR), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
