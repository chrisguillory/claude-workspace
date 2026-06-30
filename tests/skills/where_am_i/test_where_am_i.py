"""The where-am-i validator accepts a conformant map and rejects a broken one."""

from __future__ import annotations

import subprocess
from pathlib import Path

VALIDATOR = Path(__file__).parents[3] / '.claude' / 'skills' / 'where-am-i' / 'validate-quest-map.py'

CONFORMANT_MAP = """---
artifact: where-am-i
schema: 1
session:
  id: 00000000-0000-4000-8000-000000000000
  title: sample
  machine: M0
span:
  from: 2026-01-01
  to: 2026-01-02
  weeks: 1
volume:
  messages: 10
  compactions: 0
  subagents: 0
skills: {}
roots:
  total: 2
  landed: 1
  open: 1
---

WHERE AM I — session 00000000 "sample" · [1/1 → 1/2] ~1 wk · 10 msgs · 0 compactions · 0 subagents · most-used: none
The shape of it: a minimal two-root sample.

[1] ✓ a landed thread [1/1 → 1/1]
[2] an open thread [1/1 → 1/2]

— open parent quests never popped back up to —
  · the open thread
"""


def test_validator_accepts_a_conformant_map(tmp_path: Path) -> None:
    target = tmp_path / 'quest-map.md'
    target.write_text(CONFORMANT_MAP)
    result = _validate(target)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_a_broken_map(tmp_path: Path) -> None:
    target = tmp_path / 'quest-map.md'
    target.write_text('not a quest-map\n')
    assert _validate(target).returncode != 0


def test_validator_rejects_unterminated_frontmatter(tmp_path: Path) -> None:
    """An opening fence with no closing --- yields a clean issue, not a ValueError traceback."""
    target = tmp_path / 'quest-map.md'
    target.write_text('---\nartifact: where-am-i\nschema: 1\n')
    result = _validate(target)
    assert result.returncode != 0
    assert 'frontmatter not closed' in result.stdout, result.stdout + result.stderr


def test_validator_rejects_malformed_yaml(tmp_path: Path) -> None:
    """Malformed frontmatter YAML yields a clean issue, not a YAMLError traceback."""
    target = tmp_path / 'quest-map.md'
    target.write_text('---\n\tbad: [unclosed\n---\n\nbody\n')
    result = _validate(target)
    assert result.returncode != 0
    assert 'not valid YAML' in result.stdout, result.stdout + result.stderr


def _validate(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['uv', 'run', '--no-project', '--script', str(VALIDATOR), str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
