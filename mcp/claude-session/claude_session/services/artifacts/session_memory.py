"""LEGACY — Claude Code session-memory artifact (removed upstream, ~2.1.128).

Session memory was an experimental, feature-flag-gated Claude Code subsystem
(`tengu_session_memory` / `tengu_sm_compact`) that maintained a per-session summary at
`<project>/<session-id>/session-memory/summary.md` — a 10-section markdown template
written by a background model call, which auto-compaction could substitute for older
messages. It never reached general availability (no CHANGELOG entry) and was removed
upstream: the compaction consumer dropped at 2.1.105, the producer + path between 2.1.126
(present) and 2.1.128 (absent), still gone at 2.1.160. It is NOT the surviving auto-memory
feature (`MEMORY.md`), which is unrelated.

Retained as LEGACY: it produces nothing for current Claude Code, but `summary.md` files
still exist on disk for older sessions, so clone/archive/restore/move keep round-tripping
them. Do not extend this module.

References:
- claude-workspace: recipe added in c4cc838e; README env docs in c9001d44; full mechanics
  writeup in docs/session-memory.md (added in 8f0dd916).
- Upstream anthropics/claude-code: issues #13688, #13779, #15097, #14227. The 2.1.128
  removal is a binary-audit finding — upstream never announced the add or the removal.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    'SESSION_MEMORY_DIRNAME',
    'SUMMARY_FILENAME',
    'collect_session_memory',
    'write_session_memory',
]


SUMMARY_FILENAME = 'summary.md'
SESSION_MEMORY_DIRNAME = 'session-memory'


def collect_session_memory(session_dir: Path, session_id: str) -> str | None:
    """Read session-memory/summary.md if it exists.

    Args:
        session_dir: The project-encoded directory (e.g., ~/.claude/projects/<enc>/)
        session_id: The session UUID

    Returns:
        The summary content, or None if no session memory exists.
    """
    path = session_dir / session_id / SESSION_MEMORY_DIRNAME / SUMMARY_FILENAME
    if path.exists():
        return path.read_text()
    return None


def write_session_memory(session_dir: Path, session_id: str, content: str) -> Path:
    """Write session-memory/summary.md to target directory.

    Args:
        session_dir: The target project-encoded directory
        session_id: The session UUID
        content: The summary markdown content

    Returns:
        Path to the written file.
    """
    path = session_dir / session_id / SESSION_MEMORY_DIRNAME / SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
