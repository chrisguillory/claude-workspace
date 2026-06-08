"""preflight-check — machine and mesh readiness checks for Claude Code sessions."""

from __future__ import annotations

__all__ = [
    'PROJECT',
]

from cc_lib.project import Project

PROJECT = Project.from_pyproject(__file__)
