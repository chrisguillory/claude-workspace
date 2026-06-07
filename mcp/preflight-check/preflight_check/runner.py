"""Local runner: execute checks on the current host and collect a Report."""

from __future__ import annotations

__all__ = [
    'run_checks',
]

from collections.abc import Sequence

from preflight_check.checks.base import Check
from preflight_check.report import Report


def run_checks(checks: Sequence[Check]) -> Report:
    """Run each check on the current machine and aggregate findings."""
    return Report(findings=[check.detect() for check in checks])
