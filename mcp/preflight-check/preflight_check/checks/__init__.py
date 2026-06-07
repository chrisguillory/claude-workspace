"""Check registry: every readiness check the runner dispatches over."""

from __future__ import annotations

__all__ = [
    'ALL_CHECKS',
]

from collections.abc import Sequence

from preflight_check.checks.base import Check

# Empty until the first check lands (dns_resolver_wedge).
ALL_CHECKS: Sequence[Check] = ()
