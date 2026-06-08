"""Check registry: every readiness check the runner dispatches over."""

from __future__ import annotations

__all__ = [
    'ALL_CHECKS',
]

from collections.abc import Sequence

from preflight_check.checks.base import Check
from preflight_check.checks.dns_resolver_wedge import DnsResolverWedge

ALL_CHECKS: Sequence[Check] = (DnsResolverWedge(),)
