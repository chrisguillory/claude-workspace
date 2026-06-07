"""Check contract: a probe that detects one machine-readiness condition."""

from __future__ import annotations

__all__ = [
    'Check',
    'Finding',
    'Severity',
]

from typing import ClassVar, Literal, Protocol

from cc_lib.schemas.base import ClosedModel

type Severity = Literal['ok', 'warning', 'critical']


class Finding(ClosedModel):
    # Identity
    check_id: str

    # Status
    severity: Severity

    # Details
    title: str
    detail: str
    remedy: str | None  # standalone command/steps to fix; None when healthy or no known fix


class Check(Protocol):
    """One machine-readiness probe: expose id + summary, implement detect()."""

    id: ClassVar[str]
    summary: ClassVar[str]

    def detect(self) -> Finding:
        """Run the probe; return a Finding (severity 'ok' when healthy)."""
        ...
