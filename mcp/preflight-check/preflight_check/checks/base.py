"""Check contract: a probe that detects one machine-readiness condition."""

from __future__ import annotations

__all__ = [
    'Check',
    'Finding',
    'FixFailedError',
    'Fixable',
    'Severity',
]

from typing import ClassVar, Literal, Protocol, runtime_checkable

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


@runtime_checkable
class Fixable(Protocol):
    """A check that can remediate the condition it detects (may require sudo)."""

    def fix(self) -> None:
        """Apply the remediation and verify it took; raise FixFailedError if the condition persists."""
        ...


class FixFailedError(Exception):
    """fix() ran its remediation, but re-detection shows the condition persists."""
