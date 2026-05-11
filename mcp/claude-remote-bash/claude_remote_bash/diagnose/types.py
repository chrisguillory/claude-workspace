from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    'DiagnoseReport',
    'Status',
    'VectorResult',
]


class Status(str, Enum):
    OK = 'ok'
    WARN = 'warn'
    FAIL = 'fail'
    INFO = 'info'


# High-to-low severity. DiagnoseReport.overall_status returns the first match.
_SEVERITY_ORDER: Sequence[Status] = [
    Status.FAIL,
    Status.WARN,
    Status.INFO,
    Status.OK,
]


@dataclass(frozen=True, kw_only=True)
class VectorResult:
    name: str
    status: Status
    summary: str
    detail: str
    fix_suggestion: str


@dataclass(frozen=True, kw_only=True)
class DiagnoseReport:
    results: Sequence[VectorResult]

    @property
    def overall_status(self) -> Status:
        statuses = {r.status for r in self.results}
        return next((s for s in _SEVERITY_ORDER if s in statuses), Status.OK)
