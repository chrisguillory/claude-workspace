"""Aggregated result of a preflight run."""

from __future__ import annotations

__all__ = [
    'Report',
    'render_report',
]

from collections.abc import Mapping, Sequence

from cc_lib.schemas.base import ClosedModel

from preflight_check.checks.base import Finding, Severity

SEVERITY_RANK: Mapping[Severity, int] = {'ok': 0, 'warning': 1, 'critical': 2}
SEVERITY_GLYPH: Mapping[Severity, str] = {'ok': '✓', 'warning': '⚠', 'critical': '✗'}


class Report(ClosedModel):
    findings: Sequence[Finding]

    @property
    def overall(self) -> Severity:
        if not self.findings:
            return 'ok'
        return max((f.severity for f in self.findings), key=lambda s: SEVERITY_RANK[s])


def render_report(report: Report) -> str:
    """Render a human-readable report for the CLI."""
    if not report.findings:
        return '✓ preflight: no checks registered yet'

    lines = [f'preflight: {report.overall.upper()}']
    for finding in report.findings:
        lines.append(f'{SEVERITY_GLYPH[finding.severity]} {finding.check_id}: {finding.title}')
        if finding.severity != 'ok':
            lines.append(f'    {finding.detail}')
            if finding.remedy:
                lines.append(f'    fix: {finding.remedy}')
    return '\n'.join(lines)
