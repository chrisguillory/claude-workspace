from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

from claude_remote_bash.diagnose.types import DiagnoseReport, Status, VectorResult

__all__ = [
    'format_report',
]


_RESET = '\033[0m'
_STATUS_COLORS: Mapping[Status, str] = {
    Status.OK: '\033[1;32m',  # bold green
    Status.WARN: '\033[1;33m',  # bold yellow
    Status.FAIL: '\033[1;31m',  # bold red
    Status.INFO: '\033[1;34m',  # bold blue
}


def format_report(report: DiagnoseReport) -> str:
    """Render a diagnose report as ANSI-colored, indented text."""
    use_color = _should_color()
    lines: list[str] = []
    for result in report.results:
        lines.extend(_format_result(result, use_color=use_color))
        lines.append('')
    lines.append(f'Overall: {_format_status(report.overall_status, use_color=use_color)}')
    return '\n'.join(lines)


def _format_result(result: VectorResult, *, use_color: bool) -> Sequence[str]:
    out: list[str] = [
        f'{_format_status(result.status, use_color=use_color)} {result.name} — {result.summary}',
    ]
    if result.detail:
        out.extend(f'    {line}' for line in result.detail.splitlines())
    if result.fix_suggestion:
        out.append(f'    Fix: {result.fix_suggestion}')
    return out


def _format_status(status: Status, *, use_color: bool) -> str:
    label = f'[{status.value.upper():<4}]'
    if not use_color:
        return label
    return f'{_STATUS_COLORS[status]}{label}{_RESET}'


def _should_color() -> bool:
    if os.environ.get('NO_COLOR'):
        return False
    return sys.stdout.isatty()
