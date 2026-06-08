"""preflight-check CLI — verify this machine is in a good state to build."""

from __future__ import annotations

__all__ = [
    'app',
    'main',
]

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from preflight_check.checks import ALL_CHECKS
from preflight_check.checks.base import Fixable
from preflight_check.report import render_report
from preflight_check.runner import run_checks

app = create_app(help='Verify this machine (and mesh) is in a good state to build.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


@app.command()
def check() -> None:
    """Run all readiness checks on the current machine and report findings."""
    report = run_checks(ALL_CHECKS)
    typer.echo(render_report(report))
    if report.overall == 'critical':
        raise typer.Exit(2)
    if report.overall == 'warning':
        raise typer.Exit(1)


@app.command()
def fix(check_id: str) -> None:
    """Apply a check's remediation by id (may require sudo)."""
    for check in ALL_CHECKS:
        if check.id == check_id:
            if not isinstance(check, Fixable):
                typer.echo(f'{check_id}: no fix available')
                raise typer.Exit(1)
            check.fix()
            typer.echo(f'{check_id}: fix applied')
            return
    typer.echo(f'unknown check: {check_id!r}')
    raise typer.Exit(1)


@error_boundary
def main() -> None:
    run_app(app)


if __name__ == '__main__':
    main()
