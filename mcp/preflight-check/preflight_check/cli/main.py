"""preflight-check CLI — verify this machine is in a good state to build."""

from __future__ import annotations

__all__ = [
    'app',
    'main',
]

import subprocess

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from preflight_check.checks import ALL_CHECKS
from preflight_check.checks.base import Fixable, FixFailedError
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
    """Apply a check's remediation by id and verify it took (may require sudo).

    Headless (no tty): run the CLI itself under sudo — `sudo preflight-check fix <id>` — so
    nested sudo is root→root; a pre-warmed sudo timestamp does not reach child processes.
    """
    for check in ALL_CHECKS:
        if check.id == check_id:
            if not isinstance(check, Fixable):
                typer.echo(f'{check_id}: no fix available')
                raise typer.Exit(1)
            check.fix()
            typer.echo(f'{check_id}: fixed (verified by re-detection)')
            return
    typer.echo(f'unknown check: {check_id!r}')
    raise typer.Exit(1)


@error_boundary
def main() -> None:
    run_app(app)


@error_boundary.handler(FixFailedError)
def _handle_fix_failed(exc: FixFailedError) -> None:
    typer.echo(f'fix failed: {exc}', err=True)


@error_boundary.handler(subprocess.CalledProcessError)
def _handle_command_failed(exc: subprocess.CalledProcessError) -> None:
    cmd = exc.cmd if isinstance(exc.cmd, str) else ' '.join(exc.cmd)
    typer.echo(f'command failed (exit {exc.returncode}): {cmd}', err=True)


if __name__ == '__main__':
    main()
