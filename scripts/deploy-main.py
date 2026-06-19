#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
#     "pydantic>=2.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

"""Deploy origin/main across the fleet — merge latest trunk into every checkout on a target.

``deploy-main <target>`` fetches origin and, for each checkout on the target hosts (the main
working tree's branch + every linked worktree), classifies and merges ``origin/main`` in:

    clean      -> merge it (a fast-forward when the checkout is already on main)
    conflict   -> abort, leave the checkout exactly as found, report the files
    dirty      -> skip (git refuses anyway)
    current    -> skip (already contains origin/main)

The merge IS the deploy: a new skill or lib fix on main lands where each machine is working.
Conflicts are never auto-resolved — they're reported for you (or the on-host AI) to triage.
Classification uses ``git merge-tree --write-tree`` (in-memory, no worktree); the only mutation
is a clean ``git merge`` (with ``--abort`` as a safety net). Per-host, idempotent — re-run converges.

Target is a claude-remote-bash selector: one host (M2), a comma-list (M2,M3), or a group
(mac-others, mac-mesh). Source is always origin/main.

Usage:
    deploy-main mac-others                 # deploy main to every other host
    deploy-main M2 --dry-run               # classify only, mutate nothing
    deploy-main mac-mesh --format json     # machine-readable result entities (for the skill)
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Annotated, Literal, cast

import typer
import typer.completion
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import ClosedModel

error_boundary = ErrorBoundary(exit_code=1)


@error_boundary
def main() -> None:
    """Entry point: enable tab completion, then run the single command."""
    typer.completion.completion_init()
    typer.run(deploy)


def deploy(
    target: Annotated[str, typer.Argument(help='crb selector: host (M2), comma-list, or group (mac-others)')],
    dry_run: Annotated[bool, typer.Option('--dry-run', help='Classify only; mutate nothing.')] = False,
    output_format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Merge origin/main into every checkout on the target hosts."""
    result = _run_fleet(target, dry_run=dry_run)
    if output_format == 'json':
        print(result.model_dump_json(by_alias=True, indent=2))
    else:
        _emit_text(result)
    raise typer.Exit(result.overall_exit_code)


def _run_fleet(target: str, *, dry_run: bool) -> DeployResult:
    """Dispatch the pipeline to the target over crb and aggregate the typed result."""
    script = f'DRY={"1" if dry_run else "0"}\n{PIPELINE}'
    proc = subprocess.run(
        ['claude-remote-bash', 'execute', '-f', 'json', '-t', target],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    dispatch = json.loads(proc.stdout)
    hosts = [
        HostResult(
            host=r['host'],
            exit_code=r['exit_code'],
            duration_s=r['duration_s'],
            checkouts=_parse_checkouts(r['stdout']),
            error=r['error'],
        )
        for r in dispatch['results']
    ]
    return DeployResult(
        ref='origin/main', dry_run=dry_run, hosts=hosts, overall_exit_code=dispatch['overall_exit_code']
    )


type CheckoutStatus = Literal['merged', 'would-merge', 'conflict', 'skip-dirty', 'current', 'apply-aborted']


def _parse_checkouts(stdout: str) -> Sequence[CheckoutResult]:
    """Parse the per-host pipeline's CHECKOUT lines into typed results."""
    results: list[CheckoutResult] = []
    for line in stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 5 or parts[0] != 'CHECKOUT':
            continue
        _, path, branch, status, detail = parts
        results.append(CheckoutResult(path=path, branch=branch, status=cast(CheckoutStatus, status), detail=detail))
    return results


def _emit_text(result: DeployResult) -> None:
    """Human-readable fleet table to stdout."""
    verb = 'would deploy' if result.dry_run else 'deployed'
    print(f'{verb} {result.ref} -> {len(result.hosts)} host(s)')
    for host in result.hosts:
        if host.error:
            print(f'  {host.host}: ERROR {host.error}')
            continue
        print(f'  {host.host} ({host.duration_s:.1f}s)')
        for c in host.checkouts:
            tail = f'  {c.detail}' if c.detail else ''
            print(f'    [{c.status}] {c.branch}{tail}  ({c.path})')


class CheckoutResult(ClosedModel):
    """One checkout's outcome on one host."""

    path: str
    branch: str
    status: CheckoutStatus
    detail: str  # conflicted files (conflict) or the new short SHA (merged); '' otherwise


class HostResult(ClosedModel):
    """One host's deploy outcome."""

    host: str
    exit_code: int
    duration_s: float
    checkouts: Sequence[CheckoutResult]
    error: str | None


class DeployResult(ClosedModel):
    """Fleet-wide deploy outcome — the result entity the skill parses from ``--format json``."""

    ref: str
    dry_run: bool
    hosts: Sequence[HostResult]
    overall_exit_code: int


# Per-host pipeline, sent to each host over crb (a remote shell executor, so this layer is necessarily
# shell). Proven in the /tmp/dm-* sandbox: enumerate checkouts (main working tree + linked worktrees),
# classify via merge-tree, merge the clean ones, abort conflicts, skip dirty/current. Emits one
# tab-delimited ``CHECKOUT<TAB>path<TAB>branch<TAB>status<TAB>detail`` line per checkout. DRY is
# injected by _run_fleet (1 = classify only, no mutation).
PIPELINE = r"""
set -u
cd "$HOME/claude-workspace" 2>/dev/null || { printf 'ERROR\tno-repo\n'; exit 3; }
git fetch origin --quiet 2>/dev/null || { printf 'ERROR\tfetch-failed\n'; exit 4; }
REF=origin/main
git worktree list --porcelain | awk '/^worktree /{print $2}' | while IFS= read -r d; do
  br=$(git -C "$d" symbolic-ref --quiet --short HEAD 2>/dev/null || echo DETACHED)
  if [ -n "$(git -C "$d" status --porcelain 2>/dev/null)" ]; then printf 'CHECKOUT\t%s\t%s\tskip-dirty\t\n' "$d" "$br"; continue; fi
  if git -C "$d" merge-base --is-ancestor "$REF" HEAD 2>/dev/null; then printf 'CHECKOUT\t%s\t%s\tcurrent\t\n' "$d" "$br"; continue; fi
  if git -C "$d" merge-tree --write-tree HEAD "$REF" >/dev/null 2>&1; then
    if [ "$DRY" = "1" ]; then printf 'CHECKOUT\t%s\t%s\twould-merge\t\n' "$d" "$br"; continue; fi
    if git -C "$d" merge --no-edit "$REF" >/dev/null 2>&1; then
      printf 'CHECKOUT\t%s\t%s\tmerged\t%s\n' "$d" "$br" "$(git -C "$d" rev-parse --short HEAD)"
    else
      git -C "$d" merge --abort 2>/dev/null
      printf 'CHECKOUT\t%s\t%s\tapply-aborted\t\n' "$d" "$br"
    fi
  else
    files=$(git -C "$d" merge-tree --write-tree --name-only HEAD "$REF" 2>/dev/null | sed -n '2,/^$/p' | grep -v '^$' | paste -sd, -)
    printf 'CHECKOUT\t%s\t%s\tconflict\t%s\n' "$d" "$br" "$files"
  fi
done
"""


if __name__ == '__main__':
    main()
