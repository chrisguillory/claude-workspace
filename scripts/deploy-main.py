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

    current            -> already contains origin/main; nothing to do
    merged             -> clean merge applied (fast-forward when merely behind); would-merge under --dry-run
    conflict           -> commit-level merge conflict; abort, report the colliding files
    blocked-local      -> uncommitted *tracked* edits overlap the merge; report, leave untouched
    blocked-untracked  -> *untracked* files collide with incoming files; report, leave untouched
    blocked-inprogress -> checkout mid-merge/rebase/cherry-pick/revert; left strictly untouched
    apply-aborted      -> apply hit a blocker the dry-run couldn't foresee; aborted, left untouched

The merge IS the deploy: a new skill or lib fix on main lands where each machine is working. A
dirty checkout still merges when its edits don't touch the incoming files — deploy-main only steps
aside when git itself would refuse (the ``blocked-*`` cases), reporting exactly which files.
Conflicts and blocks are never forced — left for you (or the on-host AI) to triage. Classification
uses ``git merge-tree --write-tree`` (in-memory) plus a working-tree overlap check; the only mutation
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
from collections.abc import Sequence, Set
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
    return DeployResult(ref='origin/main', dry_run=dry_run, hosts=hosts, overall_exit_code=_compute_exit_code(hosts))


type CheckoutStatus = Literal[
    'merged',
    'would-merge',
    'conflict',
    'blocked-local',
    'blocked-untracked',
    'blocked-inprogress',
    'current',
    'apply-aborted',
]

# Outcomes that count as a successful deploy; any other status means the merge didn't land.
LANDED_STATUSES: Set[CheckoutStatus] = {'merged', 'would-merge', 'current'}


def _compute_exit_code(hosts: Sequence[HostResult]) -> int:
    """Process exit code: 0 only if crb ran on every host and every checkout landed.

    Non-zero when any host failed to run (transport/gate — a non-zero ``exit_code`` or an
    ``error``) or any checkout didn't land (anything outside ``merged``/``would-merge``/``current``),
    so a scripted ``deploy-main … && next`` can't read an incomplete deploy as success.
    """
    if any(host.exit_code or host.error for host in hosts):
        return 1
    landed = all(c.status in LANDED_STATUSES for host in hosts for c in host.checkouts)
    return 0 if landed else 1


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
        suffix = f'  [exit {host.exit_code}]' if host.exit_code != 0 else ''
        print(f'  {host.host} ({host.duration_s:.1f}s){suffix}')
        for c in host.checkouts:
            tail = f'  {c.detail}' if c.detail else ''
            print(f'    [{c.status}] {c.branch}{tail}  ({c.path})')


class CheckoutResult(ClosedModel):
    """One checkout's outcome on one host."""

    path: str
    branch: str
    status: CheckoutStatus
    detail: str  # colliding files (conflict, blocked-*) or the new short SHA (merged); '' otherwise


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
    """0 only if crb ran on every host and every checkout landed; non-zero otherwise."""


# Per-host pipeline, sent to each host over crb (a remote shell executor, so this layer is necessarily
# shell). Proven in the /tmp/dm-fix.* sandbox: enumerate checkouts (main working tree + linked
# worktrees), classify, merge the clean ones, abort/report the rest. A *dirty* checkout still merges
# when its edits don't overlap the incoming files; ``blocked-local`` (uncommitted tracked edits) and
# ``blocked-untracked`` (colliding untracked files) are the cases git itself would refuse, reported
# with the offending files. ``overlap`` intersects two newline lists via awk — no ``-v`` (it rejects
# embedded newlines) and no process substitution (fleet-shell portability). Emits one tab-delimited
# ``CHECKOUT<TAB>path<TAB>branch<TAB>status<TAB>detail`` line per checkout. DRY=1 = classify only.
PIPELINE = r"""
set -u
cd "$HOME/claude-workspace" 2>/dev/null || { printf 'ERROR\tno-repo\n'; exit 3; }
git fetch origin --quiet 2>/dev/null || { printf 'ERROR\tfetch-failed\n'; exit 4; }
REF=origin/main
overlap() {  # $1=set $2=candidates -> candidates that appear in set (non-empty lines)
  { printf '%s\n' "$1"; printf ':::DM-SEP:::\n'; printf '%s\n' "$2"; } | awk '
    $0==":::DM-SEP:::"{s=1;next}
    !s{if($0!="")a[$0]=1;next}
    ($0 in a)'
}
inprogress() {  # is checkout $1 mid-merge/rebase/cherry-pick/revert? (never touch it — aborting destroys it)
  git -C "$1" rev-parse --verify -q MERGE_HEAD >/dev/null 2>&1 && return 0
  git -C "$1" rev-parse --verify -q CHERRY_PICK_HEAD >/dev/null 2>&1 && return 0
  git -C "$1" rev-parse --verify -q REVERT_HEAD >/dev/null 2>&1 && return 0
  [ -d "$(git -C "$1" rev-parse --git-path rebase-merge 2>/dev/null)" ] && return 0
  [ -d "$(git -C "$1" rev-parse --git-path rebase-apply 2>/dev/null)" ] && return 0
  return 1
}
git worktree list --porcelain | sed -n 's/^worktree //p' | while IFS= read -r d; do
  br=$(git -C "$d" symbolic-ref --quiet --short HEAD 2>/dev/null || echo DETACHED)
  if git -C "$d" merge-base --is-ancestor "$REF" HEAD 2>/dev/null; then
    printf 'CHECKOUT\t%s\t%s\tcurrent\t\n' "$d" "$br"; continue
  fi
  if inprogress "$d"; then printf 'CHECKOUT\t%s\t%s\tblocked-inprogress\t\n' "$d" "$br"; continue; fi
  mt=$(git -C "$d" merge-tree --write-tree HEAD "$REF" 2>/dev/null) || {
    files=$(git -C "$d" merge-tree --write-tree --name-only HEAD "$REF" 2>/dev/null | sed -n '2,/^$/p' | grep -v '^$' | paste -sd, -)
    printf 'CHECKOUT\t%s\t%s\tconflict\t%s\n' "$d" "$br" "$files"; continue
  }
  # overlap against the merge-RESULT tree (what the merge actually writes), not the raw HEAD..REF
  # diff — the latter also lists files the checkout's own commits changed, which the merge won't touch.
  # --no-renames keeps a renamed file's *source* path in the set, so a dirty edit to it still blocks.
  inc=$(git -C "$d" diff --name-only --no-renames HEAD "$mt" 2>/dev/null)
  # local-dirty = staged + unstaged (what 'git merge' guards); diff HEAD alone misses index-only edits.
  dirty=$({ git -C "$d" diff --name-only HEAD; git -C "$d" diff --cached --name-only; } 2>/dev/null | sort -u)
  bl=$(overlap "$inc" "$dirty" | paste -sd, -)
  if [ -n "$bl" ]; then printf 'CHECKOUT\t%s\t%s\tblocked-local\t%s\n' "$d" "$br" "$bl"; continue; fi
  bu=$(overlap "$inc" "$(git -C "$d" ls-files --others --exclude-standard 2>/dev/null)" | paste -sd, -)
  if [ -n "$bu" ]; then printf 'CHECKOUT\t%s\t%s\tblocked-untracked\t%s\n' "$d" "$br" "$bu"; continue; fi
  if [ "$DRY" = "1" ]; then printf 'CHECKOUT\t%s\t%s\twould-merge\t\n' "$d" "$br"; continue; fi
  if git -C "$d" merge --no-edit "$REF" >/dev/null 2>&1; then
    printf 'CHECKOUT\t%s\t%s\tmerged\t%s\n' "$d" "$br" "$(git -C "$d" rev-parse --short HEAD)"
  else
    git -C "$d" merge --abort 2>/dev/null
    printf 'CHECKOUT\t%s\t%s\tapply-aborted\t\n' "$d" "$br"
  fi
done
"""


if __name__ == '__main__':
    main()
