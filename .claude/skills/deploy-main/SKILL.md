---
name: deploy-main
description: "Deploy origin/main across the home mesh fleet — merge the latest trunk into every
  checkout on a target. Dry-runs and presents the plan first, applies on your go, reports conflicts
  for triage. Use when rolling a merged change out to the other Macs (M2-M5)."
argument-hint: "[crb target — e.g. mac-others | M2 | M2,M3 | mac-mesh]"
user-invocable: true
disable-model-invocation: false
allowed-tools:
  - "Bash(deploy-main:*)"
  - "Bash(claude-remote-bash:*)"
---

# Deploy main across the fleet

Merge the latest `origin/main` into every checkout (the main working tree's branch + every linked
worktree) on the target hosts, via the deterministic `deploy-main` CLI (`scripts/deploy-main.py`).

**The line:** the CLI owns all git/crb mechanics — enumeration, `merge-tree` classification, clean
merge, conflict-abort, skip-dirty, the typed report. This skill owns the judgment — present the plan,
gate the apply, triage conflicts.

If `deploy-main` isn't on PATH: `scripts/install-launcher.sh scripts/deploy-main.py` (or run
`scripts/deploy-main.py` directly).

## Phase 1: Dry-run + present

Classify without mutating, then show the fleet:

```bash
deploy-main <target> --dry-run --format json
```

Parse the `DeployResult` and present a per-host table. Status values:
`would-merge` (clean, would land) · `conflict` (with the colliding files) · `skip-dirty`
(uncommitted work — untouched) · `current` (already has main). Call out the conflicts and dirties
explicitly — those are what the user weighs.

## Phase 2: Apply (on the user's go)

Once the user confirms, run the deploy — applies clean merges, aborts conflicts, skips dirty:

```bash
deploy-main <target> --format json
```

Report what landed (`merged` → new short SHA) and what was left for triage (`conflict` /
`apply-aborted`). Re-running is safe and converges.

## Phase 3: Triage conflicts

For each `conflict`, decide with the user — never auto-resolve a non-trivial one:

- **Trivial** (lockfile, generated artifact, an unambiguous one-side change): resolve on that host
  with the user's oversight, e.g.
  `claude-remote-bash execute -t <host> '<cd checkout && git merge origin/main && resolve && commit>'`.
- **Non-trivial**: leave it reported; the owner resolves on that host with full context. (Future: an
  inbox "knock-knock — I hit a conflict, can you resolve?" ping.)

## Notes

- Source is always `origin/main`; only the **target** is a parameter (a `claude-remote-bash`
  selector — one host, a comma-list, or a group).
- The deterministic core is independently usable + scriptable: `deploy-main <target> [--dry-run]`
  from any terminal, no LLM in the loop.
- A host whose **main working tree** is dirty is skipped wholesale — that's the common case, and it's
  the safe one; the owner pulls main in when ready.
