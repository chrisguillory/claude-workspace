---
name: deploy-main
description: "Deploy origin/main across the home mesh fleet — merge the latest trunk into every
  checkout on a target. Applies by default (clean merges land, conflicts abort untouched, dirty is
  skipped) and reports; --dry-run previews first. Use when rolling a merged change out to the fleet."
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
merge, conflict-abort, skip-dirty, the typed report. This skill owns the judgment — run it, read the
report, triage conflicts. The deploy is **not gated**: the protections (clean-only merge,
conflict-abort, dirty-skip) are the safety, so the default applies. `--dry-run` is there only if you
want to look before you leap.

If `deploy-main` isn't on PATH: `scripts/install-launcher.sh scripts/deploy-main.py` (or run
`scripts/deploy-main.py` directly).

## Phase 1: Deploy + report

Run the deploy — clean merges land, conflicts abort untouched, dirty/current are skipped:

```bash
deploy-main <target> --format json
```

Parse the `DeployResult` and present the per-host outcome: `merged` (→ new short SHA) ·
`conflict` (with the colliding files) · `skip-dirty` (uncommitted work — untouched) · `current`
(already has main). Call out conflicts and dirties — those are what you weigh next. Re-running is
safe and converges.

**Want to look first?** `deploy-main <target> --dry-run --format json` classifies without mutating
(`would-merge` in place of `merged`). It's opt-in, not the default — the protections are the safety.

## Phase 2: Triage conflicts

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
