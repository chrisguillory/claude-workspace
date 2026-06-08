---
name: deploy-main
description: "Deploy origin/main across the home mesh fleet — merge the latest trunk into every
  checkout on a target. Applies by default; clean checkouts merge (even atop unrelated uncommitted
  edits), while conflicts and the files git would refuse are reported untouched. --dry-run previews
  first. Use when rolling a merged change out to the fleet."
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

**The line:** the CLI owns all git/crb mechanics — enumeration, classification, the clean merge, the
typed report. This skill owns the judgment — run it, read the report, triage what didn't land. The
deploy is **not gated**: a dirty checkout merges only when its edits don't touch the incoming files,
and git refuses the rest (`blocked-*`), so your real work is never clobbered — only expendable
git-ignored files can be overwritten (see Phase 1). The default applies; `--dry-run` is there only if
you want to look before you leap.

If `deploy-main` isn't on PATH: `scripts/install-launcher.sh scripts/deploy-main.py` (or run
`scripts/deploy-main.py` directly).

## Phase 1: Deploy + report

Run the deploy — clean checkouts merge (even with unrelated uncommitted edits); anything git would
refuse is reported, untouched:

```bash
deploy-main <target> --format json
```

Parse the `DeployResult` and present the per-host outcome:

- `merged` (→ new short SHA) — main landed.
- `current` — already has main.
- `conflict` (+ files) — commit-level merge conflict; aborted, untouched.
- `blocked-local` (+ files) — uncommitted **tracked** edits overlap the incoming change; git refuses.
- `blocked-untracked` (+ files) — **untracked** files collide with incoming; git refuses (won't clobber them — *except git-ignored files, which it deems expendable and overwrites*).
- `blocked-inprogress` — the checkout is mid-merge/rebase/cherry-pick/revert; left strictly untouched (aborting would destroy that in-progress operation).
- `apply-aborted` — apply hit a blocker the dry-run couldn't foresee (a case-only filename collision on macOS's case-insensitive FS, or state that moved since the preview); the checkout is left untouched — re-run or inspect.

A host with a **non-zero `exit_code`** (or non-null `error`) hit a gate failure (`no-repo`, `fetch-failed`, or unreachable) and ran **nothing** — its `checkouts` come back empty. Surface it loudly; never count it as deployed.

Call out everything that isn't `merged`/`current` — those are what you weigh next. Re-running is safe
and converges.

**Want to look first?** `deploy-main <target> --dry-run --format json` classifies without mutating
(`would-merge` in place of `merged`). It's opt-in, not the default.

## Phase 2: Triage what didn't land

Decide with the user — never force-resolve a non-trivial case:

- **`blocked-local`** — stash or commit the overlapping edits on that host, then re-run. The edits
  are real work; the owner chooses how to preserve them.
- **`blocked-untracked`** — the colliding files are untracked (often regenerable). Move/remove them
  on that host, then re-run. Never auto-delete untracked work.
- **`blocked-inprogress`** — the checkout has an unfinished merge/rebase/cherry-pick/revert. Don't
  touch it from here; the owner finishes or aborts that operation on the host, then re-runs.
- **`conflict`** — trivial (lockfile, generated artifact, an unambiguous one-side change): resolve on
  that host with oversight, e.g.
  `claude-remote-bash execute -t <host> '<cd checkout && git merge origin/main && resolve && commit>'`.
  Non-trivial: leave it reported; the owner resolves with full context. (Future: an inbox
  "knock-knock — I hit a conflict, can you resolve?" ping.)

## Notes

- Source is always `origin/main`; only the **target** is a parameter (a `claude-remote-bash`
  selector — one host, a comma-list, or a group).
- The deterministic core is independently usable + scriptable: `deploy-main <target> [--dry-run]`
  from any terminal, no LLM in the loop.
- A dirty checkout is **not** skipped wholesale — it merges when its edits don't overlap the incoming
  change, and only blocks (`blocked-local` / `blocked-untracked`) on the files git itself would refuse.
