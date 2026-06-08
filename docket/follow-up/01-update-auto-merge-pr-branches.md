---
area: ci
title: auto-update the branch of auto-merge PRs that fall behind main
---

**Next step.** Add a GitHub Actions workflow (on push to the default branch) that, for each open PR
**with auto-merge enabled**, updates its branch against main via the *Update a pull request branch*
REST endpoint — with a PAT or GitHub App token, **not** the default `GITHUB_TOKEN` (which won't
re-trigger the required checks) — or adopt a bot (Kodiak / Mergify). **Scope: auto-merge PRs only.**
Those are the deterministic stuck state — the human already said "merge when ready," but the strict
"branch up to date" gate blocks it and nothing auto-syncs the branch (research this session: no
native setting — auto-merge only merges-when-ready; `allow_update_branch` only exposes the manual
button). So it stalls on a pointless human "Update branch" click — a mechanical, deterministic fix
the workflow can just do.

**Waits on.** Spare cycles — the manual click works, so this is convenience, not a blocker; deferred
until the auto-merge treadmill is worth the workflow + token setup.

<sub>Claude Code session <code>019e146a-eeb3-7743-b0f3-88e7e450674a</code></sub>
