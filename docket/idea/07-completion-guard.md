---
area: docket
title: completion-guard — chase unmet "After merge (required)" must-dos
---

**The spark.** A sweep that scans merged PRs (and ending sessions) for unchecked
`## After merge (required)` items — the must-dos create-pr now declares in a parseable shape — and
surfaces or chases the unmet ones. A session ending or a PR merging stops being able to orphan a
required backfill, migration, or deploy.

**Why it's exciting.** Must-dos are footguns precisely when they're forgotten — the work merges, the
session ends, and a required follow-through silently never happens, leaving the system half-done. The
completion-guard is the *enforcement* half of create-pr's *declaration*: create-pr writes the
checklist parseably, the guard makes sure it actually gets ticked.

**Shape & edge cases.** Parse the fixed `## After merge (required)` heading + checkboxes across merged
PRs, find the unchecked ones, and surface them — a periodic sweep, a session-end hook, or both; kin
to the close-session auditor. The failure mode is the *false unmet*: the action was done but the box
never ticked — so it surfaces for human confirmation rather than nagging blindly or auto-acting.

**Open questions.** Where does it run — a sweep over merged PRs, a session-end hook, a cron? How do
you tell "done but unchecked" from "genuinely undone"? And how does it chase — a notification, a
re-opened item, an auto-filed follow-up?

<sub>Claude Code session <code>be8ada35-c290-4d7d-88d3-df113a25f8b8</code></sub>
