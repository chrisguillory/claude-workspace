---
name: create-pr
description: "Draft a PR description into scratch/, pause for your review, then create or update the PR via gh. Gathers git context, indexes the session for semantic search, and follows the repo's CLAUDE.md PR guidelines."
argument-hint: "[base branch, defaults to 'main']"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(.claude/skills/create-pr/gather-pr-context.py:*)"
---

# Create PR: Context → Draft → Review → Submit

## Gathered context

!`.claude/skills/create-pr/gather-pr-context.py $ARGUMENTS`

## Instructions

### Phase 1: Context enrichment

Before drafting, get the *why*, not just the *what*:

- **Session transcript** — use `mcp__document-search__search_documents` (collection `document-chunks`) to recover the architectural decisions, user corrections, and research findings behind this change. The gather step above indexed it.
- **Related PRs** — if this is a follow-up or stacked change.

The git context (commits, files changed, existing-PR mode) is in the gathered output above. Do **not** draft until you understand why the change was made.

### Phase 2: Scratch file

Drafts live in the repo-root `scratch/` (gitignored). Filename `pr-{descriptive-slug}.md` — match the topic, not a ticket. If a `pr-*.md` for this change already exists (listed above), reuse it.

### Phase 3: Draft the description

Follow the repo's PR guidelines (CLAUDE.md → *PR Description Guidelines*):

- **Context** — a one-line TLDR: what this is and why, graspable at a glance. Not a wall of prose.
- **Description** — what changed and the key architectural decisions; old vs new behavior. The diff shows the *how* — don't relitigate it.
- **Testing** — what was verified and how (automated + empirical).

**Shape — write for an engineer skimming it for the first time** (Chris's recurring feedback):

- **TLDR-first, scannable, never imposing** — the gist in a ~20-second skim, no scrolling.
- **`<details>` for depth** — collapse mechanism write-ups, evidence, and long lists; keep at most one high-impact item at top level.
- **Terse but COMPLETE** — don't drop substance, hide it (push the bulk into `<details>`).
- **What & why, not how. No journey residue** — no "Following review…", no dev chronology, no AI fluff; readable by someone who never saw this session.
- **Re-verify numerical claims** (test counts, perf metrics) against current code before finalizing.

**GitHub auto-link traps** — the body renders as GitHub-flavored markdown: a bare `#N` links to issue/PR N (not your list item) and `@name` pings that user. Use names or `item N`, never a bare `#N` / `@`.

**Session provenance (always)** — end the body with: `<sub>Claude Code session <code>SESSION_ID</code></sub>` (the gather step prints the session ID; or `claude-session info`). ID only — the machine is recoverable from the ID across the mesh.

Write the file to `scratch/`.

### Phase 4: Pause for your review

Tell the user: "Draft is ready at `scratch/{filename}.md` — review in your IDE."

**Proceed signal:** if the user rejects a Write to the scratch file and says "create the PR", they edited it in their IDE — use the file **as-is**, do **not** re-draft.

### Phase 5: Create or update

**Create** (no existing PR):

```bash
gh pr create --base <base> --head <branch> --title "..." --body-file scratch/{filename}.md
```

Use `--body-file` — never inline `--body "$(cat ...)"` (shell-escaping).

**Update** (existing PR detected above):

```bash
gh api -X PATCH repos/chrisguillory/claude-workspace/pulls/{number} -F body="$(cat scratch/{filename}.md)"
```

Use REST — `gh pr edit --body` hits the deprecated Projects Classic API.

### Phase 6: Post-creation

Report the PR URL. Merges here gate on the `merge-gatekeeper` umbrella check (ruleset-protected, linear history) — offer `gh pr merge --squash --auto` if the user wants it to land on green.

## Key rules

- **Never inline the PR body** — always `--body-file` (create) / `gh api -F body="$(cat ...)"` (update).
- **REST API for updates** — `gh api -X PATCH`, never `gh pr edit --body`.
- **Gather context first** — read the session (decisions, feedback) before drafting.
- **Scratch file is the collaboration point** — the user refines in their IDE; the model uses it as-is.
- **What shipped, not the journey** — no iteration history in the body.
- **TLDR-first, depth in `<details>`** — write for a ~20-second skim, never a wall.
- **Session provenance** — end every body with `<sub>Claude Code session <id></sub>` (ID only).

## Anti-patterns

- **Drafting before reading the session context.**
- **Re-writing the scratch file after the user edited it** — recognize the proceed signal.
- **`gh pr edit`** — deprecated Projects Classic API; use `gh api`.
- **Inlining the body in the command** — use `--body-file`.
- **Journey residue** — "Following review…", dev chronology, padding.
- **Wall-of-text body** — if the reader must scroll to get the gist, move depth into `<details>`.
- **Bare `#N` / `@name`** — `#N` links to issue N and `@name` pings; use descriptive names or `item N`.
