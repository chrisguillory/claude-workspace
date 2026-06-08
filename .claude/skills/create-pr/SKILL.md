---
name: create-pr
description: "Draft a PR description into scratch/, pause for your review, then create or update the PR via gh. Gathers git context, indexes the session for semantic search, and follows the repo's CLAUDE.md PR guidelines."
argument-hint: "[base branch, defaults to 'main']"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(.claude/skills/create-pr/gather-pr-context.py:*)"
  - "Bash(.claude/skills/create-pr/publish-plan.py:*)"
---

# Create PR: Context → Draft → Review → Submit

## Gathered context

!`.claude/skills/create-pr/gather-pr-context.py "$ARGUMENTS"`

## Instructions

> [!IMPORTANT]
> **Run every phase — do not skip one because you think you can shortcut it.** Two skips are
> specifically forbidden:
> - **Phase 1 enrichment is mandatory even if you lived this session.** Your memory rounds off the
>   user's exact corrections; the transcript doesn't. Search it.
> - **Never drop a step over a judgment call** ("the plan has PII," "this arg looks fine"). Handle the
>   concern *inside* the step — Phase 1: search anyway; Plan: have the user scrub. Skipping is the
>   user's decision, not yours.

### Phase 1: Context enrichment

Before drafting, get the *why*, not just the *what*:

- **Session transcript** — use `mcp__document-search__search_documents` (collection `document-chunks`) to recover the architectural decisions, user corrections, and research findings behind this change. The gather step above indexed it.
- **Related PRs** — if this is a follow-up or stacked change.

The git context (commits, files changed, existing-PR mode) is in the gathered output above. Do **not** draft until you understand why the change was made.

### Phase 2: Scratch file

Drafts live in the repo-root `scratch/` (gitignored). Filename `pr-{descriptive-slug}.md` — match the topic, not a ticket. If a `pr-*.md` for this change already exists (listed above), reuse it.

### Phase 3: Draft the description

Follows the repo's PR guidelines (CLAUDE.md → *PR Description Guidelines*).

**The description is the *human* layer.** It carries what CI can't: the *why*, the
decisions, and the empirical verification no check runs. Never relitigate what CI already
proves (tests, lint, types, pins) — that's enforced on every change, and prose restating it
only goes stale. If a claim you're about to write *could* be a CI check, the ideal is to
push it there, not assert it once here.

**Title** — `type(scope): effect`, imperative, ≤50 chars. Infer the conventional-commit
prefix (`feat`/`fix`/`refactor`/`docs`/`perf`/`ci`/`test`; `!` for breaking) and scope from
the diff; name the *effect*, not the mechanics ("add search to the dashboard," not "add
`SearchController`"). It becomes the squash-merge subject and the searchable archive entry;
if it won't fit one line, the PR may be doing too much — say so.

**Body** — a lean skeleton; **drop any section that doesn't apply** (a one-line fix doesn't
get four headings):

- **Context / TLDR** — one line: what this is and why, graspable at a glance. Build it by
  *abstracting to intent* — never paraphrase the diff back at higher word count.
- **Why** — the problem, key decisions, alternatives rejected; link issues (`Fixes #N`).
  Drop if the TLDR already carries it.
- **Changes** — 3–6 behavior-level bullets; omit the obvious. The diff shows the *how*.
- **Test plan** — the **empirical residue**: what *you* verified that CI can't — "ran the
  real system, observed X," repro steps, edge cases, and **untested conditions + their
  risk**. Not "CI's green" — CI owns that.

**Shape — write for an engineer skimming it for the first time:**

- **TLDR-first, scannable, never imposing** — the gist in a ~20-second skim, no scrolling.
- **`<details>` for depth** — collapse mechanism write-ups, evidence, and long lists; keep at most one high-impact item at top level.
- **Reach for GitHub-Flavored Markdown when it beats prose** (all verified to render in a PR body) — a **table** for options / tradeoffs, an **alert** (`> [!WARNING]` / `> [!NOTE]`) for a caveat or breaking change, a **Mermaid** fenced block for architecture or flow, fenced **code** for commands + output, a **footnote** (`[^1]`) for an aside. Use only when it out-scans prose — a one-line fix needs none. **Mermaid is the only diagram engine GitHub renders**; Graphviz / D2 / PlantUML / Vega-Lite and `:::` containers / `==highlight==` fall back to raw text, so don't reach for them.
- **Terse but COMPLETE** — don't drop substance, hide it (push the bulk into `<details>`).
- **What & why, not how. No journey residue** — no "Following review…", no dev chronology, no AI fluff; readable by someone who never saw this session.
- **Re-verify numerical claims** (test counts, perf metrics) against current code before finalizing.

**GitHub auto-link safety** — the body renders as GitHub-flavored markdown:
- Bare `#N` links to issue/PR N and `@name` pings that user — and so do `GH-123` and bare
  40-char SHAs. Backtick any *generated* `#N` / `@handle` / SHA that isn't a deliberate
  reference (`` `#26` ``); use names or `item N` for list positions.
- `#N` already renders the issue's title on GitHub — don't also write the title yourself.
- Closing keywords (`Fixes #N`) auto-close **only against the default branch** and fire on
  loose wording — use exactly one per genuinely-closed issue, `Refs #N` for related ones.

**Claude Code Plan (when the session has one)** — the plan is Claude's launch point, not kept live after implementation, so treat it as *auxiliary origin context* parked at the **bottom**, never the headline. **First scan the plan for anything sensitive — secrets (tokens, keys) *and* personal/identifying details (private hostnames, home-network layout, personal paths, PII) — and sanitize it yourself** (genericize the specifics) — a gist is URL-reachable, so scrub it before it leaves the repo. Then publish the session's plan (a `*plan*.md` written this session, or a path the user names) — the **same** file, never a copy:

```bash
.claude/skills/create-pr/publish-plan.py <plan>
```

> [!IMPORTANT]
> **Publishing is automatic** — the model runs `publish-plan.py` itself (a
> `Bash(.claude/skills/create-pr/publish-plan.py:*)` allow-rule in `.claude/settings.json` clears the
> data-exfiltration gate). That makes the **sanitize step above the only gate**, so it is mandatory:
> genericize secrets + personal/network details before publishing — nothing else stands between the plan
> and the gist.

It creates the secret gist on first run and **reuses it on re-runs** (a local slug→gist-id store — no duplicate gists), then prints the gisthost viewer URL. Link it **just above the session-provenance trailer**, using that URL as the `href`:

```html
<a href="{viewUrl}">
  <img src="https://github.com/user-attachments/assets/fcc9f9e9-e066-462d-9894-1f0ac2eda6f2" alt="Claude" width="16"> Claude Code Plan
</a>
```

Skip the section entirely if there's no plan.

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

Report the PR URL. Merges here gate on CI (ruleset-protected, linear history) — offer `gh pr merge --squash --auto` if the user wants it to land on green.

**After the PR is open, never amend or force-push.** Land follow-ons — review responses,
fixes, adjustments — as **new dedicated commits** (`fix: …`, "address review", "add test")
so anyone watching can see *what changed since they last looked*; a force-push erases that
incremental view. The squash-merge collapses the trail into one clean commit on `main`, so
you keep reviewability during the PR *and* a tidy history after.

## Key rules

- **Never inline the PR body** — always `--body-file` (create) / `gh api -F body="$(cat ...)"` (update).
- **REST API for updates** — `gh api -X PATCH`, never `gh pr edit --body`.
- **Gather context first** — read the session (decisions, feedback) before drafting.
- **Scratch file is the collaboration point** — the user refines in their IDE; the model uses it as-is.
- **What shipped, not the journey** — no iteration history in the body.
- **Description ≠ CI layer** — carry the *why* + the empirical verification CI can't; never relitigate what CI proves (tests, lint, types).
- **Title** — `type(scope): effect`, imperative, ≤50 chars; name the effect, not the mechanics.
- **No force-push once open** — land follow-ons as new commits so the change-since-last-look stays visible; squash collapses them at merge.
- **TLDR-first, depth in `<details>`** — write for a ~20-second skim, never a wall.
- **Session provenance** — end every body with `<sub>Claude Code session <id></sub>` (ID only).
- **Claude Code Plan** — when the session has one, scrub it for secrets **and personal/PII details**, then publish via `.claude/skills/create-pr/publish-plan.py <plan>` (the *same* file; idempotent slug→id store, no duplicate gists) and link it at the **bottom**; auxiliary origin context.

## Anti-patterns

- **Drafting before reading the session context.**
- **Re-writing the scratch file after the user edited it** — recognize the proceed signal.
- **`gh pr edit`** — deprecated Projects Classic API; use `gh api`.
- **Inlining the body in the command** — use `--body-file`.
- **Journey residue** — "Following review…", dev chronology, padding.
- **Restating CI** — "tests pass," "lint clean," "types check" is the CI layer's job; the body's test plan is for the empirical verification CI can't run.
- **Amending / force-pushing an open PR** — destroys the reviewable "what changed since I last looked" view; add new commits instead.
- **Wall-of-text body** — if the reader must scroll to get the gist, move depth into `<details>`.
- **Bare `#N` / `@name`** — `#N` links to issue N and `@name` pings; use descriptive names or `item N`.
