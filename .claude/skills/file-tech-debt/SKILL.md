---
name: file-tech-debt
description: "File a surfaced tech-debt finding as a lightweight GitHub issue (pointer + headline), then pick it up later — ideally when a future refactor touches its area. Drafts into scratch/, pauses for your review, ensures tech-debt/area labels, creates via gh. The issue-tracker corollary to create-pr."
argument-hint: "[short finding description, optional — steer at the review pause instead]"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(.claude/skills/file-tech-debt/gather-debt-context.py:*)"
  - "Bash(gh label create:*)"
  - "Bash(gh label list:*)"
  - "Bash(gh issue create:*)"
  - "Bash(gh issue list:*)"
---

# File Tech Debt: Surface → Draft → Review → File

## When to reach for this

A review/canvass/audit pass surfaced a finding that's **real but not worth doing now** —
architectural debt, a cross-cutting cleanup, a perf nit off the critical path. Stopping to
fix it mid-flow snowballs and derails the session. **File it away** so it's off this
session's plate — captured, "checked off" — and picked up later when there are cycles, or
(the ideal) when a future refactor touches that *area* and folds in everything local to it.

The filed issue is a **lightweight pointer + headline, not a spec.** The full depth — the
reasoning, the rejected alternatives, the surrounding diff — stays recoverable from the
indexed session transcript via the session ID in the footer (same provenance model as
create-pr). Resist dumping the analysis into the issue; the transcript holds it.

This is the **sink for findings-workflow's "defer with ticket" verdict** — same category /
severity vocabulary (BUG, SECURITY, PERFORMANCE, MAINTAINABILITY, ARCHITECTURE × CRITICAL,
HIGH, MEDIUM, LOW), inverted target: an issue in the tracker, not a fix in the tree.

## Gathered context

!`.claude/skills/file-tech-debt/gather-debt-context.py`

## Instructions

### Phase 1: Recover the finding

You usually arrive here holding the finding (you just surfaced it). Pin down, per finding:

- **The effect** — what's wrong or costly, in problem terms, not mechanics.
- **Area + key files** — *this is what makes area-collection work later.* Which subsystem
  (`document-search`, `claude-session`, `linters`, `cc-lib`, hooks, a skill, …) and the
  load-bearing file paths. Get these right; the headline can be loose, the area can't.
- **Category + severity** — from the finding's existing classification if it has one.

If the finding came from a prior pass and the detail is fuzzy, recover it via
`mcp__document-search__search_documents` (collection `document-chunks`) — the gather step
indexed this session. Don't re-derive what the transcript already holds; the issue only
needs the headline.

### Phase 2: Scratch file

Drafts live in the repo-root `scratch/` (gitignored). One file per issue,
`issue-{descriptive-slug}.md` — slug the effect, not a ticket. Reuse an existing
`issue-*.md` (listed above) if you're refining one.

### Phase 3: Draft the issue — HEADLINE ONLY

**Title** — `[CATEGORY] effect`, the problem not the fix (`[PERF] index rebuild re-embeds
unchanged chunks`, not `[PERF] add content-hash skip`). Imperative-ish, ≤70 chars. It's the
searchable archive entry and the thing future-you skims in `gh issue list`.

**Body** — the minimum that lets future-you (or a refactor) decide to pick it up. Keep it to
roughly:

```markdown
**Problem.** 1–2 lines: the effect and why it's debt. Not a paragraph.

**Area.** `<subsystem>` — `path/to/key_file.py`, `path/to/other.py`
<!-- the load-bearing files; this is what `gh issue list --label area:X` collects -->

**Severity / Category.** MEDIUM / PERFORMANCE

**Fix sketch.** One or two lines pointing at the shape of the fix — a direction, not a
spec. "Content-hash chunks; skip re-embed when unchanged." The how is recoverable from
the session.

<sub>Claude Code session <code>SESSION_ID</code></sub>
```

Drop a line that doesn't apply — a one-file MAINTAINABILITY nit doesn't need a multi-path
Area block. **Do not** add acceptance criteria, effort estimates, an alternatives table, or
a mechanism write-up: that's spec depth, and it goes stale the moment the code moves. The
transcript is the spec; the issue is the pointer.

**Session provenance (always)** — end the body with `<sub>Claude Code session
<code>SESSION_ID</code></sub>` (the gather step prints the ID; or `claude-session info`). ID
only — a future session recovers the full reasoning from the indexed transcript by that ID.

**GitHub auto-link safety** — the body renders as GitHub-flavored markdown:
- Bare `#N` links to issue/PR N and `@name` pings that user — so do `GH-123` and bare
  40-char SHAs. Backtick any *generated* `#N` / `@handle` / SHA that isn't a deliberate
  reference (`` `#42` ``); use names or `item N` for list positions.
- A real cross-reference is fine unbacktick'd (`blocks #40`, `related to #12`) — use it when
  the finding genuinely depends on or duplicates another issue.

Write the file(s) to `scratch/`.

### Phase 4: Pause for your review

Tell the user: "Draft is ready at `scratch/issue-{slug}.md` — review in your IDE."

This is the **collaboration point** and where the user **steers grouping**:

- **One issue per finding is the default** — each is independently prioritizable.
- **Group several into one issue** when they share a root cause or area (e.g. three
  `document-search` indexing nits a single refactor would touch together). The user decides
  at the pause; honor it.

**Proceed signal:** if the user rejects a Write to a scratch file and says "file it," they
edited it in their IDE — use the file **as-is**, do **not** re-draft.

### Phase 5: Ensure labels (idempotent)

Labels are the spine of the refactor-time workflow, so they must exist before the issue
references them. The gather step listed which already exist. Ensure each label you're about
to use — **`--force` is idempotent** (creates if absent, updates color/description if
present):

```bash
gh label create tech-debt    --repo chrisguillory/claude-workspace --color BFD4F2 --description "Filed tech debt — pointer + headline; full context in the session transcript" --force
gh label create area:document-search --repo chrisguillory/claude-workspace --color C5DEF5 --description "Tech debt scoped to the document-search subsystem" --force
```

**Every issue gets `tech-debt` + exactly one `area:<subsystem>`.** Optionally add
`category:<lower>` (e.g. `category:performance`) if the user wants category-level collection
too. Area is the one that makes a refactor's `gh issue list --label area:X` complete — pick
the subsystem the *key files* live in (`area:document-search`, `area:claude-session`,
`area:linters`, `area:cc-lib`, `area:hooks`, `area:skills`, …). Stay consistent with area
labels already listed above; don't coin a synonym for one that exists.

### Phase 6: Create the issue(s)

```bash
gh issue create \
  --repo chrisguillory/claude-workspace \
  --title "[PERF] index rebuild re-embeds unchanged chunks" \
  --body-file scratch/issue-reembed-unchanged-chunks.md \
  --label tech-debt --label area:document-search
```

Use `--body-file` — never inline `--body "$(cat ...)"` (shell-escaping). One `gh issue
create` per scratch file. **Report every issue URL** it prints.

### Phase 7: Closing the loop later (document this for the user)

The payoff is at refactor time, not now. When a future session sets out to refactor an area:

```bash
gh issue list --repo chrisguillory/claude-workspace --label area:document-search --label tech-debt
```

collects every debt local to that area — fold them into the refactor's scope, then close
them as the refactor lands them (`gh issue close <N> --comment "addressed in #<PR>"`). Each
issue's session footer reopens the original reasoning if the headline isn't enough.

## Key rules

- **Headline, not spec** — title + 1–2-line problem + area/files + severity/category + a
  one-line fix *sketch*. The transcript holds the depth; the issue is the pointer.
- **Area + key files are load-bearing** — they make `gh issue list --label area:X` complete
  at refactor time. Get them right even if the title is loose.
- **Ensure labels idempotently before creating** — `gh label create … --force`. Every issue:
  `tech-debt` + one `area:<subsystem>`.
- **Scratch file is the collaboration point** — the user refines and steers grouping in
  their IDE; the model uses it as-is on the proceed signal.
- **One issue per finding by default** — group only when the user asks (shared root cause /
  area).
- **Never inline the issue body** — always `--body-file`.
- **Session provenance** — end every body with `<sub>Claude Code session <id></sub>` (ID
  only).
- **Same vocabulary as findings-workflow** — BUG / SECURITY / PERFORMANCE / MAINTAINABILITY /
  ARCHITECTURE × CRITICAL / HIGH / MEDIUM / LOW.

## Anti-patterns

- **Dumping the full analysis into the issue** — acceptance criteria, effort estimates,
  alternatives tables, mechanism write-ups. That's spec depth; it goes stale when code
  moves. File the pointer; the transcript is the spec.
- **Vague or missing area / files** — defeats area-collection; the issue can never be found
  by the refactor that should absorb it.
- **Inventing a new area label when a synonym already exists** — fragments the collection.
  Reuse the labels the gather step listed.
- **Filing before the user reviews** — the scratch pause is mandatory; grouping is the
  user's call.
- **Re-drafting the scratch file after the user edited it** — recognize the proceed signal.
- **Inlining the body** (`--body "$(cat …)"`) — shell-escaping bites; use `--body-file`.
- **Bare `#N` / `@name`** in generated text — `#N` links to issue N and `@name` pings;
  backtick generated ones, reserve bare for deliberate cross-references.
- **Filing what you should just fix** — if it's a one-line fix in flow, fix it. This is for
  debt that *would* derail the session.
