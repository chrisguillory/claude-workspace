---
name: add-to-docket
description: "Add a deferred matter — tech-debt, feature, follow-up, or idea — to the repo's docket: a versioned NN-slug.md under docket/<type>/, reviewed in the normal PR flow and picked up later. Repo-native source-of-truth corollary to create-pr; resolve by deleting the file."
argument-hint: "[short description of the matter, optional — steer type/grouping at the review pause]"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(.claude/skills/add-to-docket/gather-context.py:*)"
  - "Write"
---

# Add to Docket: Surface → Classify → Draft → Write

## When to reach for this

A review / canvass / audit / implementation pass surfaced a matter that's **real but not for
now** — architectural debt, a cross-cutting cleanup, a feature you've decided on but won't
build yet, a follow-on from work just done, or a half-formed idea. Acting on it mid-flow
snowballs and derails the session. **Add it to the docket** so it's off this session's plate —
captured, versioned — and picked up later when there are cycles, or (the ideal) when a future
refactor touches that *area* and folds in everything local to it.

The docket is **repo-native and the source of truth**: each matter is a markdown file at
`docket/<type>/NN-slug.md`, committed via the normal PR flow and reviewed in the diff you
already use — no `scratch/` go-between, no separate tracker. The entry is a **pointer +
headline, not a spec**; the full depth (reasoning, rejected alternatives, surrounding diff)
stays recoverable from the indexed session transcript via the session ID in the footer.
Resist dumping the analysis into the entry; the transcript holds it.

**Resolve = delete the file.** Existence is the state — an open matter exists, a resolved one
is removed (the PR that addresses it deletes it; git history is the archive). No status field,
no "closed" pile.

## When NOT to add — just fix inline

Add only what's **costly or risky from a *human* perspective** — a big refactor, a
cross-cutting change, something needing a real decision or prioritization. The test is **"is
this costly/risky for a human?"** — not "is this outside the current diff?" Out-of-scope ≠
docket-worthy.

Do **not** docket cheap, low-risk, mechanical fixes an agent can just do now — **especially
anything in the stub layer or other repo-owned infrastructure we fully control, with no
production risk.** Fix those inline. Docketing a near-zero-cost fix is pure overhead: a file
to triage and pick up later, for work that was cheaper to just do.

## The four buckets — what sorts a matter (the classifier)

**Type is the directory.** Every matter lands in exactly one of
`docket/{tech-debt,feature,follow-up,idea}/`. The axis that sorts them is **how much of the
human's judgment the matter still needs before it's actionable**:

| Type          | The test                                                                      | Needs from the human        |
|---------------|-------------------------------------------------------------------------------|-----------------------------|
| **tech-debt** | A known code-quality fix in *existing* code — *what* & *why* already clear?   | a slot (scheduling)         |
| **follow-up** | The *tail* of something just done — a loose-end bound to a recent change?     | a slot                      |
| **feature**   | A *decided* net-new capability — the what is settled, it needs building?      | verification (it's net-new) |
| **idea**      | A *cool maybe* still needing the human's cycles to judge if it's worth doing? | judgment / vetting *first*  |

`idea → feature` is a promotion — vetting graduates it; since type = directory, that's a
`git mv docket/idea/… → docket/feature/…`. `tech-debt` and `follow-up` are clear work awaiting
a slot. The subtle line is `idea` vs `feature`: ask **"has the human decided to do it?"** —
decided → `feature`, still weighing → `idea`. **Surface your classification at the review
pause; the human can re-bucket.**

## Gathered context

!`.claude/skills/add-to-docket/gather-context.py`

## Instructions

### Phase 1: Recover the matter

You usually arrive holding the matter (you just surfaced it). Pin down:

- **The effect** — what's wrong, missing, or wanted, in problem/intent terms, not mechanics.
- **Area + key files** — *this is what makes area-collection work later.* Which subsystem
  (`document-search`, `claude-session`, `linters`, `cc-lib`, `hooks`, a skill, …) and the
  load-bearing file paths. Get these right; the headline can be loose, the area can't.
- **Category + severity** — for `tech-debt` (and where it fits): BUG / SECURITY / PERFORMANCE
  / MAINTAINABILITY / ARCHITECTURE × CRITICAL / HIGH / MEDIUM / LOW (same vocabulary as
  findings-workflow). Optional for `feature` / `idea` / `follow-up`.

If the matter came from a prior pass and the detail is fuzzy, recover it via
`mcp__document-search__search_documents` (collection `document-chunks`) — the gather step
indexed this session. Don't re-derive what the transcript already holds; the entry only needs
the headline.

### Phase 2: Classify + number

Pick the **type** (→ the directory) with the four-bucket classifier above. Then take the
**next `NN`** for that directory from the gather output — per-directory, 2-digit, sequential
(`01`, `02`, …). One matter per entry by default.

### Phase 3: Draft the entry — HEADLINE ONLY

Path: `docket/<type>/NN-slug.md` — slug the effect (`re-embeds-unchanged-chunks`), not a
ticket. **No `type` field** — the directory *is* the type.

```markdown
---
area: document-search
category: PERFORMANCE   # tech-debt mainly; omit where it doesn't fit
severity: MEDIUM        # tech-debt mainly; omit where it doesn't fit
title: index rebuild re-embeds unchanged chunks
---

**Problem.** 1–2 lines: the effect and why it earns a docket slot. Not a paragraph.

**Area.** `document-search` — `path/to/key_file.py`, `path/to/other.py`

**Fix / approach sketch.** One or two lines pointing at the shape — a direction, not a
spec. The how is recoverable from the session.

<sub>Claude Code session <code>SESSION_ID</code></sub>
```

`area` + `title` are required (the invariant test checks them); `category` / `severity` are
for `tech-debt` (omit where they don't fit). Drop a body line that doesn't apply. **Do not**
add acceptance criteria, effort estimates, alternatives tables, or mechanism write-ups — that's
spec depth, and it goes stale when the code moves. The transcript is the spec; the entry is the
pointer.

**Session provenance (always)** — end the body with `<sub>Claude Code session
<code>SESSION_ID</code></sub>` (the gather step prints the ID; or `claude-session info`). ID
only — a future session recovers the full reasoning from the indexed transcript by that ID.

**GitHub auto-link safety** — entries render as GitHub-flavored markdown in the PR: a bare
`#N` links to issue/PR N and `@name` pings; backtick any *generated* `#N` / `@handle` / SHA,
reserve bare for a deliberate cross-reference.

### Phase 4: Write it + pause for review

`Write` the entry directly to `docket/<type>/NN-slug.md`. **The Write diff is the review** —
there's no `scratch/` go-between; the file in the tree *is* the artifact, reviewed like any
change (and renderable via markdown-kit).

This is the **collaboration point** — the user confirms (or re-buckets the type, adjusts the
slug) and **steers grouping**:

- **One matter per entry is the default** — each independently prioritizable.
- **Group several into one entry** only when they share a root cause / area and a single
  future change would touch them together. The user decides; honor it.

The entry now sits in the working tree. It rides into a PR — **committing and merging is the
human-gated step, separate from this skill.** The agent never reasons about push-time
conflicts; the docket-invariant `tests` check guards per-directory `NN` uniqueness at PR time.

## Key rules

- **Headline, not spec** — frontmatter (area / category / severity / title) + 1–2-line problem
  + area/files + a one-line fix sketch. The transcript holds the depth.
- **Type is the directory; resolve is deletion** — pick the right bucket; no status field. A
  resolved matter is deleted (git history is the archive).
- **Area + key files are load-bearing** — they make `grep -rl 'area: X' docket/` complete at
  refactor time. Get them right even if the title is loose.
- **The file *is* the source of truth** — write directly to `docket/`, no scratch go-between;
  the Write diff is the review.
- **One matter per entry by default** — group only when the user asks (shared root cause /
  area).
- **Human/PR-gated** — the skill writes the entry; the human commits and merges. Don't
  `git commit` or open the PR from here unless asked.
- **Session provenance** — end every entry with `<sub>Claude Code session <id></sub>` (ID
  only).
- **Same vocabulary as findings-workflow** — BUG / SECURITY / PERFORMANCE / MAINTAINABILITY /
  ARCHITECTURE × CRITICAL / HIGH / MEDIUM / LOW.

## Anti-patterns

- **Dumping the full analysis into the entry** — acceptance criteria, effort estimates,
  alternatives tables, mechanism write-ups. That's spec depth; it goes stale when code moves.
  Write the pointer; the transcript is the spec.
- **Wrong bucket** — an `idea` filed as a `feature` implies a commitment never made; debt
  filed as an `idea` implies a decision already settled. Run the classifier; surface it at the
  pause.
- **Vague or missing area / files** — defeats area-collection; the matter can never be found by
  the refactor that should absorb it.
- **A scratch go-between** — there isn't one. Write the entry into `docket/` directly.
- **Committing or merging from the skill** — that's the human-gated PR step, operationally
  separate from local capture.
- **Bare `#N` / `@name`** in generated text — backtick generated ones; reserve bare for a
  deliberate cross-reference.
- **Docketing what you should just fix** — cheap, low-risk, mechanical fixes (especially
  stub-layer / repo-owned infra with no production risk) get done inline. The bar is human
  cost/risk, not out-of-scope-ness.
