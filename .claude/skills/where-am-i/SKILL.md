---
name: where-am-i
description: Build a glanceable quest-map of a long session — pure user-intent roots, ground-truth check-marks, [from → to] ranges, PR/docket overlays, a metadata header. A shared recap-on-steroids built by a fresh unbiased agent; light at the top, full detail one cat away. On-demand orientation for when the arc of a long, compacted, multi-thread session is lost and you (and the user) need to decide what to tackle next.
argument-hint: "[session-id prefix — defaults to the current session]"
user-invocable: true
disable-model-invocation: true
allowed-tools:
  - "Bash(.claude/skills/where-am-i/gather-where-am-i.py:*)"
---

# Where Am I — orient a long session as a quest-map

A long session drifts: threads open, sub-quests get pushed and never popped, compactions blur the arc.
This builds a **shared map both you and the user read** to decide what's next — *what the user drove,
and how far each thread got*. Intent is the tree; commits and PRs are delivery laid over it. Light at
the top; **full detail is one `cat` away** (progressive disclosure — completed threads aren't reloaded,
so the session reaches its context ceiling slower).

## Gathered (deterministic)

!`.claude/skills/where-am-i/gather-where-am-i.py $ARGUMENTS`

The gather wrote three inputs (paths above): **`user-intent-spine.txt`** (the user's verbatim messages —
the load-bearing record of intent, which a model summary rounds off and the native recap loses across
compactions), **`gh-ground-truth.txt`** (PRs merged in the session window + open, recent commits,
worktrees), and **`session-metadata.txt`** (compactions, subagents, most-used skills).

## Build it — run the build workflow

**Do not build the map from your own working memory** — that drift is exactly what this skill exists to
escape. The build is a committed **Workflow** (`build-map.js` here) so its control flow runs the same way
every time and resumes cleanly if an agent dies mid-run. Invoke it with the gather's output dir:

```
Workflow({ scriptPath: '.claude/skills/where-am-i/build-where-am-i.js', args: { gatherDir: '{dir}' } })
```

`{dir}` is the `OUTPUT DIR` the gather printed — it holds `user-intent-spine.txt` / `gh-ground-truth.txt`
/ `session-metadata.txt` and receives `quest-map.md` + `nodes/`. The workflow runs **roots → nodes → top**,
deliberately in that order so the map is a roll-up of ground-truthed nodes, never a frozen first guess:

1. **Roots** — one agent reads the spine and returns the user-driven thread list. Pure intent, no
   ground-truthing yet; it's the only thing that must exist before the fan-out.
2. **Nodes** (parallel) — one agent per root authoritatively ground-truths it (spine + truth + `git log`
   directly, *not* a broad guess), writes `nodes/{slug}.md`, and returns its facts (landed?, PRs, span).
3. **Quest-map** — assembled *last* from those node facts + `session-metadata.txt`, then self-validated
   against `validate-quest-map.py` until `valid ✓`. The structural validator can't catch a wrong PR label, so
   the nodes are the semantic check — that's why they run first.

The heavy gathering happens in those throwaway build contexts, so the *main* session never loads it; later,
opening a node is a `cat`, not a recompute. If a run dies (an overload mid-fan-out), resume it — finished
agents return from cache, only the tail re-runs. The map and its nodes are machine-local artifacts — full
fidelity, never committed.

### Render spec (the agent follows this exactly)

**Node rule — PURE user-intent.** A node exists ONLY if the USER drove it: a goal they stated, or
something they explicitly reacted to ("let's do that", "great — build on it"). NOT system notifications,
NOT other-sessions merely discovered, NOT AI-incidental work or tooling — those are HOW, not goals.
Commits / PR-numbers / assistant-work are NEVER nodes; they only decide a node's check-mark. Multiple
roots for independent threads, numbered `[1]`..`[N]` — the handle for paging a node in later; sub-bullets
for the sub-quests under each.

**Header.** YAML frontmatter first (machine-readable: session, span, volume, skills, roots, provenance),
then a prose header line:
`WHERE AM I — session {id} "{title}" · [{from} → {to}] ~{span} · {n} msgs · {compactions} compactions · {subagents} subagents · most-used: {top skills}`
followed by one honest "the shape of it" sentence — how a person would narrate the arc to a colleague.
Frontmatter counts (`roots.total` / `landed` / `open`) must match the rendered tree exactly —
`roots.landed` equals the number of `✓`-marked roots, and `total = landed + open`. **`session.id` is
the full session UUID** (the gather prints it) — the artifact's traceable identity back to its
transcript, not the short prefix the prose header line may abbreviate to. The bundled `validate-quest-map.py`
enforces this structure and is the post-run conformance gate.

**Check-marks.** `✓` = landed (verified against truth); the **absence** of a mark = open. No other emojis.
A root is marked `✓` on its own line when its goal landed; it stays unmarked while a sub-quest is open.

**Date ranges — `[from → to]`** (square brackets, the definitive notation), computed from spine
timestamps (already Pacific — render as-is, never re-interpret as UTC). Per item
`[first-touched → last-touched]`; for an open/abandoned item the end = **when we lost it** (last
activity). For an open item touched **within the last week**, include the time: `[… → 6/8 3pm]` — a
finer how-recently-did-this-go-cold signal.

**PR / docket overlays — never structural.** Below the tree: a PR overlay where a PR spanning multiple
roots floats on its own line *above* the roots it served, and a PR within one root attaches to that
root; then a docket overlay mapping each entry seeded this session to the root it backs. Items never
bend to fit PR or docket boundaries.

**Provenance.** If a thread entered from elsewhere (a `git pull` of another session's work, a discovered
session the user then built on), note its origin (session, machine, still-alive?). It counts as a node
only if the user reacted to it.

**Footer.** End with `— open parent quests never popped back up to —`: the dangling frames, sorted
oldest-first by their start date.

**Conventions.** Arrows are `→` / `←`, never ASCII `->`. File paths use `{squiggly}` braces, never
`<angle>` (angle brackets are for CLI help text only). Plain ASCII tree, glanceable in seconds. Top
level only — the per-node detail lives in `nodes/{slug}.md`, never inlined here.

## Present

`cat` the written `quest-map.md` and show it verbatim. **It IS the artifact** — do not regenerate, summarize,
or paraphrase it; you and the user read the same file. When the user picks a thread to act on,
**`cat` its pre-gathered `nodes/{slug}.md`** — the detail was computed once at build time, so paging it
in is a file read, not a re-search, and the rest of the session stays out of context.
