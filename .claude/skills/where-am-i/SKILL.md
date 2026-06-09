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

The gather wrote three inputs (paths above): the **spine** (the user's verbatim messages — the
load-bearing record of intent, which a model summary rounds off and the native recap loses across
compactions), the **truth** (merged/open PRs, recent commits, worktrees), and rough **meta**
(compactions, subagents, most-used skills).

## Build it — launch ONE unbiased agent

**Do not build the map from your own working memory** — that drift is exactly what this skill exists to
escape. Launch a single **`general-purpose` agent** (fresh, unbiased context) given the three gathered
file paths plus the render spec below. Instruct it to ground-truth every check-mark against `truth.txt`
(and `gh` / `git` where needed), **Write the finished map to the `top.md` path printed above**, and
return only a one-line confirmation — not the map itself (you'll `cat` it).

### Render spec (the agent follows this exactly)

**Node rule — PURE user-intent.** A node exists ONLY if the USER drove it: a goal they stated, or
something they explicitly reacted to ("let's do that", "great — build on it"). NOT system notifications,
NOT other-sessions merely discovered, NOT AI-incidental work or tooling — those are HOW, not goals.
Commits / PR-numbers / assistant-work are NEVER nodes; they only decide a node's check-mark. Multiple
roots for independent threads; sub-bullets for the sub-quests under each.

**Header.** YAML frontmatter first (machine-readable: session, span, volume, skills, roots, provenance),
then a prose header line:
`WHERE AM I — session {id} "{title}" · [{from} → {to}] ~{span} · {n} msgs · {compactions} compactions · {subagents} subagents · most-used: {top skills}`
followed by one honest "the shape of it" sentence — how a person would narrate the arc to a colleague.

**Check-marks.** `✓` = landed (verified against truth); the **absence** of a mark = open. No other emojis.

**Date ranges — `[from → to]`** (square brackets, the definitive notation), computed from spine
timestamps. Per item `[first-touched → last-touched]`; for an open/abandoned item the end = **when we
lost it** (last activity). For an open item touched **within the last week**, include the time:
`[… → 6/8 3pm]` — a finer how-recently-did-this-go-cold signal.

**PR / docket overlays — never structural.** Below the tree: a PR overlay where a PR spanning multiple
roots floats on its own line *above* the roots it served, and a PR within one root attaches to that
root; then a docket overlay mapping each entry seeded this session to the root it backs. Items never
bend to fit PR or docket boundaries.

**Provenance.** If a thread entered from elsewhere (a `git pull` of another session's work, a discovered
session the user then built on), note its origin (session, machine, still-alive?). It counts as a node
only if the user reacted to it.

**Footer.** End with `— open parent quests never popped back up to —`: the dangling frames, oldest first.

**Conventions.** Arrows are `→` / `←`, never ASCII `->`. File paths use `{squiggly}` braces, never
`<angle>` (angle brackets are for CLI help text only). Plain ASCII tree, glanceable in seconds. Top
level only — no inlined per-node detail.

## Present

`cat` the written `top.md` and show it verbatim. **It IS the artifact** — do not regenerate, summarize,
or paraphrase it; you and the user read the same file. When the user picks a thread to act on, page in
just that node — a targeted spine / transcript search for its slug — rather than reloading the whole
session.
