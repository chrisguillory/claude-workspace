---
area: docket
title: session-miner — extract the parked-deferral backlog from past sessions
---

**The spark.** Semantically mine the mesh-wide *session* corpus for matters discussed at length but
never codified — the ideas, features, and debt left behind in long-lived sessions that fanned out
and never concluded — and surface them as candidate docket entries for the human to curate.

**Why it's exciting.** It drains a real graveyard. Until the docket existed, the only way to
"resolve" an idea was to *execute* it, so sessions stayed open as incubators, spawned more, and
piled up unconcluded — a backlog of parked intent scattered across transcripts. The session-miner
*extracts* that backlog instead of leaving it to rot or re-deriving it from scratch.

**Shape & edge cases.** Built on `document-search` over the session corpus; it *proposes*, the human
*curates* (the docket holds only aligned entries; the corpus holds everything). It's the
*extract-the-existing* counterpart to daydream's *generate-the-new* — same corpus, opposite intent.
The failure mode is re-surfacing things already resolved or already filed: dedup every candidate
against the current code and docket before proposing it.

**Open questions.** How do you detect "discussed but never codified" — the semantic gap between
session-talk and what actually landed in code/docket? How far back, and how wide across the mesh?
And does it run on a cadence, or on-demand when you go hunting?

<sub>Claude Code session <code>be8ada35-c290-4d7d-88d3-df113a25f8b8</code></sub>
