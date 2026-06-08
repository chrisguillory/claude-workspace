---
area: docket
title: daydream — AI-led idea generation from the full corpus
---

**The spark.** Flip ideation from human-led to model-led: Claude Code mines *all* corpuses — the
code, the session history, and the open web — and proposes candidate `idea` / `feature` /
`tech-debt` entries for the human to vet. The system starts drafting its own roadmap instead of
waiting to be asked. Moniker **daydream**: the unbiased fact-check cleared it — Claude Code's
"dream" / "Dreams" is *memory consolidation*, not work-generation, and "daydream" collides with
nothing in its docs or changelog.

**Why it's exciting.** It surfaces opportunities and problems you'd never have thought to look for.
Concrete daydream outputs, each grounded in a real corpus hit: a new embedding model that clears a
box you were stuck on; a recurring error mined from session + bash history (or a spot where Claude
itself keeps tripping); an emerging tool worth adopting (`ty`); unknown foot-guns or design-principle
drift; stray `# type: ignore`s; a pre-commit check we keep wishing we had.

**Shape & edge cases.** Two layers — the *local* corpus (code + sessions) and the *web* (industry
consensus, new tools/models) — reading whatever breadth the target type implies (per the
corpus-scale). Leans on existing infra: `deep-research` for the web layer, `document-search` for the
corpus. The real failure mode is **noise**: it needs a grounding gate (every proposal cites a real
corpus hit *and* a real external source), human curation, and a *periodic* cadence — not a
continuous firehose.

**Open questions.** How far does the web reach go — release notes, changelogs, papers? What cadence
(daily / weekly / on-demand)? And is "daydream" worth keeping, or does a non-`dream` root buy enough
distance from the memory feature to matter?

<sub>Claude Code session <code>be8ada35-c290-4d7d-88d3-df113a25f8b8</code></sub>
