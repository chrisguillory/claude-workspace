---
area: docket
title: corollary-enrichment — surface prior art when filing an entry
---

**The spark.** When you file an entry — especially an `idea` — `gather` semantically searches the
session corpus for *corollaries*: related prior discussion that bolsters, complicates, or opens a new
angle on what you're capturing. Not "you said this exact thing before," but the neighbouring threads
you'd want in front of you while the entry takes shape.

**Why it's exciting.** An idea is rarely fresh — its prior art is scattered across transcripts you've
forgotten. Surfacing it *at file time* feeds the playback loop with more to align against, connects
threads, and keeps you from re-deriving what you already worked out. The entry comes out sharper than
the moment alone would make it.

**Shape & edge cases.** Extends the `add-to-docket` gather step with a relevance-gated semantic pass
over `document-search`, scaled by type (an `idea` pulls the widest corpus, per the corpus-scale; a
`tech-debt` barely needs it). The failure mode is noise — weak "corollaries" that distract; only
strong neighbours earn a place, and the human decides what to fold in.

**Open questions.** What relevance bar keeps it useful, not noisy? Does the extra search slow gather
enough to matter? And is it idea-only, or worth a lighter version for every type?

<sub>Claude Code session <code>be8ada35-c290-4d7d-88d3-df113a25f8b8</code></sub>
