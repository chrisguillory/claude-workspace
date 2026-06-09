---
area: document-search
title: index page frontmatter as queryable metadata, not just path
---

**The spark.** Beyond today's path-scoped recall, index each archived page's metadata — Notion id, title, source URL, and any structured attributes — as document-search fields, so a query can filter on *attributes* rather than only "files under this teamspace dir." The processor already emits `index.json` (id → title · path · `notion.so` url); that's the seed metadata waiting to be promoted into the index.

**Why it's exciting.** Path scoping inherits the export's folder shape — coarse, and blind to anything cross-cutting. Attribute indexing would let recall cut across the hierarchy (by id, by attribute) and join the on-disk archive back to its live source more precisely than a directory walk.

**Shape & edge cases.** document-search reads per-file frontmatter into indexed fields; search gains attribute filters alongside the existing path filter. Edge: most archived pages carry only id/title/url, so the payoff scales with how much richer per-page metadata actually exists — thin frontmatter, thin win.

**Open questions.** Is path scoping genuinely insufficient (currently YAGNI — the filesystem hierarchy covers the queries we've needed)? Which fields earn a place in the index? Does this generalize past Notion to the other `context/` sources (e.g. granola)?

<sub>Claude Code session <code>ef5667b6-8b3f-4842-8584-f14f98bdf375</code></sub>
