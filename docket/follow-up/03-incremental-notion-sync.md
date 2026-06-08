---
area: context
title: incremental Notion sync — re-export only the pages that changed
---

**Next step.** Diff each page's `last_edited_time` against the archived `index.json`, then page-level re-export (`enqueueTask` API — `recursive`, pollable via `getTasks`) of only the changed roots, folding the result back into `context/notion/{workspace}/` instead of re-running the full workspace export. `scripts/process-notion-export.py` already lays down the corpus + `index.json`; this adds the change-driven path on top.

**Waits on.** A second sync actually being needed — the one-time workspace export covers the corpus now — and real drift to diff against. Building the incremental machinery before then is guessing at a shape we haven't seen.

<sub>Claude Code session <code>ef5667b6-8b3f-4842-8584-f14f98bdf375</code></sub>
