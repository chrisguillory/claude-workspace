# Follow-up

A concrete next step that surfaced while doing other work — a loose end you noticed and set down to
keep moving. The session that surfaced it (via the footer) is the thread back to the full context,
so the entry just needs the next step itself.

## What to capture

A pointer plus a headline, not a spec — the next step and where it lands. The session-provenance
footer carries how it came up, so don't re-litigate the lineage here; it's discoverable when the
entry is picked up.

```markdown
---
area: document-search
title: commit the search_timeout regression test
---

**Next step.** Land the regression test for the `search_timeout` lower-bound guard.

**Sketch.** Mirror the existing `document-search` test layout; assert `0` / `-5` reject, `None` accepts.
```

## Ask before filing

- What's the concrete next step?
- Which `area` owns it, and where does the work land?
