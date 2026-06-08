# Tech-debt

A **known issue** already in the tree — a shortcut taken, a cleanup deferred, suboptimal-but-working
code. The *issue* is known and named; the *fix* need not be clear or well-understood yet. Deferred,
not forgotten.

## What to capture

A **pointer + headline, not a spec.** The full depth — reasoning, rejected alternatives, the
surrounding diff — stays recoverable from the indexed session transcript via the provenance footer.
Resist dumping the analysis here; the transcript holds it.

```markdown
---
area: selenium-browser
category: MAINTAINABILITY  # BUG | SECURITY | PERFORMANCE | MAINTAINABILITY | ARCHITECTURE
severity: MEDIUM           # CRITICAL | HIGH | MEDIUM | LOW
title: untyped dicts where a Pydantic model belongs
---

**Problem.** 1–2 lines: the effect, in problem terms — what's wrong or costly, not the mechanics.

**Area.** `selenium-browser` — `selenium_browser/mcp/main.py`

**Fix / sketch.** A line or two pointing at the shape — a direction. May be empty if the fix is
still unknown; the issue stands regardless.
```

`area` + the key files are load-bearing — they make `rg -l 'area: X' docket/` (or a semantic search
over the docket) complete when a future refactor touches that subsystem and folds in everything
local to it. The headline can be loose; the `area` can't.

## Ask before filing

- **What's the effect, in problem terms?** What's wrong or costly — not the code change.
- **Which subsystem and which files?** The `area` and the load-bearing paths; these drive
  area-collection later.
- **Category and severity?** From `BUG | SECURITY | PERFORMANCE | MAINTAINABILITY | ARCHITECTURE`
  and `CRITICAL | HIGH | MEDIUM | LOW`.
- **Is there a fix direction yet?** A sketch if one exists — and fine if not; the issue is known
  even when the fix isn't.
