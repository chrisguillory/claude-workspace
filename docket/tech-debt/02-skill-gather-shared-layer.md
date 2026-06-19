---
area: skills
category: ARCHITECTURE      # BUG | SECURITY | PERFORMANCE | MAINTAINABILITY | ARCHITECTURE
severity: MEDIUM            # CRITICAL | HIGH | MEDIUM | LOW
title: session resolution duplicated across skill gathers — no shared consumer layer
---

**Problem.** Every skill gather that needs the current session reimplements the same `claude-session info --format json` resolution — a local `SessionInfo(SubsetModel)` plus the subprocess + parse. The obvious shared home, `cc_lib`, is a layering violation: cc_lib sits *below* `claude-session`, so modeling its CLI output there points the dependency up. There's no consumer-side layer the gathers can share, so the duplication persists and the next gather copies it again.

**Area.** `skills` — `.claude/skills/where-am-i/gather-where-am-i.py`, `.claude/skills/create-pr/gather-pr-context.py`, `.claude/skills/add-to-docket/gather-context.py` (each carries a local `SessionInfo` + `claude-session info` call; recover-session/close-session likely too).

**Fix / sketch.** A new *consumer-side* package (e.g. `skill-lib/`) depending on cc_lib and sitting *above* `claude-session`, holding the shared `SessionInfo` + a `resolve_session(arg, *, required)` helper that covers both the bubble (where-am-i, required) and graceful-degrade (create-pr/add-to-docket, optional) modes; migrate the gathers to import it. PEP 723 skill scripts can only share code through an installed package, so this is a new package — not a cc_lib module — and warrants its own focused refactor, not a tack-on to a feature PR.

<sub>Claude Code session <code>019e146a-eeb3-7743-b0f3-88e7e450674a</code></sub>
