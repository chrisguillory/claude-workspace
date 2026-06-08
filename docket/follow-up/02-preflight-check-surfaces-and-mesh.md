---
area: preflight-check
title: surface findings at session start and run checks across the mesh
---

**Next step.** Extend `preflight-check` past its first check (which ships in #231): (1) prove which
SessionStart channel actually reaches the user, then wire `hooks/session-start.py` to surface the
fast local checks at startup/resume (silent when healthy); (2) a mesh runner that fans the check
suite across hosts via `claude-remote-bash` — an unreachable host is a `Finding`, not a crash;
(3) the lower-priority tail — a statusLine health glyph and `preflight-check-mcp` check/fix tools.
Design notes + the empirical SessionStart-channel findings are in `mcp/preflight-check/PLAN.md`.

**Waits on.** Human cycles — deliberately deferred to ship the first working check
(`dns_resolver_wedge`) standalone; these surfaces aren't needed for it to be useful. Pick up when a
session next touches `preflight-check`.

<sub>Claude Code session <code>2f2986bf-e0c8-4f74-9c4f-81e4dec6f63e</code></sub>
