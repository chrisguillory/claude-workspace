---
area: selenium-browser
title: Selenium fallback for deep-research on blocked sites + the concurrent-browser layer it needs
---

**The spark.** Fork the bundled `deep-research` dynamic workflow into a **repo-owned, versioned**
variant whose Fetch stage **falls back to the `selenium-browser` MCP when WebFetch can't deliver** —
blocked by permission, returned empty, bot-walled, paywalled, or JS-rendered (Amazon is the
canonical case). WebFetch stays the fast default; Selenium is the fallback that reaches what WebFetch
can't see. The load-bearing sub-requirement: **`selenium-browser` must support concurrent multi-agent
access** — a browser-pool / multi-context layer — because deep-research fans out up to ~15 fetch
agents in parallel against what is today a **single shared browser**. Fold in the **in-progress
multi-agent Selenium work** rather than re-deriving it. Codify the whole thing in-repo (a workflow
script + a thin skill wrapper) so it's PR-reviewable, versioned, and ours to iterate on.

**Why it's exciting.** This very session is the charter example. The deep-research-style fan-out (and
ad-hoc research subagents) kept hitting **WebFetch permission blocks and bot walls**, and the only way
through was the model **hand-driving Selenium** for the high-value sources — logged-in order history,
live prices, the things WebFetch simply cannot see. Worse, the snippet-only research that resulted
produced **confidently wrong facts** (a ~$130 price that was really $378; a "Sonnet Twin25G" that was a
desktop PCIe card, not a Mac adapter) that only live browsing caught. A research harness that
**natively falls back to a real browser** closes that gap — automated research becomes trustworthy on
exactly the commerce / paywalled / JS-heavy sources that defeat WebFetch. And building it the repo's
way — fork the best of what exists, own it, version it — turns a one-off manual rescue into a durable
capability instead of a trick the model re-improvises every time.

**Shape & edge cases.**
- **Where it lives:** a workflow script in-repo plus a project-skill wrapper — the same two-hats
  pattern the bundled `deep-research` already uses (a `/skill` entry point on top, a dynamic workflow
  underneath). Start from the bundled workflow's *materialized* script as the template
  (`~/.claude/projects/.../workflows/scripts/deep-research-wf_*.js` is the real, readable source). The
  `claude-code-best/claude-code` community reconstruction (`packages/workflow-engine/`,
  `examples/research-report/`, `docs/superpowers/specs/`) is **conceptual reference only** — a
  deobfuscated third-party build, not authoritative for what actually ships.
- **The hard part is the browser, not the swap.** Changing the Fetch prompt from "use WebFetch" to
  "use selenium-browser" is trivial; surviving **~15 concurrent fetch agents on one browser** is the
  real design problem. Options: serialize the fetch stage (simple, slow); a **browser-pool** of N
  isolated contexts (the proper layered fix); or **hybrid** — WebFetch-first, Selenium only for
  blocked/empty results, serializing just those (keeps speed on normal pages, spends the browser only
  where it earns its cost).
- **This is a "missing-layer" signal** per the repo's layered-architecture principle: two callers now
  want concurrent browser access (workflow fetch agents *and* the other contributor's multi-agent
  work), so the lower **concurrency layer in `selenium-browser`** should exist. Coordinate with /
  absorb that in-progress work — don't fork a second mechanism.
- **Existing single-instance constraints bound the design:** `navigate_with_user_data_dir`'s
  exclusive SingletonLock, the Chrome-vs-Chromium keychain binding, and profile/auth state (the
  logged-in-Amazon path) all have to survive whatever pooling model wins — isolated `--user-data-dir`s
  vs. browser contexts vs. tabs each trade off isolation, auth-sharing, and memory differently.
- **Fits the repo principles cleanly:** *Fork or patch, don't wait* (extend the bundled workflow + the
  ecosystem's best), *Leverage existing infrastructure* (build on the `selenium-browser` MCP + the
  workflow engine), *Ideal state over backwards compat* (fix the source — give the MCP real
  concurrency — rather than papering each call with a manual workaround).

**Open questions.**
- **Fetch strategy:** hybrid WebFetch-first-then-Selenium-fallback vs. Selenium-always for known-hard
  domains? What signal marks a fetch "failed enough" to escalate — empty body, a bot-wall heuristic,
  an HTTP status, a content-length floor?
- **Concurrency model & pool size:** separate user-data-dirs vs. contexts vs. tabs; how many parallel
  browsers are safe/fast on one machine; how auth/profile state is shared or isolated per worker.
- **Coordination:** what exactly *is* the other contributor's multi-agent Selenium work, and how do we
  fold it in instead of duplicating it? Needs a sync before any implementation.
- **Relationship to `bughunter`** — the sibling architecture deep-research was "ported from" (per its
  own header comment). Not yet understood; worth reading when this is picked up, in case the right move
  is a shared fan-out → adversarial-verify spine that both workflows reuse.
- **Upstream vs. local:** keep this a repo-owned fork, or is any of it worth shaping toward the bundled
  workflow so it could be contributed back?

<sub>Claude Code session <code>b76db342-d3c8-434a-b015-1b1ee4619314</code></sub>
