---
area: docket
title: code-miner — surface latent debt the linters don't catch
---

**The spark.** Mine the *code* corpus for debt hiding in plain sight — accumulating `# type:
ignore`s, `TODO` / `FIXME` / foot-gun comments, and slow design-principle drift the linters don't
yet catch — and propose `tech-debt` entries before it festers.

**Why it's exciting.** The linters enforce *known* patterns at commit time; the code-miner finds the
*latent, accumulating* stuff — a `type: ignore` pile that grew one suppression at a time, a principle
quietly eroding across files — and names it while it's still cheap to fix. Debt nobody filed because
nobody was looking finally gets looked at.

**Shape & edge cases.** `rg` + semantic scan over the code, proposing specific candidates
(`file:line` + why it's debt); the human curates. It's the *code-corpus* twin of the session-miner.
Two failure modes: false positives (a suppression can be legitimate — propose, never auto-file), and
the line with the linters — when a pattern is clear and recurring it should *become* a linter (a
just-do), not a finding the miner re-surfaces forever.

**Open questions.** Which signals carry the best signal-to-noise — suppressions, comment markers,
principle-drift? How do you tell a legit suppression from real debt without a human in the loop? And
where's the handoff from "miner proposes" to "linter enforces"?

<sub>Claude Code session <code>be8ada35-c290-4d7d-88d3-df113a25f8b8</code></sub>
