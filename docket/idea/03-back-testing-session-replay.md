---
area: docket
title: back-testing — replay a past session against a changed system
---

**The spark.** Re-drive a *past* session fresh — in the system state it actually ran in, but with
one thing changed (a new linter, a doc tweak, a convention) — and compare the outcome to what
happened. Back-testing, applied to your own workflow: "does this change actually make the session go
better?" The enabler is **Time Machine**, which snapshots the *full* constituent state — untracked
files and system areas, not just git-tracked — so the replay is faithful in a way `git checkout`
can't be.

**Why it's exciting.** It's empirical verification turned on the *meta* level — the repo's own
"run it against real conditions" value, pointed at how the AI works rather than at the product. A
tooling or doc change stops being a guess and gets a measured before/after.

**Shape & edge cases.** Reconstruct the past state → replay with the new input → measure the delta.
It's the *backward, convergent* twin of daydream (which generates forward) — shared "run the repo in
an earlier state" machinery, opposite jobs. The real confound is **LLM non-determinism + model
drift**: a single replay proves little. Measure distributions across N replays, and pin the model so
the change is the only moving variable.

**Open questions.** What's the outcome metric — fewer correction rounds, less wasted tool-calls,
faster convergence? How many replays make a signal? And how much of "the system state" does Time
Machine actually need to restore for a faithful run?

<sub>Claude Code session <code>be8ada35-c290-4d7d-88d3-df113a25f8b8</code></sub>
