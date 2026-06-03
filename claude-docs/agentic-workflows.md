# Agentic workflows

Patterns for orchestrating multi-agent work deterministically (fan-out, phases, judging). The mechanics live in the `Workflow` tool; this doc captures the *patterns* worth reaching for and when they pay off. For the verification techniques these workflows run, see [empirical-verification](empirical-verification.md).

## The bake-off

The workhorse pattern, for a hard problem with several plausible fixes: don't bet on one — run them against each other and let an empirical judge pick the winner. Four phases, hard barrier between each:

1. **Map** — fan out wide (one read-only/empirical agent per region) to enumerate and probe every candidate cause; each returns a structured finding.
2. **Synthesize inventory** *(barrier)* — collapse the findings into one master inventory before any fix work, so no partial fix can win by accident.
3. **Pressure-test** — fan out a *small* set of agents, each in its own isolated git worktree off one clean baseline, each implementing a *different* competing strategy against the full inventory, each proving its fix empirically (reproduction × N + full suite), returning a typed pass/fail result.
4. **Judge** *(barrier)* — a single agent receives all results as JSON and picks the winner against pre-declared **hard gates** (it actually fixes it; it didn't cheat) then **ideal-state tie-breakers** (most direct root-cause, smallest blast radius, least fragile).

Then the orchestrator **independently re-verifies** the winning diff on the real branch before committing — a workflow's self-reported winner is a claim, not a fact.

## Techniques

- **Competing strategies in isolated worktrees.** Each agent implements a *different* candidate strategy in its own git worktree branched from one clean committed baseline. Worktree isolation lets mutually-conflicting edits run in parallel without interference; the shared baseline keeps results comparable. (Pre-commit any unrelated cleanup first, so the only variable across branches is the target.)
- **Map → synthesize → pressure-test → judge, with hard barriers.** Separate cheap parallel *investigation* from expensive parallel *implementation* with a synthesis barrier between them. Barriers force completeness before the next, costlier stage begins.
- **Ban the escape hatch.** When a cheap band-aid exists (suppress the symptom, a broad catch-all), forbid it in every implementer's brief and make *"did you use it?"* a required field the judge gates on. Removing the easy way out forces the root-cause fix — and often reveals it's simpler than the band-aid.
- **Empirical pass/fail as the judge signal.** Pick a target whose verification is a clean, repeatable yes/no the agents run themselves; require each candidate to *prove* itself with counts, not argue for itself. The judge ranks on objective gates first, subjective tie-breakers only among qualifiers.
- **Typed contracts between agents.** Define a JSON schema per phase (finding / result / verdict). Structured returns let the orchestrator mechanically aggregate, filter, count, and hand off clean data — no fragile prose parsing between stages.
- **Right-size the fan-out.** Wide for cheap read-heavy enumeration; a deliberately *small* set of competitors for the expensive implementation phase (each isolated environment pays setup overhead). Match the fan-out to the problem's real parallelism and the token/time budget.
- **Independent re-verification of the winner.** The orchestrator re-runs the empirical check on the winning result before landing it — catching over-claiming and environment-specific results a single agent's self-report hides.

## When a bake-off pays off

The preconditions that make the overhead worth it:

- **Deterministic source, non-deterministic surfacing** — the root causes are a finite, enumerable set of code sites even if *when/where* the failure manifests is random. Enumeration is bounded (Map is feasible) and repeated runs become a meaningful judge signal.
- **A clean, self-runnable verification** — a known reproduction + suite yielding an objective yes/no, reliable enough that running it N times converts a flaky symptom into a high-confidence pass.
- **A cheap band-aid exists *and* can be banned** — the tension between suppression and the proper fix is exactly what the orchestration resolves in favor of the root cause.
- **The work partitions cleanly** — read-heavy enumeration parallelizes wide; a few conflicting fixes can each run isolated within budget.

If these don't hold, a single focused pass usually beats the orchestration overhead.

## Lexicon

| Term                                                   | Meaning                                                                                                                                                               |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **bake-off**                                           | running 2–3 competing strategies against the same problem in parallel isolated environments, with an empirical judge picking the winner                               |
| **pressure-test**                                      | subjecting a candidate to a deliberately harsh empirical bar (reproduction × N + full suite) to prove it holds, especially against a non-deterministic failure        |
| **map / inventory phase**                              | the wide fan-out that exhaustively enumerates and confirms candidate causes into one master inventory the fix must address in full                                    |
| **judge phase**                                        | a single agent that selects the winner from structured results against hard gates + ideal-state tie-breakers                                                          |
| **deterministic-source / non-deterministic-surfacing** | a failure whose causes are a fixed enumerable set even though its manifestation is random — the precondition that makes a bake-off worthwhile                         |
| **ban the escape hatch**                               | forbidding the cheap symptom-suppressing shortcut and gating the judge on it, to force the root-cause fix                                                             |
| **convergent strategies**                              | independent approaches arriving at the same root cause (corroboration the diagnosis is right); convergent *failure* instead implicates the substrate, not the methods |
| **innocent bystander**                                 | a symptom that lands on an arbitrary, blameless location each run — a tell that several symptoms share one upstream cause                                             |
| **isolated worktree**                                  | a per-agent git worktree off one clean baseline, so conflicting candidate edits build and test in parallel on comparable footing                                      |
| **independent re-verification**                        | the orchestrator re-running the check on the winner itself before committing — treating a self-report as a claim, not a fact                                          |
