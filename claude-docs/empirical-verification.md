# Empirical verification

How we verify here: run the real system against real (or deliberately induced) conditions and read the result, in preference to mocked tests. The normative principle lives in [CLAUDE.md → Test discipline](../CLAUDE.md#test-discipline); this doc captures the *loop*, the recurring *techniques*, and the *lexicon*.

## The loop

Build a small increment → **induce** the real condition → **observe** ground truth → iterate. Each step lands working and verified before the next. A change isn't done because it *should* work; it's done when you've watched it work.

## Techniques

- **Probe.** A minimal, throwaway, single-purpose experiment that induces *one* real behavior and reads ground truth from an observable marker. Run it in an isolated scratch dir (e.g. a temp dir — let Claude Code pick where), driven by a deterministic, no-preamble instruction that ends by echoing a marker you read back (e.g. `echo PROBE_RESULT cwd=$(pwd) …`).
- **Apparatus.** A probe promoted to a reusable rig — the repeatable harness around the experiment.
- **Probe-then-productionize.** Prove the mechanism with a probe first; build the production component only once it's proven — you can't build on a mechanism you haven't confirmed.
- **Revert → reproduce → reapply.** Put the system in the broken state, confirm the failure, then fix and re-verify — unambiguous before/after proof instead of reconstructing causality from logs.
- **Falsify the guard.** A regression test proves nothing until you've watched it fail *without* the fix — strip the fix, confirm the test goes red on the exact bug, restore. A test that stays green when the bug is reintroduced is a *false guard*: it locks nothing, yet its green reads as proof.
- **Capture reality → empirical surface → re-ground.** Capture the real interface from observation (e.g. live traffic) into a doc, and re-ground the plan against that captured surface rather than trusting upstream docs. A captured baseline is valid only for the inputs it was taken against — re-capture when the seeds, schema, or shape change, or a diff against the stale baseline becomes a silent false verdict.
- **Claim → Test → Verdict.** Every load-bearing claim gets a real run with verbatim output and a ✅ / ⚠️ / ❌ verdict; *"should work because X"* with no actual output is the smell. Interrogate *why* it passed — a pass via a spurious correlation is a false pass — and name the concrete misclassifications instead of banking an aggregate score.
- **Cross-harness ground truth.** Let independent evidence override assumption — run two harnesses and confirm they agree; trust what a run actually observed over what a single path assumes.
- **Trust the evidence, distrust the verdict.** Read the actual output — a green "CONFIRMED" can be a two-layer lie where the wrapper reports one thing and the evidence says another. Where the surface signal is flaky (a spinner, a banner), key the verdict on a structured side-channel — a token-count delta, an API metric — that can't lie.
- **Bake-off competing approaches.** When several candidate fixes or designs exist, run them against each other and let the results pick the winner rather than arguing it out — don't assume which works, let the run reveal it. (Running this at scale — parallel isolated worktrees + a judge — is its own pattern; see [agentic-workflows](agentic-workflows.md).)
- **Verify the artifact, not only the effect.** Decide *what* to assert against: the generated artifact (the exact rendered query/command/config the code emits) or the runtime effect (seed, run, observe). Prefer the artifact when a human can eyeball it — a behavioral pass only proves the test environment reproduced the behavior, which may not match production.
- **Fleet-variance differential diagnosis.** When a change passes on some cases/hosts and fails on others, treat the natural pass/fail split as a ready-made controlled experiment: measure the suspected variable across *both* groups *before* patching. The falsification is the payoff — if the variable is identical across pass and fail, the hypothesis is dead and the real cause lies elsewhere.
- **Planted-tracer positive control.** To prove a read path is genuinely live (not cached, stale, or hallucinated), salt the upstream source with a unique marker and confirm that exact string surfaces downstream. Its presence verifies the path; its absence exposes a fake read.

## Lexicon

| Term                        | Meaning                                                                              |
|-----------------------------|--------------------------------------------------------------------------------------|
| **probe**                   | minimal throwaway experiment inducing one real behavior, observed via a marker       |
| **apparatus**               | a probe promoted to a reusable rig/harness                                           |
| **marker** / `PROBE_RESULT` | the echoed signal that makes a behavior observable as ground truth                   |
| **empirical surface**       | the real interface captured from observation (e.g. live traffic), not from docs      |
| **ground truth**            | observed evidence, treated as authoritative over assumptions or a tool's own verdict |
| **load-bearing**            | the single decisive empirically-established fact in an argument                      |
| **false guard**             | a regression test that stays green when its bug is reintroduced — it locks nothing   |
| **re-grounding**            | re-aligning a plan against freshly captured reality                                  |

## When a test earns its place instead

Empirical verification is primary, but an automated test is right for what's hard to stage empirically — a regression for a fixed bug, a rare/hard-to-trigger failure mode, wire-format round-trips (see [CLAUDE.md → Test discipline](../CLAUDE.md#test-discipline)). Happy-path tests, library-behavior tests, and assertions that re-prove what every empirical run already shows do **not** earn their place; they become long-lived drift.
