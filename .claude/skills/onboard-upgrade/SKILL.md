---
name: onboard-upgrade
description: "Absorb a new Claude Code version and/or model into this repo — the reusable recipe for what to investigate and how to judge it. Use when CC updates, the model changes, or a loud failure smells like version/model drift."
argument-hint: "[from→to version, and/or model A→B]"
disable-model-invocation: false
effort: max
---

# Onboard a Claude Code / Model Upgrade

The reusable recipe for absorbing a new Claude Code version or model. You are the
**judge**; this is your recipe, not a script. It says *what to look for and how to
think* — never *how mechanically*. Fill in the how at run time.

## The two levels (don't collapse them)

- **Meta-skill** — this file. Reusable across every future upgrade.
- **Instance** — what one run produces: a dated action plan + onboarding synopsis for
  *that* specific upgrade (first exemplar: `scratch/cc-upgrade-2026-05-29/`).

An instance is this skill run once. Overarching lessons from an instance roll back *up*
into this file (see Self-improve); the little fixes stay in the instance.

## When this fires

- The user says some form of *"a new CC version / model dropped — onboard it."*
- A loud failure surfaces that smells like drift (a schema `ValidationError` under the
  strict tripwire, a statusline crash, a workaround behaving differently).

The repo is **designed to fail loudly** when its assumptions break — we can't predict
upstream and we don't try to. **Those loud failures are your entry points, not bugs to
engineer away.** Do not build machinery to suppress them.

## Mindset

- **LLM-as-judge.** You will hit unknown-unknowns no fixed tool can anticipate.
  Investigate with judgment — canvas, code review, research, binary inspection — not
  brittle automation. Trust the judge; that is the point.
- **Read-only by default, human-gated.** Evaluate and propose; the human decides what to
  apply.
- **Right altitude.** Resist dropping into extractors, pipelines, or parallelism plumbing.
  Those are run-time details, not the recipe.

## Bifurcate two axes

A "new version" is usually two independent things that sometimes ship together (a model
can land *inside* a CC version):

- **CC version** — plumbing: hook / statusline / settings schemas, env vars, CLI flags &
  commands, new tools, the binary itself.
- **Model** — semantics: capabilities, effort / thinking / context behavior, pricing,
  model-specific failure modes, prompting guidance.

Always disentangle *"is this a plumbing change or a model change?"* — the version delta
can be tiny while the model jump is everything (2.1.138→2.1.156 was really Opus 4.7→4.8).
Watch the silent hazard: a stable alias (`opus[1m]`) re-resolving to a new model.

## The vectors — investigate each with judgment

**External**
- Changelog delta for the version range (the cached changelog is byte-faithful ground truth).
- Model capabilities & deltas vs the prior model.
- Community feedback & known / brick bugs — weight the ones matching *this* setup.
- Notable *new* features worth adopting — don't only hunt for breakage.

**Internal (our repo)**
- What did this invalidate in our code? Right tool for the signal: keyword-anchored facts
  → search; semantic drift (stale docs, over-strict models, missing guards) →
  canvas / code-review-as-judge. Scope to first-party; skip vendored deps.
- **Inspect the binary as ground truth.** The installed CC binary is the authoritative
  record of current behavior — it's how you confirm a workaround is truly dead, and how you
  catch that your *own* assumptions about its shape have gone stale (its format changes
  between releases). Judge from it; don't presume its structure.
- **Hunt rotted version-pins.** Stale version-stamped constants and snapshots — hardcoded
  version numbers, "derived @vX" doc sections, enumerations captured at an old version — are
  the most common drift by far. Spotting them is the vector; decide hand-fix vs re-derive
  per case. That's judgment, not a reason to build extractors.
- Diagnose anything visibly broken to root cause.
- Interrogate existing workarounds: *still needed, or did upstream fix it?* Decide with
  evidence (e.g., version-keyed telemetry gone silent across releases = dead code).

## The two deliverables

1. **Bifurcated action list** — what to change, grouped by subsystem × severity, each
   grounded in evidence (file:line, source, version). Read-only proposals.
2. **Onboarding synopsis** — new capabilities worth trying.

Write them as a dated instance; read the previous instance first for baseline.

## Discipline (non-negotiable)

- **Ground everything; distrust your own setup too.** Fabrication is endemic — subagents,
  the reviewer, *and the orchestrator* invent plausible-but-false details (a mis-wired
  fan-out, an invented metric or quote) and then rationalize them. Verify your *own* wiring
  and inputs before fanning out; every finding cites file:line / URL / offset; high-impact
  claims get an independent check; cross-check against artifacts that can't lie.
- **Right tool for the signal.** Don't run a wall-to-wall canvas over facts a grep would
  catch; don't grep for things only judgment can see.
- **Scale to the delta.** A patch bump is a changelog scan; a model-family change is the
  full sweep + a brick-bug hunt.

## Self-improve

After each run, fold *overarching* lessons back into this file — not the little fixes.
Seeded from the first run (2026-05-29):

- The CC-vs-model bifurcation, and the silent-alias resolution hazard.
- Ground / adversarially verify — fabrication is endemic (subagents, reviewer, *and*
  orchestrator); treat every summary, including your own, as a hypothesis until grounded,
  and verify your own fan-out wiring before trusting it.
- Right-tool-for-signal — an over-scoped canvas is mostly waste; grep the keyword-anchored,
  canvas the semantic.
- Fail-loud is the signal, not the enemy — don't build "anti-rot" automation that fights the
  repo's own design. (But *spotting* rotted version-pins is a legitimate judgment vector.)
