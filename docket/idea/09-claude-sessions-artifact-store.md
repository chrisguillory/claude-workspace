---
area: claude-sessions
title: claude-sessions — a browsable per-session artifact store the model reads + writes
---

**The spark.** A top-level **`claude-sessions/`** directory, one folder per session keyed by the
first 8 chars of the session ID (`claude-sessions/4d854b28/`), as the durable, **browsable** home
for the artifacts a session produces that today get **relegated to scattered locations** —
`/private/tmp/…/tasks`, ad-hoc temp dirs, wherever a given tool happens to drop its output.
Gitignored / machine-local. The model is taught (a CLAUDE.md / system-prompt convention) to
**write artifacts there and read prior ones back** — both an archive you can open and click
through *and* a working-memory surface, not just storage.

**Why it's exciting.** Today a session's output is *exhaust* — flung into temp dirs you never see,
gone or unfindable later. Nothing is a *named place you can open and ask "what did this session
make?"* This turns ephemeral output into an **owned, navigable layer**: you browse it like a
filesystem (it is one), the model can resume a past session by reading its folder, and tooling
accretes around a stable location instead of chasing temp paths. The artifact that surfaced the
idea — wanting a durable, private home for `personal-identifier-hygiene-details.md` — is the
charter example: something that wanted to persist and be findable, with nowhere good to live.

**Shape & edge cases.**
- **Gitignored / local-only** is deliberate: the repo is public and sessions are private, so
  committing output would re-import the PII/noise hygiene work removes. Local by design — not
  committed or synced; cross-machine *access*, when needed, is claude-remote-bash's domain
  (already solved on the mesh), not this store's to re-solve.
- **Leaves `~/.claude/projects/<id>/` alone.** The native session/transcript dir is not relocated
  or touched; this store is for the *out-of-the-box*, otherwise-homeless artifacts. (Which
  *canonical* session artifacts — session memory, per-session configurable state, the things
  definitive to every Claude Code session apparatus — might also be worth surviving here is an
  open research question, below.)
- **One flat directory for every session, including worktree ones.** A session that runs in a git
  worktree still lands in the single `claude-sessions/<id8>/` tree; its worktree origin is
  **metadata** on the folder, not a separate location. Keying by session ID (not by where it ran)
  is what lets all sessions coexist long-lived in one browsable place.
- **Prefix-collision strategy:** 8 hex chars collide only astronomically rarely; when it happens,
  fall back to the **full session UUID** for the colliding folder(s).
- **where-am-i (idea/PR #244) stays standalone** — its quest-map could land here as one artifact
  among many, but there's no coupling; the idea is the *location*, neutral to producers. Distinct
  from `session-miner` (idea 04) and `corollary-enrichment` (idea 06), which *read* the transcript
  corpus; this *stores* artifacts.
- Write/read is **convention-first** (the model follows a documented rule); a hook that
  auto-routes output into the session folder is a heavier later option.

**Open questions.**
- **Which canonical Claude Code session artifacts** (session memory, per-session configurables —
  whatever's "definitive for every session apparatus") should also survive here vs. stay in their
  native homes? Wants a **dynamic-workflow / deep-research** pass when the idea is picked up.
- How does the model reliably *know* to use it — a CLAUDE.md convention, a system-prompt line, or a
  hook — and does on-by-default writing create clutter?
- What's worth persisting into the store vs. left as ephemeral temp output?
- Per-session **index / manifest** (a `top.md`?), or is the bare folder listing enough?

<sub>Claude Code session <code>4d854b28-0603-4eaa-9185-a91bfc8198d5</code></sub>
