---
area: public-repo-hygiene
title: second workplace-email scrub + pin GitHub email surfaces + path scrub
---

**Next step.** One cleanup PR plus an account-settings pass:

1. **Second history scrub** — workplace-email commits re-drifted onto `origin/main` after the
   first scrub; rewrite them out with the round-1 playbook (method, gotchas, machine/session
   pointers, backup location: see the operator-private gist named
   `personal-identifier-hygiene-details` — resolve via `gh gist list`; deliberately not linked
   here).
2. **Root cause, so it can't drift a third time** — invert the misconfigured machine's git
   identity to the settled design (global = personal; `includeIf` scopes the work email to the
   work checkout only), and converge a per-clone noreply via `dotfiles/install.sh`
   (`git config user.email`).
3. **GitHub account surfaces** — the web-merge email dropdown allows a mis-click among verified
   emails that no git config prevents: investigate primary/web-commit email, the keep-private +
   block-exposing-pushes settings, and whether the work address should be de-verified entirely.
4. **Path scrub** — the 16 tracked files carrying literal `/Users/{name}` home paths, per
   CLAUDE.md *Path Documentation* + *Public Repository*.

**Waits on.** The dotfiles PR landing (`install.sh` — the per-clone convergence home — ships in
it), and a deliberate moment for the rewrite: round 1 proved a history rewrite auto-closes every
open PR (11 open today) and ripples across the mesh — schedule it, don't drive-by it.

<sub>Claude Code session <code>4d854b28-0603-4eaa-9185-a91bfc8198d5</code></sub>
