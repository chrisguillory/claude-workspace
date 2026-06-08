# Docket

Repo-native store for deferred work. An entry lives as `docket/<type>/NN-slug.md`, reviewed in the
normal PR flow and resolved by deleting the file — git history is the archive.

Filed via the [`add-to-docket`](../.claude/skills/add-to-docket/SKILL.md) skill.

## Types

Each is a directory with its own README codifying what it is:

- [`tech-debt/`](tech-debt/README.md)
- [`feature/`](feature/README.md)
- [`follow-up/`](follow-up/README.md)
- [`idea/`](idea/README.md)

## File vs. fix inline

File what needs the human's active verification or onboarding — net-new, judgment-laden, costly or
risky for a *human*. Just fix cheap / mechanical / repo-owned things inline. The test is "costly or
risky for a *human*?" — not "outside the current diff?"

## Numbering

Per-directory, sequential 2-digit `NN`, assigned by the skill (`max + 1`; gaps from deletions
stay). Two PRs can claim the same `NN` (different slugs → no git conflict), so
[`tests/docket/`](../tests/docket/) fails that merged state and forces a renumber.
