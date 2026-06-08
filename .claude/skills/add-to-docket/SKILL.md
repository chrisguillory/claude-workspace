---
name: add-to-docket
description: "Add a deferred entry to the repo's docket — a versioned NN-slug.md under docket/<type>/ (tech-debt, feature, follow-up, or idea), reviewed in the normal PR flow and picked up later. Resolve by deleting the file."
argument-hint: "<tech-debt|feature|follow-up|idea> [short description]"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(.claude/skills/add-to-docket/gather-context.py:*)"
  - "Write"
---

# Add to Docket

A review / canvass / audit / implementation pass surfaced something real but not for now. Acting
on it mid-flow derails the session — add it to the docket so it's off your plate, captured and
versioned, and picked up later (ideally when a future refactor touches that *area* and folds in
everything local to it).

Each entry is `docket/<type>/NN-slug.md`, reviewed in the diff you already use. See
[`docket/README.md`](../../../docket/README.md) for the model; each `docket/<type>/README.md`
codifies what that type is and what its entry looks like.

The type comes from the invocation — the human names it; you don't classify. You usually already
hold the thing (you just surfaced it); recover fuzzy detail via
`mcp__document-search__search_documents` (collection `document-chunks`) rather than re-deriving it.

## Gather

!`.claude/skills/add-to-docket/gather-context.py $ARGUMENTS`

For the named type, gather gives you:

- **Write to** — `docket/<type>/NN-{slug}.md`, with `NN` already assigned. Fill `{slug}` with a
  short kebab-case slug of the *effect* (`untyped-dicts-where-a-model-belongs`), not a ticket.
- that type's **README**, pushed out inline — follow it for what the entry captures.
- the **Footer** — append it verbatim (session provenance; the depth is recoverable from the
  indexed transcript by that ID).

It **fails fast** if the target directory already holds a duplicate `NN`. If gather shows only the
overview (it didn't resolve a type), re-run it with the type:
`.claude/skills/add-to-docket/gather-context.py <type>`.

## File it

Draft per the type's README, fill the slug, append the Footer, and `Write` it to the path — the
Write diff is the review. The entry rides into a PR like any change, and `tests/docket/` enforces
per-dir `NN` uniqueness when it merges to main.

## Key rules

- **The type is named, not classified.** Follow the README gather surfaces for what the entry
  captures and how deep it goes — that varies by type.
- **The transcript holds the depth** — the footer's session ID recovers the full reasoning.
- **Area + key files are load-bearing** — they make `rg -l 'area: X' docket/` (or a semantic
  search over the docket) complete when a future refactor folds in everything local to that area.
- **Resolve is deletion** — git history is the archive.
- **Just fix cheap/mechanical/repo-owned things inline** — the bar is human cost/risk, not
  out-of-scope-ness.
