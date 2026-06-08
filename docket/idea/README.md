# Idea

A *cool maybe* — unvetted, captured **now** because the thinking is hot. Not "should we do this?"
settled, just a spark with enough behind it that letting it cool would lose something.

A human is single-threaded: the creativity, the motivation, the edge cases that surface in the
moment don't survive deferral to a terse one-liner later — you keep the headline and lose the
thought process. So an idea is worth talking out **rambly, while you can still feel it**. That
ramble lives in the session transcript (the entry's provenance footer points back to it); it isn't
copied into the entry. What the ramble *enables* is the entry.

The entry holds the **aligned elements** — the distilled essence the AI plays back and the human
confirms (*"yes, that's exactly what I meant"*). Turning a hot, messy dictation into a
representation a human validates is the AI's strength; capturing that validated representation is
this type's whole job.

Getting there is **iterative**: dictation → playback → refine, looped as many rounds as it takes —
the human reacts, the AI re-plays, gaps close — until it's **aligned enough** (a fresh playback
draws no new forks, edges, or corrections; the human's replies have collapsed from long
counter-dictations to *"that's it"*). The entry is that converged output, not any single pass. The
probes in **Ask before filing** drive the loop.

## What to capture

The aligned essence, not the raw ramble and not a headline. Frontmatter stays minimal; the body
holds what survived the loop — distilled, but generous: an idea's nuance, motivation, and edge
cases are the point. When in doubt, keep it; losing the texture of a hot thought can't be undone,
and an idea is worth more captured rich than trimmed early. The full reasoning is always
recoverable from the session via the provenance footer the skill appends.

```markdown
---
area: docket
title: semantic search over the docket
---

**The spark.** What it is, distilled to what the human confirmed — the essence of the ramble, not
its transcript. Keep the real shape of the thought.

**Why it's exciting.** The motivation, the pull — what makes it worth doing. The pitch to
future-you who's forgotten the moment.

**Shape & edge cases.** Where it might live, how it might work, what it interacts with; the
gotchas, the "this breaks if…," the directions surfaced in the loop. Sketches welcome.

**Open questions.** What's still genuinely undecided — the calls a human has to make before this is
real. (Differs from edge cases: these stayed open *through* alignment.)
```

`area` ties the idea to the subsystem it touches (`document-search`, `selenium-browser`, `hooks`,
`cc-lib`, `claude-remote-bash`, …) so a future session working there rediscovers it. `title` is the
one-line handle. The body is the aligned essence.

## Ask before filing

General probes to drive the dictation → playback → refine loop — run a round, play your take back,
let the human react, repeat until aligned enough. The point isn't to interrogate against a
checklist; it's to surface the nuance, motivation, and edge cases that vanish if the thought cools.
Pull the thread wherever it's rich, and keep looping while playback still draws corrections:

- **The spark** — what's the idea, in your own words? Don't compress it; ramble.
- **The pull** — what makes this exciting *right now*? What would future-you regret losing?
- **The pain** — what's annoying / missing / clumsy today that this fixes?
- **Shape** — how might it work, even roughly? Where would it live?
- **Edges** — where does it get hard, weird, or break? What's the gnarly case?
- **Stakes** — what does it unlock if it works? What's it worth?
- **Forks** — what's genuinely undecided — the calls that need a human's cycles?
- **Anything left** — what haven't I asked that you're still holding?

Then play it back, and play it back again. The essence lands across rounds, not in one pass. Stop
when a playback adds nothing new and the human confirms — that converged playback is the entry.
