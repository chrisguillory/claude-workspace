export const meta = {
  name: 'where-am-i-build',
  description: 'Build the where-am-i quest-map: roots → per-node ground-truthing → a top map that rolls the nodes up',
  phases: [
    { title: 'Roots', detail: 'identify the user-driven threads from the spine (intent only)' },
    { title: 'Nodes', detail: 'one agent per root: ground-truth against git, write detail, return facts' },
    { title: 'Quest-map', detail: 'assemble the top map from the node facts, then validate' },
  ],
}

// Deterministic, resumable build. The phase order is load-bearing: the top map is assembled LAST, from the
// nodes' authoritative per-root ground-truthing — so it can't freeze a wrong overlay the way a top-first pass
// did. The structural validator can't catch a wrong PR label; the nodes are the semantic check, so they run
// first. Roots come from the roots agent's return (not args), sidestepping the args-as-string trap.
//   args: { gatherDir } — holds user-intent-spine.txt / gh-ground-truth.txt / session-metadata.txt;
//   quest-map.md + nodes/{slug}.md are written there.
const input = typeof args === 'string' ? JSON.parse(args) : args
const dir = input.gatherDir
const SPEC = '.claude/skills/where-am-i/SKILL.md'
const VALIDATOR = '.claude/skills/where-am-i/validate-quest-map.py'
const SPINE = `${dir}/user-intent-spine.txt`
const TRUTH = `${dir}/gh-ground-truth.txt`
const META = `${dir}/session-metadata.txt`
const MAP = `${dir}/quest-map.md`

const ROOTS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['roots'],
  properties: {
    roots: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'slug', 'title'],
        properties: {
          number: { type: 'integer' },
          slug: { type: 'string' }, // NN-kebab, e.g. 01-bootstrap-hygiene
          title: { type: 'string' },
        },
      },
    },
  },
}

const FACTS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['slug', 'landed', 'prs', 'span', 'status'],
  properties: {
    slug: { type: 'string' },
    landed: { type: 'boolean' }, // did the thread's goal land?
    prs: { type: 'array', items: { type: 'string' } }, // authoritative PR numbers for this root
    span: { type: 'string' }, // [from → to]
    status: { type: 'string' }, // one-line current status
  },
}

phase('Roots')
const skeleton = await agent(
  `Read the render spec at ${SPEC} — especially "Node rule — PURE user-intent". Read ${SPINE} (the user's
verbatim messages, in order). Identify the distinct threads the USER drove — a goal they stated, or something
they explicitly reacted to. Do NOT ground-truth or look at git yet; this pass is intent structure only.
Return the root list: each root's number (1..N, first-touched order), an NN-kebab slug, and a one-line title.`,
  { label: 'roots', phase: 'Roots', schema: ROOTS_SCHEMA },
)
if (!skeleton?.roots?.length) throw new Error('no roots identified from the spine')
log(`${skeleton.roots.length} roots → fanning out per-node ground-truthing`)

phase('Nodes')
const facts = (
  await parallel(
    skeleton.roots.map((r) => () =>
      agent(
        `Read the render spec at ${SPEC} for context. Pre-gather ONE node of the quest-map — the
"compute once, cat later" detail file someone opens weeks later and never has to re-search.
Root [${r.number}] "${r.title}". From ${SPINE} (the verbatim user directives) pull this thread's sub-arc of
directives (quote the load-bearing ones, in order). GROUND-TRUTH the deliverables AUTHORITATIVELY: consult
${TRUTH} and go to \`git log\` / \`gh\` directly for THIS root's real PRs — don't trust a broad guess. Capture
the decisions + why and what's still open. Write the detail to ${dir}/nodes/${r.slug}.md, then return this
root's facts: slug, whether it landed, its authoritative PR numbers, its [from → to] span, a one-line status.`,
        { label: `node:${r.slug}`, phase: 'Nodes', schema: FACTS_SCHEMA },
      ),
    ),
  )
).filter(Boolean)

if (facts.length < skeleton.roots.length) {
  const got = new Set(facts.map((f) => f.slug))
  const missing = skeleton.roots.filter((r) => !got.has(r.slug)).map((r) => r.slug)
  throw new Error(`nodes incomplete (${missing.join(', ')}) — resume to retry just these`)
}
log(`${facts.length} nodes ground-truthed → assembling the quest-map from their facts`)

phase('Quest-map')
const bySlug = Object.fromEntries(facts.map((f) => [f.slug, f]))
const roots = skeleton.roots.map((r) => ({ ...r, ...bySlug[r.slug] }))
await agent(
  `Read the render spec at ${SPEC} and follow it exactly. Assemble the TOP quest-map and write it to ${MAP}.
These roots were already authoritatively ground-truthed per-node — trust their facts (landed, prs, span,
status) over any broad guess for the check-marks, ranges, and PR overlay:
${JSON.stringify(roots, null, 2)}
Pull the header counts from ${META}; build the cross-root PR overlay + docket overlay per the spec (read
${TRUTH} for the full PR list). Then run \`uv run --no-project --script ${VALIDATOR} ${MAP}\` and fix every
issue until it prints "valid ✓". Return only a one-line confirmation.`,
  { label: 'quest-map', phase: 'Quest-map' },
)

return { map: MAP, nodes: facts.map((f) => f.slug) }
