export const meta = {
  name: 'where-am-i-build',
  description: 'Build the where-am-i quest-map: roots → per-node ground-truthing → a top map that rolls the nodes up',
  phases: [
    { title: 'Roots', detail: 'identify the user-driven threads from the spine (intent only)' },
    { title: 'Nodes', detail: 'one agent per root: ground-truth against git, write detail, return facts' },
    { title: 'Quest-map', detail: 'assemble the top map from the node facts, then validate' },
  ],
}

// A dynamic Workflow — the `Workflow` tool executes this file. The plumbing it provides:
//   phase(title)            groups the agents below it in the progress display
//   agent(prompt, {schema}) spawns a subagent; with a JSON `schema`, its return is validated to that shape
//   parallel(thunks)        runs thunks concurrently → results array (null for any agent that died)
//   log(msg) / args         narrator line / the { gatherDir } object passed by the caller
// Plain JS is mandatory (the runtime rejects TS syntax); the JSDoc @typedefs below are the typing layer.
//
// The phase order is load-bearing: the top map is assembled LAST, from the nodes' authoritative per-root
// ground-truthing — so it can't freeze a wrong overlay the way a top-first pass did (the structural validator
// can't catch a wrong PR label; the nodes are the semantic check, so they run first). Roots come from the roots
// agent's return, not args, sidestepping the args-as-string trap.
//
// args: { gatherDir } — holds user-intent-spine.txt / gh-ground-truth.txt / session-metadata.txt;
// quest-map.md + nodes/{slug}.md are written there.

/** @typedef {{ number: number, slug: string, title: string }} Root — one user-driven thread (Roots phase). */
/** @typedef {{ slug: string, landed: boolean, prs: string[], span: string, status: string }} NodeFacts
 *  A root's ground-truthed facts (Nodes phase). `slug` is attached by the workflow from the known Root —
 *  the node does NOT return it, because a re-reported slug can drift and break the facts→top merge. */

const input = typeof args === 'string' ? JSON.parse(args) : args
const gatherDir = input.gatherDir
const renderSpec = '.claude/skills/where-am-i/SKILL.md'
const validator = '.claude/skills/where-am-i/validate-quest-map.py'
const userIntentSpine = `${gatherDir}/user-intent-spine.txt`
const ghGroundTruth = `${gatherDir}/gh-ground-truth.txt`
const sessionMetadata = `${gatherDir}/session-metadata.txt`
const questMap = `${gatherDir}/quest-map.md`

// Roots phase: the user-driven thread list, intent only — no ground-truthing yet.
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

// Nodes phase: a root's authoritative facts. No `slug` here — the workflow attaches the known Root slug
// after the agent returns (a node-reported slug can drift from the Root's and break the merge below).
const FACTS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['landed', 'prs', 'span', 'status'],
  properties: {
    landed: { type: 'boolean' }, // did the thread's goal land?
    prs: { type: 'array', items: { type: 'string' } }, // authoritative PR numbers for this root
    span: { type: 'string' }, // [from → to]
    status: { type: 'string' }, // one-line current status
  },
}

phase('Roots')
const skeleton = await agent(
  `Read the render spec at ${renderSpec} — especially "Node rule — PURE user-intent". Read ${userIntentSpine}
(the user's verbatim messages, in order). Identify the distinct threads the USER drove — a goal they stated, or
something they explicitly reacted to. Do NOT ground-truth or look at git yet; this pass is intent structure only.
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
        `Read the render spec at ${renderSpec} for context. Pre-gather ONE node of the quest-map — the
"compute once, cat later" detail file someone opens weeks later and never has to re-search.
Root [${r.number}] "${r.title}". From ${userIntentSpine} (the verbatim user directives) pull this thread's
sub-arc of directives (quote the load-bearing ones, in order). GROUND-TRUTH the deliverables AUTHORITATIVELY:
consult ${ghGroundTruth} and go to \`git log\` / \`gh\` directly for THIS root's real PRs — don't trust a broad
guess. Capture the decisions + why and what's still open. Write the detail to ${gatherDir}/nodes/${r.slug}.md,
then return this root's facts: whether it landed, its authoritative PR numbers, its [from → to] span, a one-line
status.`,
        { label: `node:${r.slug}`, phase: 'Nodes', schema: FACTS_SCHEMA },
      ).then((nodeFacts) => nodeFacts && { ...nodeFacts, slug: r.slug }),
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
  `Read the render spec at ${renderSpec} and follow it exactly. Assemble the TOP quest-map and write it to
${questMap}. These roots were already authoritatively ground-truthed per-node — trust their facts (landed, prs,
span, status) over any broad guess for the check-marks, ranges, and PR overlay:
${JSON.stringify(roots, null, 2)}
Pull the header counts from ${sessionMetadata}; build the cross-root PR overlay + docket overlay per the spec
(read ${ghGroundTruth} for the full PR list). Then run \`uv run --no-project --script ${validator} ${questMap}\`
and fix every issue until it prints "valid ✓". Return only a one-line confirmation.`,
  { label: 'quest-map', phase: 'Quest-map' },
)

return { map: questMap, nodes: facts.map((f) => f.slug) }
