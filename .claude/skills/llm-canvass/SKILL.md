---
name: llm-canvass
description: "Apply an LLM judgment task to every in-scope file in the repo, in parallel. Bin-packs files into balanced token slices, fans out general-purpose subagents that read full files, synthesizes results."
argument-hint: "--type|--glob|--all <task>"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(.claude/skills/llm-canvass/plan-slices.py:*)"
  - Agent
---

# LLM-Canvass: Parallel LLM Judgment over the Codebase

Run a structured LLM judgment task across every in-scope source file in
parallel. The user gives the task; the skill handles enumeration, slicing,
fan-out, and synthesis.

## Phase 1: Plan slices

Run `plan-slices.py` via the `Bash` tool — single-quote the task description so
parens, backticks, quotes, and other shell metacharacters in the user's task
pass through unparsed. Flags (`--type`, `--glob`, `--all`, `--agent`,
`--background`, `--max-agents`, `--per-agent-tokens`) come before the task.

```
.claude/skills/llm-canvass/plan-slices.py --type py 'your task description'
```

The companion script:
- Enumerates files via `rg --files` under the required scope (`--type`, `--glob`, or `--all`)
- Estimates tokens (bytes ÷ 4)
- Greedy bin-packs into balanced slices (~120K tokens each, default cap 8)
- Writes `brief.md` and `slice-N.txt` to the session scratchpad
- Prints plan summary: file count, total tokens, slice distribution, scratchpad path

## Phase 2: Fan out

In a single message, emit N parallel `Agent` calls. Use the
`subagent_type` Phase 1 printed (default `general-purpose`; user override
via `--agent`). Use the `run_in_background` value Phase 1 printed
(default `false`; user opts into background via `--background`).

For each slice N, the agent prompt is (substitute `<scratchpad>` with the
absolute path Phase 1 printed on its `Scratchpad:` line):

> Read `<scratchpad>/brief.md` for your task. Then read
> `<scratchpad>/slice-N.txt` for your file list. Read every file in the
> list IN FULL. Apply the task.

## Phase 3: Synthesize

When all agents return, synthesize the N `tool_result` blocks into the
shape the task asked for.

## Operating defaults

- **Scope**: required — `--type`, `--glob`, or `--all`.
- **Agent type**: `general-purpose` (override `--agent <type>`).
- **Execution mode**: synchronous (override `--background`).
- **Per-agent budget**: ~120K source tokens.
- **Agent count**: `ceil(total_tokens / 120_000)`, capped at 8.
- **Output format**: dictated by the task. The skill imposes none.
