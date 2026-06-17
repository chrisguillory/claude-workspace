---
name: recover-session
description: "Recover in-flight context from a compacted, crashed, or context-limited
  session. Indexes transcript for semantic search, reconstructs the session arc."
argument-hint: "[session ID prefix, e.g. 'ea0afc58']"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(~/.claude/skills/recover-session/gather-session-data.py:*)"
---

# Recover Session: Index, Search, Reconstruct

Recover context from a dead, compacted, or context-limited session by indexing its
transcript and session directory for semantic search, then reconstructing the session
arc through targeted queries.

## Session data (auto-gathered and indexed)

!`~/.claude/skills/recover-session/gather-session-data.py $ARGUMENTS`

## Instructions

### Phase 1: Triage forks (from the gathered "Fork analysis")

Read the "Fork analysis" and "Fork-direct user messages" sections above (computed from the
transcript's uuid→parentUuid tree) before reconstructing:

- **STALE FORK SUSPECTED** ⇒ the resumed context is anchored on a dead twig: the live chain
  diverged *after* a larger orphaned branch finished, so the latest real work sits
  off-context in that branch. It's in the indexed transcript — recover it (semantic search,
  or read the JSONL at the printed leaf uuid) and present it as the current state.
- **Orphaned + fork-direct user messages** are instructions a normal resume never replays.
  Fold each into the narrative as primary intent (the content is also in the semantic index).

### Phase 2: Orient from git state

Ground truth of what actually shipped. Run:

```bash
git status
git log --oneline -20
git diff --stat
git stash list
git worktree list
git branch -v
```

Git shows what was ACTUALLY delivered vs what was planned. Cross-reference against
recovered context in later phases.

### Phase 3: Establish the session narrative

The session is a chronology. Things debated early may be superseded later. Present
the arc, not just the endpoint.

- If a compaction summary exists in the current conversation context, use it as the
  chronological backbone — it contains all user messages and key decisions
- If recovering a different session (no compaction summary), use semantic search to
  reconstruct the flow of user messages and topic shifts
- Identify the overall arc: opening goals, exploration, implementation, refinement, tangents

### Phase 4: Targeted semantic recovery

Use `mcp__document-search__search_documents` against the indexed transcript and session
directory. Do NOT "read last N lines" — use semantic search for targeted recovery.

| Goal                      | Search type | Example queries                                       |
|---------------------------|-------------|-------------------------------------------------------|
| Task definitions          | hybrid      | "task", "todo", "need to", "should also"              |
| Architectural decisions   | hybrid      | "decided to", "approach", "architecture", "pattern"   |
| User corrections          | hybrid      | "actually", "no I meant", "instead", "don't do"       |
| Deferred items            | hybrid      | "defer", "later", "skip for now", "not in this PR"    |
| Background worker results | hybrid      | "agent", "worker", "subagent", "background"           |
| PR scope                  | hybrid      | "PR", "branch", "pull request", "scope"               |
| Error recovery            | lexical     | exact error messages, function names                  |

For each finding, note its chronological position (early/mid/late) and whether it was
superseded by later discussion.

Search the session directory specifically for detailed subagent research that was not
fully surfaced to the main conversation.

When search points to a relevant area, THEN drill into the JSONL for more context.

### Phase 5: Check persistent artifacts

- `.claude/plans/` for active plans from the session
- Memory files for anything saved during the session
- Handoff documents, research notes, or scratch files

### Phase 6: Reconstruct and present

Synthesize into a structured summary showing the arc:

1. **Session narrative** — the overall arc, not just where it ended
2. **Branch and PR context** — what branch, what PR, what's the goal
3. **Verified task list** — cross-referenced against git commits
4. **Key decisions and resolutions** — what was debated, where it landed
5. **Current state** — in-progress work, immediate next steps
6. **Deferred items** — explicitly called out for later
7. **User feedback/corrections** — memory candidates if not already saved

Ask user to verify accuracy before resuming work.

### Phase 7: Surface discoveries (self-improving)

During recovery, reflect on the process itself:

- Did any search patterns work particularly well for finding specific context?
- Were there types of context that were consistently hard to recover?
- Are there novel heuristics worth generalizing? (e.g., "the first message after each
  compaction summary reveals the user's priority shift")
- Are there new search terms that should be added to the strategy table?

Use `AskUserQuestion` to propose folding generalizable learnings back into this skill.
One discovery per question, concrete proposal for where it would go.

## Key principles

- **Git is ground truth** — the transcript shows intent, git shows what shipped. When
  they conflict, git wins.
- **Sessions are chronologies** — early debates may be superseded. Present the arc
  (debated, resolved, current) not just the endpoint.
- **Semantic search, not tail reading** — use targeted queries, not "read last N lines."
  Drill into the JSONL only when search points to something specific.
- **Verify before asserting** — recovered context may be stale. Present findings as
  "recovered context" rather than established fact.
- **The session directory contains subagent research** — `subagents/` and agent JSONL
  files have detailed findings not fully surfaced to the main conversation.

## Ongoing use

After initial recovery, the indexed transcript remains searchable for the rest of the
session. Use `mcp__document-search__search_documents` anytime you need to reference
prior context while working.
