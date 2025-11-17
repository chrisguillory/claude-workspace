---
description: "Updates CLAUDE.md using multi-phase workflow: analyze current state, draft updates (Sonnet), invoke claude-md-architect agent (Opus) for complete refactoring, compare both versions, synthesize final result"
argument-hint: <patterns/principles/changes to incorporate>
allowed-tools: Read, Task, AskUserQuestion, Bash(git branch:*), Bash(git log:*), Bash(git status:*)
disable-model-invocation: false
---

# CLAUDE.md Documentation Update Orchestrator

## GIT CONTEXT

**Current branch:** !`git branch --show-current`
**CLAUDE.md status:** !`git status --short CLAUDE.md || echo "CLAUDE.md clean"`
**Last CLAUDE.md update:** !`git log -1 --format="%h %s (%cr)" CLAUDE.md 2>/dev/null || echo "No commit history"`

This helps ensure we're:
- On the right branch for documentation updates
- Not overwriting uncommitted changes
- Aware of recent updates to avoid duplication

---

## INPUT VALIDATION

User request: $ARGUMENTS

**Verify:** Is the request specific and actionable?
- What patterns/principles to incorporate?
- Any removals or major restructuring needed?
- Any specific examples or sections to update?

If the request is vague or unclear, ask the user for specific details before proceeding.

---

## EXECUTION LOG

Update this checklist as you progress:

- [ ] Phase 1: Read and analyze current CLAUDE.md
- [ ] Phase 2: Draft updates (Sonnet perspective)
- [ ] Phase 3: Invoke claude-md-architect agent (Opus)
- [ ] Phase 4: Compare both versions
- [ ] Phase 5: Synthesize and present final version

---

## PHASE 1: ANALYZE CURRENT STATE

Read the current documentation: @CLAUDE.md

**Identify:**
1. Current structure and organization (sections, headers, flow)
2. Existing patterns and principles documented
3. What works well (preserve these strengths)
4. Gaps or areas that could be improved
5. Where new information from $ARGUMENTS should fit naturally
6. Any redundancies or outdated content

**Checkpoint:** Confirm you understand the current state and have a clear mental model of the document's structure before proceeding.

---

## PHASE 2: DRAFT UPDATES (Sonnet Perspective)

Based on $ARGUMENTS, draft YOUR version of how to incorporate the new information into CLAUDE.md.

**Consider:**
- Where does new information fit most naturally in existing structure?
- What sections need adjustment to accommodate new patterns?
- Should any sections be reorganized or restructured?
- What examples should be added (good vs bad patterns)?
- How to maintain terminology consistency?
- What should be removed or deprecated?

**Create:** Your proposed complete CLAUDE.md with updates integrated.

**Important:** Do NOT show this draft yet - hold it for comparison in Phase 4.

**Quality criteria for your draft:**
- Maintains existing document voice and style
- New information integrates naturally, not bolted on
- Examples are clear and concrete
- Principles are terse but complete
- No redundancies

---

## PHASE 3: INVOKE CLAUDE-MD-ARCHITECT AGENT (Opus Refactoring)

Use the Task tool to invoke the `claude-md-architect` agent with a comprehensive prompt.

**Provide to the agent:**
1. The current CLAUDE.md content (full text)
2. The new information to incorporate (from $ARGUMENTS)
3. Any context about:
   - Why these patterns emerged
   - What problems they solve
   - How they relate to existing principles
4. Instruction to generate a COMPLETE new CLAUDE.md that holistically integrates everything

**Example Task invocation:**
```
Use Task tool with:
- subagent_type: "claude-md-architect"
- prompt: "Here is the current CLAUDE.md: [content]. Incorporate these new patterns: [from $ARGUMENTS]. Context: [why they matter]. Generate a complete new CLAUDE.md that holistically integrates old + new with optimal structure."
```

**Expected output:** The agent will return a complete refactored CLAUDE.md.

**Error handling:** If agent invocation fails, note the error and continue with just your draft from Phase 2.

**Important:** Do NOT show the agent's version yet - hold it for comparison in Phase 4.

---

## PHASE 4: COMPARATIVE ANALYSIS

Now compare YOUR draft (Phase 2) vs the AGENT's version (Phase 3).

**Analyze across these dimensions:**

1. **Structure & Organization**
   - Which version has clearer sections?
   - Better information hierarchy?
   - More logical flow?

2. **Integration Quality**
   - Where does new information fit more naturally?
   - Which approach better preserves existing strengths?
   - Any awkward transitions or forced additions?

3. **Completeness**
   - Did one version miss important patterns?
   - Are examples sufficient and clear?
   - Any gaps in either version?

4. **Clarity & Coherence**
   - Which version is more readable?
   - Better maintains consistent voice?
   - Clearer principles and patterns?

5. **Examples & Specifics**
   - Which version has better code examples?
   - More concrete guidance?
   - Clearer good vs bad patterns?

**Document:** Create a brief analysis noting specific strengths of each version. This helps justify your synthesis decisions.

**Show to user:** Brief summary of both approaches:
- "Sonnet draft: [key characteristics]"
- "Opus refactoring: [key characteristics]"
- "Comparative insights: [what each does well]"

---

## PHASE 5: SYNTHESIS & PRESENTATION

Create the FINAL version by combining the best elements from both drafts.

**Synthesis guidelines:**
- Use the version with clearer overall structure
- Take better examples from either version
- Choose clearer phrasing regardless of source
- Preserve successful existing patterns
- Ensure consistent terminology throughout
- Maintain the document's established voice

**Decision criteria when versions conflict:**
1. Prioritize clarity and maintainability
2. Keep patterns that have proven successful
3. Integrate new patterns only where they add clear value
4. Prefer explicit over implicit
5. Follow "terse but complete" principle

**Present to user:**

1. **Summary of changes:**
   - What was added
   - What was restructured
   - What was removed (if anything)
   - Key improvements

2. **The complete new CLAUDE.md:**
   - Show in a code block for easy copying
   - Formatted and ready to use

3. **Synthesis note:**
   - "Combined Sonnet draft + Opus refactoring"
   - Brief note on which elements came from where
   - Any decisions made when approaches differed

4. **Next steps:**
   - User can review the diff manually
   - User can apply changes if satisfied
   - User can request adjustments

**Important:** Do NOT write the file directly. Present the content for user review and approval.

---

## QUALITY CHECKPOINTS

Before presenting the final version, verify:

- [ ] All information from $ARGUMENTS is incorporated
- [ ] Document structure is logical and coherent
- [ ] No redundancies or contradictions
- [ ] Examples are clear and helpful
- [ ] Terminology is consistent
- [ ] Voice matches existing CLAUDE.md style
- [ ] "Terse but complete" principle followed
- [ ] Ready for user review

If any checkpoint fails, revise before presenting.

---

## EXECUTION NOTES

- Be transparent about each phase
- Show your reasoning for synthesis decisions
- If agent invocation fails, gracefully fall back to Sonnet draft
- Let user interrupt at any phase if they want to redirect
- Trust the signal: if information is worth incorporating, integrate it naturally