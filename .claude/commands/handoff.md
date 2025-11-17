---
description: "Generate comprehensive handoff document capturing session context, work completed, pending tasks, and key decisions"
argument-hint: <optional focus areas or specific instructions>
allowed-tools: Read, Write, Bash(git log:*), Bash(git status:*), Bash(git diff:*)
disable-model-invocation: false
---

# Session Handoff Document Generator

Generate a comprehensive, self-contained handoff document that preserves critical session context.

## Instructions: $ARGUMENTS

*If no specific instructions provided, create comprehensive handoff covering all aspects.*

---

## PHASE 1: GATHER CONTEXT

### Session Information
- Current session ID from environment
- Working directory and project context
- Current git branch and status

### Recent Work
- Check git log for recent commits (if any)
- Check git status for uncommitted changes
- Review recent file modifications

### Todo State
- Current todos (if accessible)
- Completed vs pending work
- Priority items identified

---

## PHASE 2: ANALYZE & ORGANIZE

Review the conversation to identify:

### 1. Primary Request & Work Completed
- What was the user asking for?
- What was actually accomplished?
- What approaches were tried?

### 2. Key Technical Concepts
- Patterns established or discovered
- Design decisions made
- Technical insights gained

### 3. User Preferences Revealed
- Working style observations
- Communication patterns noted
- Specific preferences expressed

### 4. Problems & Solutions
- Issues encountered and how they were resolved
- Debugging approaches that worked
- Dead ends to avoid

### 5. Pending Work & Questions
- Unfinished tasks with context
- Open questions needing answers
- Stream-of-consciousness ideas to organize

### 6. Code Changes
- Files modified with purpose
- Key code sections with line numbers
- Patterns implemented

---

## PHASE 3: STRUCTURE THE HANDOFF

Create a HANDOFF.md with this structure:

```markdown
# Session Handoff: [Brief Title]

## Quick Context
- **Session ID:** [current]
- **Date:** [today]
- **Working Directory:** [path]
- **Primary Focus:** [one line summary]

## Executive Summary
[2-3 paragraphs of what happened and current state]

## Completed Work

### [Category 1]
- Specific accomplishment with context
- Key decisions made
- Files affected: `path/to/file.py`

### [Category 2]
...

## Technical Discoveries

### [Pattern/Insight Name]
**What we learned:**
**Why it matters:**
**How to apply:**

## Pending Tasks

### Priority 1: [Task Name]
**Status:** [Current state]
**Context:** [Why this matters]
**Next steps:**
1. [Specific action]
2. [Specific action]

**Questions to resolve:**
- [Open question]

**Related files/code:**
- `file:line` - [what to look at]

### Priority 2: [Task Name]
...

## Stream of Consciousness Captures

### [Topic]
User's thoughts (organized):
- [Key point from rambling]
- [Important tangent]
- [Idea to investigate]

## User Working Preferences Observed
[Any new patterns noticed about how user works]

## Code Context

### Key Changes Made
\`\`\`python
# file_path.py:LINE_NUMBER
[Relevant code snippet]
\`\`\`

### Patterns Established
[Code patterns with examples]

## Environment State

### Git Status
[Current branch, uncommitted changes]

### System State
[Relevant system information]

## Continuation Instructions

**For next session:**
1. Start with [specific task]
2. Review [specific file/section]
3. Consider [design decision]

**Critical context to remember:**
- [Key point that must not be lost]
- [Important decision made]

**User's last explicit request:**
> [Quote of what user actually wanted]
```

---

## PHASE 4: WRITE THE HANDOFF

Based on $ARGUMENTS:
- If specific focus requested, emphasize those areas
- If no arguments, be comprehensive
- Always make it self-contained

Write the handoff to:
1. First choice: `scratch/HANDOFF-[date].md`
2. If scratch doesn't exist: `HANDOFF.md` in project root

Include at the bottom:
```
---
*Generated with /handoff command*
*Instructions: $ARGUMENTS*
```

---

## PHASE 5: CONFIRM COMPLETION

Report to user:
- Handoff document created at: [path]
- Word count: [approximate]
- Key sections included: [list]
- Suggest: "Review and adjust as needed"

---

## QUALITY CHECKLIST

Before saving, verify:
- [ ] Self-contained (no external context needed)
- [ ] Current state is clear
- [ ] Next steps are actionable
- [ ] Stream-of-consciousness organized
- [ ] Technical decisions documented
- [ ] User preferences captured
- [ ] Critical context preserved