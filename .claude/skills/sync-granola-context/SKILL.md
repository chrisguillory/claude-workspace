---
name: sync-granola-context
description: "Sync Granola meetings to local archive. Fetches meeting list,
  downloads missing notes and transcripts."
user-invocable: true
disable-model-invocation: false
effort: low
allowed-tools:
  - "Bash(scripts/sync-granola-context.py:*)"
---

# Sync Granola Meetings

!`scripts/sync-granola-context.py`

## Instructions

Report the sync results above to the user. The script handles everything —
fetching the meeting list, comparing against local state, downloading missing
meetings, and updating tracking files.

If the script reports errors, note them. Common non-errors:
- "No note available" — meeting has no AI-generated notes yet (normal)
- "No transcript" — meeting wasn't recorded (normal)
