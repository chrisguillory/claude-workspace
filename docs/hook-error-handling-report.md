# Hook Error Handling in Claude Code: Research Report

## Executive Summary

When PostToolUse hooks fail with exit code 1 (non-blocking error), Claude Code logs the full error to the debug file and shows a brief "hook error" line to the user, but **returns an empty array `[]` to the API** -- meaning the AI assistant sees absolutely nothing. This is by design: non-blocking errors are treated as UI-only events. The fix is straightforward: hooks that need to surface errors to the assistant should use exit code 2 (blocking) or exit code 0 with JSON `additionalContext`.

---

## 1. Exact Hook Error Handling Flow in Claude Code

### Source Files Analyzed

- `/src/utils/hooks.ts` -- core hook execution engine (`executeHooks` generator)
- `/src/services/tools/toolHooks.ts` -- PostToolUse hook orchestration
- `/src/utils/messages.ts` -- `normalizeAttachmentForAPI()` converts messages to what the model sees
- `/src/components/messages/AttachmentMessage.tsx` -- renders messages for the user
- `/src/utils/hooks/hookEvents.ts` -- debug log emission
- `/src/types/hooks.ts` -- Zod schemas and type definitions

### Hook Execution Flow

```
1. Tool completes (e.g., Edit, Write)
2. toolHooks.ts calls executePostToolHooks()
3. hooks.ts executeHooks() matches hooks by tool name
4. For each matched hook:
   a. Spawn child process with hook input on stdin (JSON)
   b. Wait for exit
   c. Based on exit code, yield different result types
5. toolHooks.ts receives results, creates attachment messages
6. messages.ts normalizeAttachmentForAPI() converts to API messages
```

### Exit Code Handling (hooks.ts lines 2617-2697)

| Exit Code | Result Type | Outcome |
|-----------|-------------|---------|
| 0 | `hook_success` attachment (content = stdout) | `'success'` |
| 2 | `blockingError` object (stderr content) | `'blocking'` |
| 1 (or any other non-zero) | `hook_non_blocking_error` attachment (stderr content) | `'non_blocking_error'` |

### What Each Attachment Type Becomes in normalizeAttachmentForAPI()

This is the critical function (messages.ts ~line 4090). It converts attachment messages into API messages that the model actually sees:

| Attachment Type | API Messages (what model sees) |
|---|---|
| `hook_blocking_error` | **System reminder with full error text** -- `"${hookName} hook blocking error from command: ..."` |
| `hook_success` | **Empty `[]`** for PostToolUse (only SessionStart and UserPromptSubmit produce model-visible output) |
| `hook_non_blocking_error` | **Empty `[]`** -- completely invisible to the model |
| `hook_error_during_execution` | **Empty `[]`** -- completely invisible to the model |
| `hook_cancelled` | **Empty `[]`** -- completely invisible to the model |
| `hook_additional_context` | **System reminder** -- `"${hookName} hook additional context: ..."` |
| `hook_stopped_continuation` | **System reminder** -- `"${hookName} hook stopped continuation: ..."` |

The relevant code (messages.ts lines 4252-4261):
```typescript
case 'already_read_file':
case 'command_permissions':
case 'edited_image_file':
case 'hook_cancelled':
case 'hook_error_during_execution':
case 'hook_non_blocking_error':    // <-- EXIT CODE 1 GOES HERE
case 'hook_system_message':
case 'structured_output':
case 'hook_permission_decision':
  return []                        // <-- EMPTY: model sees NOTHING
```

---

## 2. What the User Sees vs What the Assistant Sees

### Exit Code 0 (Success)

| Audience | What They See |
|----------|---------------|
| **User** | Nothing (hook_success renders as `null` in AttachmentMessage.tsx) |
| **Model** | Nothing for PostToolUse (returns `[]`). Only SessionStart/UserPromptSubmit hooks surface stdout content |
| **Debug log** | Full stdout/stderr logged via `emitHookResponse` -> `logForDebugging` |

### Exit Code 1 (Non-blocking Error)

| Audience | What They See |
|----------|---------------|
| **User** | Red error line: `"PostToolUse:Edit hook error"` (AttachmentMessage.tsx line 303) |
| **Model** | **NOTHING** -- `hook_non_blocking_error` returns `[]` in normalizeAttachmentForAPI |
| **Debug log** | Full stderr/traceback: `"Hook PostToolUse:Edit (PostToolUse) error:\n<traceback>"` |

### Exit Code 2 (Blocking Error)

| Audience | What They See |
|----------|---------------|
| **User** | Red error line: `"PostToolUse:Edit hook returned blocking error"` + stderr content |
| **Model** | **Full error as system reminder**: `"PostToolUse:Edit hook blocking error from command: 'hooks/run-linter.py ...': [command]: <stderr>"` |
| **Debug log** | Full stderr/output logged |

### Exit Code 0 + JSON with additionalContext

| Audience | What They See |
|----------|---------------|
| **User** | Nothing visible (unless `systemMessage` is set) |
| **Model** | **System reminder**: `"PostToolUse:Edit hook additional context: <text>"` |
| **Debug log** | Full output logged |

---

## 3. Ways to Surface Information to the Assistant

There are exactly three paths that produce model-visible output for PostToolUse hooks:

### Path 1: Exit code 2 (Blocking Error)
```bash
echo "Violation found: ..." >&2
exit 2
```
- Model sees: `"PostToolUse:Edit hook blocking error from command: ..."`
- Pros: Simple, works with plain text hooks
- Cons: Semantically means "blocking" even though PostToolUse cannot actually block (tool already ran). The docs confirm: "PostToolUse: No (already ran) -- Shows stderr to Claude"

### Path 2: Exit code 0 + JSON additionalContext
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "Linter found violations: ..."
  }
}
```
- Model sees: `"PostToolUse:Edit hook additional context: ..."`
- Pros: Semantically correct, non-blocking, provides context
- Cons: Requires JSON output, more complex hook implementation

### Path 3: Exit code 0 + JSON decision: "block"
```json
{
  "decision": "block",
  "reason": "Linting errors found: ..."
}
```
- Model sees: `"PostToolUse:Edit hook blocking error from command: ..."`
- Pros: Same visibility as exit code 2 but with structured control
- Cons: Same semantic issue as exit code 2

**No other mechanism exists.** Exit code 1 is explicitly designed to be invisible to the model.

---

## 4. Analysis of the Current run-linter.py Behavior

### Current Design
The `run-linter.py` hook (at `/Users/chris/claude-workspace/hooks/run-linter.py`) correctly uses exit code 2 when linter violations are found:

```python
if result.returncode != 0:
    sys.stderr.buffer.write(result.stdout)
    return 2  # Blocking error -- stderr shown to Claude
return 0
```

### The Schema Mismatch Problem
The hook uses `PostToolUseHookInput.model_validate_json()` from `cc_lib.schemas.hooks` with `extra='forbid'` semantics (ClosedModel). When Claude Code adds new fields to the hook input (like `agent_type`), the Pydantic model rejects them:

```
pydantic.ValidationError: 1 validation error for PostToolUseHookInput
agent_type
  Extra inputs are not permitted [type=extra_forbidden, ...]
```

This causes the Python process to crash with a traceback on stderr and exit code 1. The exit code 1 means:
1. User sees: "PostToolUse:Edit hook error"
2. Model sees: nothing
3. All four linters silently stop working
4. The assistant has no idea and keeps making edits without linting

### Why Exit Code 1 Is Wrong Here
This is not a "non-critical warning." The entire linting pipeline is broken. The hook should either:
- Fix the schema to accept new fields (use `StrictModel` with `extra='allow'`)
- Surface the error to the model so it can report it

---

## 5. Recommendations

### Immediate Fix: Change PostToolUseHookInput to StrictModel

The `PostToolUseHookInput` in `cc_lib/schemas/hooks.py` should use `StrictModel` (which defaults to `extra='allow'` per CLAUDE.md) instead of `ClosedModel`. Claude Code's hook input schema evolves independently -- new fields like `agent_type` are added without notice. This is exactly the use case `StrictModel` was designed for:

> **StrictModel**: `'allow'` default, `'forbid'` via env var. Protocol data (Claude Code hooks, MCP) -- forward-compatible.

### Structural Fix: Change run-linter.py Error Handling

The hook should catch Pydantic validation errors and provide a useful error to the model:

```python
try:
    payload = PostToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
except ValidationError as e:
    # Schema mismatch -- surface to the model via additionalContext
    import json
    json.dump({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"Hook schema error in run-linter.py: {e}"
        }
    }, sys.stdout)
    return 0  # Exit 0 so JSON is parsed
```

Or more simply, just use exit code 2:
```python
except ValidationError as e:
    print(f"Hook schema error: {e}", file=sys.stderr)
    return 2  # Blocking -- model sees the error
```

### Documentation: Add Hook Debugging Section to CLAUDE.md

Document the exit code visibility matrix and the debugging workflow:

```markdown
## Hook Debugging

### Exit Code Visibility Matrix
| Exit Code | User Sees | Model Sees | Debug Log |
|-----------|-----------|------------|-----------|
| 0 | Nothing | Nothing (PostToolUse) | Full output |
| 1 | "hook error" (red) | **NOTHING** | Full stderr |
| 2 | "blocking error" + stderr | **Full error as system reminder** | Full output |

### Debugging Workflow
1. Check `~/.claude/debug/{session_id}.txt` for hook errors
2. Search for `Hook PostToolUse:` to find hook execution results
3. Errors from exit code 1 are only visible in this debug log
4. Press Ctrl+O in Claude Code for verbose mode

### Schema Evolution
Hook inputs add new fields as Claude Code evolves. Use `StrictModel`
(extra='allow') for all hook input models to stay forward-compatible.
```

### Whether to Use Exit Code 2 for Schema Errors

**Yes.** For PostToolUse hooks specifically, exit code 2 is the correct choice for schema errors because:

1. PostToolUse "blocking" does not actually block anything (the tool already ran)
2. It is the only simple mechanism to surface errors to the model
3. The alternative (JSON additionalContext) requires more complex error handling
4. The model needs to know the linting pipeline is broken

For PreToolUse hooks, exit code 2 would actually block the tool call, so JSON additionalContext on exit 0 would be more appropriate for schema errors there.

### Separate Debug Log Reader

Not recommended. The information is already available via:
- `Ctrl+O` verbose mode
- `~/.claude/debug/{session_id}.txt`
- The `transcript_path` in the hook input

A dedicated MCP tool for reading debug logs would add complexity without solving the root cause: hooks that should surface errors to the model need to use exit code 2 or JSON additionalContext.

---

## 6. Summary of Key Source Code References

| File | Key Lines | What It Does |
|------|-----------|--------------|
| `src/utils/hooks.ts:2646-2697` | Exit code branching | Routes exit codes to different result types |
| `src/utils/hooks.ts:2682-2696` | Non-blocking error path | Creates `hook_non_blocking_error` attachment |
| `src/utils/messages.ts:4252-4261` | API normalization | Returns `[]` for non-blocking errors (model sees nothing) |
| `src/utils/messages.ts:4090-4098` | Blocking error path | Creates system reminder with full error (model sees it) |
| `src/utils/messages.ts:4117-4128` | Additional context path | Creates system reminder with context (model sees it) |
| `src/components/messages/AttachmentMessage.tsx:296-303` | User rendering | Shows "hook error" line to user |
| `src/utils/hooks/hookEvents.ts:163-169` | Debug logging | Logs full output to debug file |
| `src/services/tools/toolHooks.ts:56-187` | PostToolUse orchestration | Passes hook results as attachment messages |
