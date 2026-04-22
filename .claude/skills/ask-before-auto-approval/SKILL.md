---
name: ask-before-auto-approval
description: "Toggle the ask-before-auto-approval gate for this session"
argument-hint: on | off | status  (empty = status)
allowed-tools: Bash(mkdir:*), Bash(touch:*), Bash(rm -f:*), Bash(ls:*), Bash(test:*)
disable-model-invocation: true
---

# Ask-Before-Auto-Approval Gate

Toggle the per-session marker that controls `hooks/ask-before-auto-approval.py`.
When the marker is present, the hook short-circuits (returns exit 0) and
auto mode behaves as if the hook weren't installed. When absent, the hook
emits `{"permissionDecision": "ask"}` for tools in `GATED_TOOLS`, which —
combined with the `hook-ask-no-override` binary patch — surfaces a user prompt.

Marker path: `~/.claude-workspace/ask-before-auto-approval/disabled-$CLAUDE_CODE_SESSION_ID`

## Subcommand: $ARGUMENTS

Dispatch based on the value of `$ARGUMENTS`. If it's empty, treat it as `status`.

### off

Touch the per-session marker to silence the hook for this session:

```bash
mkdir -p ~/.claude-workspace/ask-before-auto-approval
touch ~/.claude-workspace/ask-before-auto-approval/disabled-$CLAUDE_CODE_SESSION_ID
```

Confirm in one sentence including the session ID.

### on

Remove the per-session marker so the hook re-engages:

```bash
rm -f ~/.claude-workspace/ask-before-auto-approval/disabled-$CLAUDE_CODE_SESSION_ID
```

Confirm in one sentence including the session ID.

### status  (default when `$ARGUMENTS` is empty)

Report the gate state in exactly three lines:

- `Gate state: enabled` (no marker) or `Gate state: disabled` (marker present)
- `Session: <CLAUDE_CODE_SESSION_ID>`
- `Marker: <full path>  (absent | exists)`

Determine marker presence with:

```bash
test -f ~/.claude-workspace/ask-before-auto-approval/disabled-$CLAUDE_CODE_SESSION_ID && echo exists || echo absent
```

### Unknown argument

Print a short usage hint (list the four valid forms) and do NOT execute any filesystem changes.

## Notes

- Scope is per-session only: the hook check is `(GATE_DIR / f'disabled-{payload.session_id}').exists()`; there is no global marker.
- Two-mechanism composition: the binary patch (`hook-ask-no-override`) guarantees the hook's ask surfaces as a prompt *when the hook speaks*; this marker controls *whether the hook speaks*. Toggling off doesn't subvert the patch — it skips the branch the patch touches entirely (see `Ma_` step [2]).