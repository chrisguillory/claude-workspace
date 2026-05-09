---
name: auth-refresh
description: "Refresh browser-session cookies for any tool with an auth recipe. Tries the deterministic recipe first; LLM-driven fallback on failure. Hands captured profile state to the tool's auth-import."
argument-hint: "<recipe-path-or-tool-name> [state-path]"
user-invocable: true
disable-model-invocation: false
effort: max
allowed-tools:
  - "Bash(~/claude-workspace/.claude/skills/auth-refresh/run-recipe.py:*)"
  - "Bash(~/claude-workspace/scripts/gh-upload-auth-recipe.py:*)"
  - "Bash(~/claude-workspace/mcp/grok-kit/auth-recipe.py:*)"
  - "Bash(gh-upload:*)"
  - "Bash(grok-kit:*)"
  - "Bash(selenium-browser:*)"
  - "Bash(op:*)"
  - "Bash(jq:*)"
  - "Bash(mktemp:*)"
  - "Bash(rm:*)"
  - "Bash(diff:*)"
  - Read
  - AskUserQuestion
  - "mcp__selenium-browser__*"
---

# Refresh browser-session cookies via auth recipe

Drive a fresh login for whichever browser-cookie-using tool the recipe targets,
and ingest the captured state via the tool's `auth-import`. Phase 0 runs the
deterministic recipe at skill load. If the recipe fails (UI changed, new
challenge, missing credentials), Phase 1 drives the flow dynamically via
selenium-browser MCP tools. Phase 3 suggests a recipe diff so the recipe
self-improves over time.

The recipe encapsulates everything tool-specific: which 1Password item to fetch
(URL + title + login), which URL to navigate, which selectors to drive, which
`<tool> auth-import` to hand off to. The skill is generic.

## Recipe execution (auto-runs at skill load)

!`~/claude-workspace/.claude/skills/auth-refresh/run-recipe.py $ARGUMENTS`

## Instructions

### Phase 0 — Read the recipe outcome

If the recipe printed `→ handing off to <tool> auth-import (state=…)` and
exited 0, **authentication is done**. Phase 0 already invoked the tool's
`auth-import`; nothing left unless the user wants verification (see "Verifying
the result" below).

If the recipe exited non-zero, proceed to Phase 1. The final stderr lines tell
you what failed:
- `auth-refresh: cannot resolve recipe: <name>` → user passed a recipe path
  or tool name that didn't resolve. Re-run with a path that does, or list
  available recipes (the wrapper does that when invoked with no args).
- `op …` failure → 1Password not signed in, item missing, or biometric prompt
  dismissed. Re-running often resolves; persistent failures need user attention.
- `selenium-browser …` failure → form selectors no longer match the site's UI
  or 2FA flow changed. Recipe needs updating; Phase 1 captures the new flow.
- `AuthRecipeError: no 1Password item with title==… AND login==…` → the
  recipe's `MATCH_TITLE`/`MATCH_LOGIN` constants no longer find the item.
  Likely cause: 1Password item was renamed or its URL/login was changed.
- `AuthRecipeError: saved state has no cookies` → login completed in selenium
  but didn't actually authenticate. Likely: a challenge appeared and the
  recipe didn't detect it.

### Phase 1 — LLM-driven fallback (only if Phase 0 failed)

Drive the login flow dynamically. The recipe is your reference for what the
tool intended; you'll adapt based on what's actually on the page.

1. **Read the recipe** to learn the target URL, the 1Password match keys, and
   the tool's `auth-import` command. Recipe paths are visible in Phase 0
   output; if not, list them with `ls ~/claude-workspace/scripts/*-auth-recipe.py
   ~/claude-workspace/mcp/*/auth-recipe.py`.
2. **Pick a fresh state path**: `STATE_PATH=$(mktemp -t auth-state).json`.
3. **Open a fresh browser** to the target URL the recipe was using:
   `selenium-browser navigate <URL> --fresh-browser --browser chromium`.
4. **Read the page**: `mcp__selenium-browser__get_aria_snapshot('main')`.
5. **Drive form fill** with `mcp__selenium-browser__click`,
   `mcp__selenium-browser__type_text`, `mcp__selenium-browser__press_key`.
   Prefer class-based CSS selectors over synthetic IDs for stability.
6. **Fetch credentials** via `op` using the recipe's MATCH_URL + MATCH_TITLE +
   MATCH_LOGIN constants (read them from the recipe). Bundle reads to minimize
   biometric prompts:
   ```bash
   op item get <ID-from-recipe-resolution> --reveal --format json   # username + password
   op item get <ID-from-recipe-resolution> --otp                    # fresh TOTP
   ```
7. **Handle 2FA branching** based on whichever method the site presents (TOTP,
   SMS, push, hardware key). For TOTP: locate the input, type the OTP.
8. **Handle interactive challenges** (CAPTCHA, device verification,
   account-protection prompts): escalate via `AskUserQuestion` — ask the user
   to complete the challenge in the open browser window and answer "done"
   when ready to continue.
9. **Capture state once the post-login page loads**:
   `mcp__selenium-browser__save_profile_state(filename=STATE_PATH)`.

### Phase 2 — Hand off via the tool's auth-import (Phase 1 path only)

Read the recipe to find the tool name (it's the first argument to
`subprocess.run([…, 'auth-import', …])` near the bottom of `main()`), then
call it:

```bash
<tool> auth-import "$STATE_PATH"
```

Confirm the output shows `Missing load-bearing: none ✓`. If load-bearing
cookies are missing, login did not complete fully — restart Phase 1 with
attention to whichever step the recipe was failing on.

### Phase 3 — Recipe-update suggestion (only if Phase 1 deviated from the recipe)

If Phase 1 used selectors or steps that differ from the recipe, produce a
unified diff against the recipe file. **DO NOT auto-apply** — humans review
and apply via PR.

```diff
--- a/<recipe-path>
+++ b/<recipe-path>
@@ -<line>,<n> +<line>,<n> @@
-        {'tool': 'click', 'params': {'css_selector': '<old>'}},
+        {'tool': 'click', 'params': {'css_selector': '<new>'}},
```

Annotate each change with WHY it was needed (UI version observed, what the
old selector matched against now, etc.) so the user can judge the change.

## Verifying the result

```bash
<tool> auth-status
# Expected: "Missing: none ✓"
```

For a wet smoke test that exercises the freshly-imported cookies:
- `gh-upload`: `gh-upload upload <small-file>` returns a markdown link
- `grok-kit`: `grok-kit list -n 1` returns a conversation

## Why this skill exists

Browser session cookies expire on a clock (12-72h for grok, ~2 weeks for
GitHub). Tools that depend on those cookies need periodic refresh. This skill
is the manual-but-fast path back to a working session: deterministic when site
UIs are stable, LLM-adaptive when they aren't, with the recipe self-improving
over time as deviations surface.