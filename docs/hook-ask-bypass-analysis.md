---
title: "Hook-Ask Bypass in Claude Code Auto Mode"
date: 2026-04-22
version_studied: "Claude Code 2.1.116"
status: "Analysis complete — patch designs ready for decision"
---

# Hook-Ask Bypass in Claude Code Auto Mode

> [!NOTE]
> **What this document is.** A full analysis of how Claude Code's auto-mode LLM classifier silently overrides user-hook `permissionDecision: "ask"` emissions under specific conditions, a decompilation walkthrough of the responsible code, an upstream-issue landscape survey, and 3–4 candidate binary-patch designs at varying levels of precision.

[toc]

---

## 1. TL;DR

| | |
|---|---|
| **What happens** | Your `ask-before-auto-approval.py` PreToolUse hook returns `{"permissionDecision":"ask"}` for `Edit .claude/settings.json` under auto mode. Claude Code silently discards the decision, routes to the LLM classifier, the classifier returns "allow", the Edit runs without a prompt. |
| **Why** | Function `Ma_` (outer permission resolver) drops the hook result `H` when built-in safety check `FJH` *also* returns `"ask"`. The fallback re-entry in auto mode uses the classifier, not a prompt. |
| **Affected paths** | `.claude/settings.json`, `.claude/settings.local.json`, all 5 managed settings categories, `.claude/commands/*`, `.claude/agents/*`, `.claude/skills/*`, and an internal "sensitive files" list. |
| **Our hook's practical reach** | Only fires for `{Edit, Write, mcp__…}` in auto mode. Outside those, no hook ask exists and the bypass can't apply. |
| **Classification verdict** | **`FIX`** (high confidence). Anthropic has already fixed the symmetric `deny` case (#39344 → v2.1.101); this is the same pattern in the `ask` dimension. |
| **Recommended patch** | **Option A**: 1-byte flip (`behavior==="ask"` → `behavior==="xsk"`) inside the FJH-ask branch of `Ma_`. Same-length, stable from v2.1.109, narrow effect. Optionally pair with Option C (narrowed) for defense-in-depth. |

---

## 2. How the Bypass Works — Code Walkthrough

### 2.1 High-level decision flow

```
                    Edit <any file>
                          │
                          ▼
       ┌───────────────────────────────────┐
       │ Ja_ — run PreToolUse hooks         │
       │ Your hook returns "ask"            │
       └───────────────────────────────────┘
                          │
                          ▼ (H = hook result, Y = H.behavior = "ask")
       ┌───────────────────────────────────┐
       │ Ma_ — resolve permission decision  │
       └───────────────────────────────────┘
                          │
                 ┌────────┴────────┐
       H.behavior=="deny"      else
            │                    │
            ▼                    ▼
          DENY          ┌──────────────────┐
                        │ D = await FJH(…) │   ← static rules + tool safetyCheck
                        └──────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
      D.behavior=="deny"  D.behavior=="ask"   D is null / passthrough
              │                 │                 │
              ▼                 ▼                 ▼
            DENY        ┌───────────────┐   Y=="allow"? ──→ ALLOW (hook wins)
                        │ Drop H        │
                        │ await O(…)    │   else ──→ await O(…, H) ──→ PROMPT
                        │ (no H passed) │
                        └───────────────┘
                                │
                          Auto mode?
                        ┌──┴──┐
                    yes     no
                     │       │
                     ▼       ▼
                CLASSIFIER  PROMPT
                (LLM call)
                     │
              ┌──────┴──────┐
           allow          deny
             │              │
             ▼              ▼
         RUN TOOL        DENY
         (silent)
```

> [!IMPORTANT]
> **The structural asymmetry**: `await O(…, H)` (with hook result as 6th arg) is the documented "honor the hook's ask" path — every `canUseTool` wrapper has a `$ ?? await JM(...)` short-circuit for exactly this shape. `await O(…)` (5-arg, no H) re-enters the full pipeline, which in auto mode substitutes the classifier for a prompt.

### 2.2 `Ma_` — outer permission resolver

<details>
<summary><strong>Click to expand: <code>Ma_</code> decompiled + raw bytes</strong></summary>

**Pseudocode (de-minified, v2.1.116):**

```js
async function Ma_(hookResult, tool, input, ctx, fallbackDecide, toolUseId, asstMsg) {
  const requiresUserInteraction = tool.requiresUserInteraction?.();
  const requireCanUseTool = ctx.requireCanUseTool;

  // [1] Hook deny wins unconditionally — short-circuits before FJH runs
  if (hookResult?.behavior === "deny") {
    log(`Hook denied tool use for ${tool.name}`);
    return { decision: hookResult, input };
  }

  // [2] Hook didn't opine → full pipeline, no hook context
  if (hookResult?.behavior !== "allow" && hookResult?.behavior !== "ask") {
    return { decision: await fallbackDecide(tool, input, ctx, toolUseId, asstMsg), input };
  }

  const Y = hookResult.behavior;                            // "allow" | "ask"
  const w = hookResult.updatedInput ?? input;
  const j = requiresUserInteraction && hookResult.updatedInput !== undefined;

  // [3] Hook=allow but canUseTool required → fallback without hook
  if (Y === "allow" && (requiresUserInteraction && !j || requireCanUseTool)) {
    return { decision: await fallbackDecide(tool, w, ctx, toolUseId, asstMsg), input: w };
  }

  // [4] Consult built-in rules + safetyCheck
  const D = await FJH(tool, w, ctx);

  if (D?.behavior === "deny") {
    log(`Hook returned '${Y}' for ${tool.name}, but deny rule overrides`);
    return { decision: D, input: w };
  }

  // [5] ★ THE BYPASS ★
  //     When FJH also says "ask", we ABANDON the hook result and
  //     re-enter the full pipeline WITHOUT hook context.
  //     In auto mode the full pipeline calls the LLM classifier.
  if (D?.behavior === "ask") {
    log(`Hook returned '${Y}' for ${tool.name}, but ask rule/safety check requires full permission pipeline`);
    return { decision: await fallbackDecide(tool, w, ctx, toolUseId, asstMsg), input: w };
  }

  // [6] Hook=allow, FJH clear → honor hook
  if (Y === "allow") {
    log(`Hook approved tool use for ${tool.name}, bypassing permission prompt`);
    return { decision: hookResult, input: w };
  }

  // [7] Hook=ask, FJH clear → full pipeline WITH hook context (produces prompt)
  return { decision: await fallbackDecide(tool, w, ctx, toolUseId, asstMsg, hookResult), input: w };
}
```

**Minified source (exact bytes, line ~3517 of unpacked JS):**

```js
async function Ma_(H,_,q,K,O,T,$){let A=_.requiresUserInteraction?.(),z=K.requireCanUseTool;if(H?.behavior==="deny")return y(`Hook denied tool use for ${_.name}`),{decision:H,input:q};if(H?.behavior!=="allow"&&H?.behavior!=="ask")return{decision:await O(_,q,K,T,$),input:q};let Y=H.behavior,w=H.updatedInput??q,j=A&&H.updatedInput!==void 0;if(Y==="allow"&&(A&&!j||z))return y(`Hook approved tool use for ${_.name}, but canUseTool is required`),{decision:await O(_,w,K,T,$),input:w};let D=await FJH(_,w,K);if(D?.behavior==="deny")return y(`Hook returned '${Y}' for ${_.name}, but deny rule overrides: ${D.message}`),{decision:D,input:w};if(D?.behavior==="ask")return y(`Hook returned '${Y}' for ${_.name}, but ask rule/safety check requires full permission pipeline`),{decision:await O(_,w,K,T,$),input:w};if(Y==="allow")return y(j?`Hook satisfied user interaction for ${_.name} via updatedInput`:`Hook approved tool use for ${_.name}, bypassing permission prompt`),{decision:H,input:w};return{decision:await O(_,w,K,T,$,H),input:w}}
```

**Byte sequence of target line (169 bytes):**
```
if(D?.behavior==="ask")return y(`Hook returned '${Y}' for ${_.name}, but ask rule/safety check requires full permission pipeline`),{decision:await O(_,w,K,T,$),input:w};
```

Packed binary offsets: Mach-O at 79992897, `__BUN` duplicate at 194599769.

</details>

### 2.3 `FJH` — static rules + tool safety check

<details>
<summary><strong>Click to expand: <code>FJH</code> decompiled</strong></summary>

```js
async function FJH(tool, input, ctx) {
  const state = ctx.getAppState();

  // Deny rules take absolute priority
  const denyMatch = Hs_(state.toolPermissionContext, tool);
  if (denyMatch) return {
    behavior: "deny",
    decisionReason: { type: "rule", rule: denyMatch },
    message: `Permission to use ${tool.name} has been denied.`
  };

  // Ask rules (with sandboxing carve-out for Bash)
  const askMatch = rG7(state.toolPermissionContext, tool);
  if (askMatch) {
    if (!(tool.name === BASH && sandboxingEnabled && autoAllowBashIfSandboxed && hL(input))) {
      return {
        behavior: "ask",
        decisionReason: { type: "rule", rule: askMatch },
        message: KO(tool.name)
      };
    }
  }

  // Tool's own per-input checkPermissions — this is the settings.json path
  let $ = { behavior: "passthrough", message: KO(tool.name) };
  try {
    const parsed = tool.inputSchema.parse(input);
    $ = await tool.checkPermissions(parsed, ctx);
  } catch (err) { /* swallow non-MO/z3 errors */ }

  if ($?.behavior === "deny") return $;
  if ($?.behavior === "ask" && i$8($.decisionReason)) return $;    // rule-based ask
  if ($?.behavior === "ask" && (SPH($.decisionReason) || $.decisionReason?.type === "sandboxOverride"))
    return $;                                                       // safety-check-based ask ← settings.json path
  return null;    // allow/passthrough
}
```

</details>

### 2.4 `Ae_`, `TB5`, `deH` — the path matcher + safetyCheck emission

<details>
<summary><strong>Click to expand: path matching + <code>classifierApprovable</code> emission</strong></summary>

**Path resolution** (`Ae_` at offset 11086511):

```js
function Ae_(path) {
  let resolved = CJ(vq(path));
  if (resolved.endsWith('/.claude/settings.json') ||
      resolved.endsWith('/.claude/settings.local.json'))
    return true;
  return OB5().some(p => CJ(p) === resolved);  // all 5 managed settings locations
}
```

`OB5()` expands `["userSettings","projectSettings","localSettings","flagSettings","policySettings"]` into absolute paths. So `Ae_` matches every Claude-managed `settings.json`.

`TB5` is a strict superset — also matches `.claude/commands`, `.claude/agents`, `.claude/skills`.

**Safety check emission** (`deH` at offset 11088934):

```js
function deH(path, resolvedPaths, _, isRemoteMode) {
  // suspicious-windows-path check: classifierApprovable: false (safety-critical)
  for (const $ of paths) if (fn7($))
    return { safe: false, message: "...suspicious Windows path...", classifierApprovable: false };

  for (const $ of paths) {
    if (isRemoteMode) {
      if (Ae_($)) return { safe: false,
        message: `Claude requested permissions to write to ${path}, but you haven't granted it yet.`,
        classifierApprovable: true };                   // ← site A (SDK remote)
    } else {
      if (TB5($)) return { safe: false,
        message: `Claude requested permissions to write to ${path}, but you haven't granted it yet.`,
        classifierApprovable: true };                   // ← site B (CLI — this is us)
    }
  }
  for (const $ of paths) {
    if (jB5($, isRemoteMode)) return { safe: false,
      message: `Claude requested permissions to edit ${path} which is a sensitive file.`,
      classifierApprovable: true };                     // ← site C (sensitive-file list)
  }
  return { safe: true };
}
```

**Flow into `Ma_`:**

```
Edit.checkPermissions (hJH, offset 11095424)
   └─ deH(path, ...) → { safe:false, classifierApprovable: true }
   └─ returns { behavior:"ask", decisionReason:{ type:"safetyCheck", ..., classifierApprovable: true } }
FJH (filters via SPH) → returns the ask unchanged
Ma_ → sees D.behavior==="ask" → FIRES BYPASS BRANCH
   └─ await O(_,w,K,T,$)  ← 5-arg form, no hook context
JM → mode=="auto" → SPH(..., J=>!J.classifierApprovable) finds nothing
   (because J.classifierApprovable === true for settings.json)
   └─ calls the LLM classifier (CK_)
classifier returns "allow"  → Edit runs silently
```

**Why `protocol.py` doesn't hit this path:** `Ae_`/`TB5`/`jB5` don't match it. `deH` returns `{safe:true}`. `Edit.checkPermissions` returns `{behavior:"passthrough"}`. `FJH` returns `null`. In `Ma_`, `D?.behavior === "ask"` is false → falls through to the final `await O(_,w,K,T,$,H)` (6-arg) → adapter routes to user prompt.

</details>

### 2.5 `JM` — fallback decider (the `canUseTool` bound to `O`)

<details>
<summary><strong>Click to expand: <code>JM</code> decompiled — how the classifier silently overrides</strong></summary>

```js
JM = async (tool, input, ctx, asstMsg, classifierApprovals) => {
  const T = await EM5(tool, input, ctx);    // like FJH + mode checks

  if (T.behavior === "allow") { /* ... */ return T; }

  if (T.behavior === "ask") {
    const state = ctx.getAppState();
    if (state.toolPermissionContext.mode === "dontAsk") return denyResult;

    if (mode === "auto" || (mode === "plan" && isAutoModeActive())) {
      // Check if safety check is NOT classifier-approvable (would keep ask)
      const nonApprovable = SPH(T.decisionReason, J => !J.classifierApprovable);
      const isSandboxOverride = T.decisionReason?.type === "sandboxOverride";
      if (nonApprovable || isSandboxOverride) {
        if (shouldAvoidPermissionPrompts) return denyResult;
        if (nonApprovable) return T;       // ← keep ask, skip classifier
      }
      // ... acceptEdits fast-path check, auto-mode allowlist check ...

      const j = await CK_(...);             // ★ THE CLASSIFIER (LLM call) ★
      if (j.shouldBlock) return denyResult;
      return allowResult;                   // ← silent allow
    }
    // ... non-auto mode: surface prompt ...
  }
  return T;
};
```

**Evidence `JM` is the `O`:** `canUseTool: JM` at `11120126`, flowing through `Ly → fO5 → new I1_(…, T, …) → Dq_ → x35 → m35 → Ma_(…,O,T,$)`. Every `canUseTool` wrapper (GO5/createCanUseTool/FzK/UzK variants) accepts 6 args and has a `$ ?? await JM(...)` short-circuit — **passing `H` as the 6th arg skips JM entirely and routes directly to the user prompt via `zKH()` (CLI) or SDK `can_use_tool` request.**

</details>

### 2.6 Debug-log evidence from this session

<details>
<summary><strong>Live trace: settings.json (bypass) vs protocol.py (hook honored)</strong></summary>

**Settings.json Edit (bypass triggered):**
```
08:02:45.186  Hook PreToolUse:Edit returned permissionDecision: ask
08:02:45.198  Hook returned 'ask' for Edit, but ask rule/safety check requires full permission pipeline
08:02:45.200  [auto-mode] new action being classified: Edit /Users/chris/.../settings.json
08:02:50.204  Slow permission decision: 5006ms for Edit (mode=auto, behavior=allow)
```

**Protocol.py Edit (hook ask honored):**
```
08:11:xx  Hook PreToolUse:Edit returned permissionDecision: ask
08:11:xx  Hook result has permissionBehavior=ask
          executePermissionRequestHooks called for tool: Edit
08:11:xx  (user sees prompt, declines)
08:11:xx  Edit tool permission denied
```

The `"but ask rule/safety check requires full permission pipeline"` line is the *unique diagnostic* for the bypass.

</details>

---

## 3. Is This a Bug?

### 3.1 Upstream issue landscape

> [!IMPORTANT]
> **Zero Anthropic collaborator comments on any of the 11 directly-relevant open issues.** The only staff comments on this broader area are two brief close-outs — and they establish precedent that hook-`ask` semantic violations are worth fixing (see §3.4).

**Canonical parent & directly-on-point issues:**

| # | Title | State | Relevance | Staff Response |
|---|---|---|---|---|
| [#42797](https://github.com/anthropics/claude-code/issues/42797) | Auto-mode ignores `permissions.ask` | 🔴 OPEN (19d) | Canonical parent — *static* `permissions.ask` bypassed by auto-mode classifier. Our user `chrisguillory` commented noting the hook polyfill workaround — we've now shown that workaround *itself* is bypassed when combined with `safetyCheck`. | None. Labeled `bug`, `area:permissions`. |
| [#51255](https://github.com/anthropics/claude-code/issues/51255) | PreToolUse hook `ask` auto-approved in auto mode — no way to force prompt | 🔴 OPEN (2d) | **Most directly on point, filed 1 day before this research.** Exact repro for `Bash(git commit:*)`. Quotes author: *"there is no permissionDecision value that means 'always prompt, regardless of mode'."* | None. Labeled `bug`, `has repro`. |
| [#51676](https://github.com/anthropics/claude-code/issues/51676) | Auto-mode decider denies remediation retries after hook denial | 🔴 OPEN (1d) | Confirms the auto-mode decider is opaque and does more than allow/deny. | None. |
| [#41615](https://github.com/anthropics/claude-code/issues/41615) | `permissions.allow` and PreToolUse hooks cannot override `.claude/` sensitive-file prompt | 🔴 OPEN (22d) | **Symmetric on the allow side.** Both `permissions.allow` and hook `{permissionDecision: "allow"}` fail to override the safetyCheck. Same root cause: safetyCheck is bypass-immune relative to the hook layer. | None. |
| [#37157](https://github.com/anthropics/claude-code/issues/37157) | `.claude/skills/` not exempt from protected-directory prompt | 🔴 OPEN (30+d) | **Contains the definitive source-level analysis** — community reverse-engineered functions `dN1`, `IHY`, `uHY`. *"PreToolUse with permissionDecision: 'allow' fires but doesn't override (runs before dN1). PermissionRequest hooks never fire in SDK subprocess mode."* | None. Open across 5+ versions. |
| [#51484](https://github.com/anthropics/claude-code/issues/51484) | Request: user-level opt-out for hardcoded sensitive-file check | 🔴 OPEN (1d) | Empirically confirms (2026-04-21) `PreToolUse` hook returning `"allow"` does not suppress the sensitive-file prompt. Requests: expose the check to hooks. | None. Labeled `enhancement`. |
| [#38500](https://github.com/anthropics/claude-code/issues/38500) | Feature request: pre-classifier hook | 🔴 OPEN (28d) | User explicitly requests a hook that runs **before** the classifier. | None. |
| [#35895](https://github.com/anthropics/claude-code/issues/35895) | v2.1.78 rejecting edits in `.claude/` in `dontAsk` mode | 🔴 OPEN (35d) | Community analysis explicitly names the model: *"2-layer permission model — Tool execution level (bypassable) and File-level sensitive check (NOT bypassed)"*. | None. Labeled `regression`. |
| [#30519](https://github.com/anthropics/claude-code/issues/30519) | Meta-issue: permissions matching is fundamentally broken | 🔴 OPEN (50+d) | Community-authored meta-issue. *"No milestones. No Anthropic-authored PRs. No roadmap. No tracking issue."* | None. |
| [#50331](https://github.com/anthropics/claude-code/issues/50331) | Auto mode injects undocumented behavioral system-reminder | 🔴 OPEN | Adjacent — documents the auto-mode system-reminder (we observe this same reminder in session). | None. |

**Closed with precedent-setting fixes (both by the same collaborator):**

| # | Title | State | What it fixed | Staff Quote |
|---|---|---|---|---|
| [#39344](https://github.com/anthropics/claude-code/issues/39344) | Hook `ask` silently overrides `permissions.deny` | 🟢 CLOSED v2.1.101 | Hook `ask` no longer outranks explicit `permissions.deny`. **Establishes precedent: hook-`ask` semantics are corrigible.** | `ashwin-ant` (Apr 18): *"This was fixed in v2.1.101 — A PreToolUse hook returning permissionDecision 'ask' no longer overrides explicit permissions.deny rules."* |
| [#41763](https://github.com/anthropics/claude-code/issues/41763) | bypassPermissions downgrade after approving suspicious path | 🟢 CLOSED v2.1.97 | Contains the literal source quote for the `safetyCheck` bypass-immune branch (equivalent to our `Ma_`). | `ashwin-ant` (Apr 18): *"This was fixed in v2.1.97"* |

### 3.2 The source-level quote from #41763

<details>
<summary><strong>Click: community-extracted <code>permissions.ts</code> comment (the design intent)</strong></summary>

Quoted verbatim from #41763 by community researcher:

```js
// 1g. Safety checks are bypass-immune — they must prompt even in
//     bypassPermissions mode.
if (toolPermissionResult?.behavior === "ask" &&
    toolPermissionResult.decisionReason?.type === "safetyCheck")
  return toolPermissionResult;
```

**Interpretation:** safety checks are architected to *always* produce a prompt, even overriding `bypassPermissions` mode. This is the intended design — safetyCheck = "must prompt no matter what."

**Our finding is that auto mode breaks this invariant.** In auto mode, the "must prompt" contract silently becomes "classifier decides (binary allow/deny)." The hook's dynamic `ask` is dropped, and the safety check's intent to prompt is also lost. Two layers of user-protection collapse into one classifier call.

</details>

### 3.3 Documentation coverage

| Concept | Documented? |
|---|---|
| `classifierApprovable` flag | ❌ **Undocumented** — leaks in binary/decompile only |
| auto-mode classifier bypass of `permissions.ask` | ⚠️ Described generally in [permission-modes.md](https://code.claude.com/docs/en/permission-modes); binary allow/deny output space **not explicitly stated** |
| hook-`ask` semantics in auto mode | ⚠️ **Ambiguous**. Hooks doc defines `"ask"` as *"prompts the user to confirm"* but doesn't address auto mode. |
| `safetyCheck` bypass-immunity | ❌ **Undocumented** as a named concept |
| Protected-paths list | ✅ [permissions.md](https://code.claude.com/docs/en/permissions) — *"writes to protected paths... in auto they route to the classifier"* |

The hook contract quote that's closest to addressing this:

> *"Hook decisions do not bypass permission rules. Deny and ask rules are evaluated regardless of what a PreToolUse hook returns, so a matching deny rule blocks the call and a matching ask rule still prompts even when the hook returned 'allow' or 'ask'."*
> — [/en/permissions](https://code.claude.com/docs/en/permissions)

Note what this quote covers vs. what it doesn't:
- ✅ Covers: `permissions.ask` rule + hook anything → prompt (normal mode)
- ❌ **Doesn't cover**: hook `ask` + built-in `safetyCheck` ask → classifier in auto mode

That gap is our bug.

### 3.4 Classification verdict

> [!IMPORTANT]
> **`FIX`** — with high confidence.

**Rationale:**

1. **Precedent is on our side.** Anthropic has explicitly fixed the symmetric case (#39344: hook `ask` vs `permissions.deny`). Staff response there establishes hook-`ask` semantics are corrigible.
2. **No official framing defends current behavior.** Zero docs, zero commit messages, zero maintainer comments describe this as intentional. Every user who encounters it files a bug or feature request.
3. **The documented hook contract** (`"ask"` → *"prompts the user to confirm"*) is violated.
4. **The documented safetyCheck invariant** (*"must prompt even in bypassPermissions mode"*) is also violated under auto mode.

**Counter-arguments:**
- Parent issue #42797 has been open 19 days with zero staff response. Anthropic's response velocity on this class is slow — possibly indicating a silent design preference (over-trust the classifier).
- But this pushes confidence from "certain" to "high" — not enough to classify differently.

**Not `TWEAK`**: we're not restoring prior behavior (it never worked this way) — we're aligning behavior with the documented contract.

**Not `FEATURE`**: this is a documented contract being honored, not a new capability being unlocked.

---

## 4. Patch Design Options

### 4.1 Option A — Narrow 1-byte flip in `Ma_` (recommended)

<details>
<summary><strong>Byte-level design</strong></summary>

**Anchor:** `b'ask rule/safety check requires full permission pipeline'` (unique diagnostic string; verified not to collide elsewhere).

**Old bytes (16 bytes):**
```
behavior==="ask"
```

**New bytes (16 bytes):**
```
behavior==="xsk"
```

**Net change:** 1-byte flip (`a` → `x`) inside the `===` comparison literal, inside the condition-scoped phrase `behavior==="ask"`. Same length.

**PatchDef:**
```python
PatchDef(
    name='hook-ask-no-override',
    description='Prevent Ma_ FJH-ask branch from overriding a user-hook permission ask (skips auto-mode classifier silent override for safety-check-gated paths)',
    kind=PatchKind.FIX,
    anchor=b'ask rule/safety check requires full permission pipeline',
    old=b'behavior==="ask"',
    new=b'behavior==="xsk"',
    window=200,
    min_version='2.1.109',
)
```

**Effect:**
- `D?.behavior === "xsk"` is never true (no real `behavior` value is `"xsk"`). Branch [5] (the bypass) becomes dead code.
- Control flow falls through to branch [7]: `await O(_,w,K,T,$,H)` (6-arg) → adapter routes to user prompt.
- Branches [1] (hook-deny), [4] (FJH-deny), [6] (hook-allow) all continue working.

</details>

**Pros:**
- Surgical, same-length, 1-byte flip. Maximum auditability.
- Stable bytes across v2.1.109..v2.1.116 (agent B verified).
- Safe failure mode if minifier renames surrounding variables — `scan_binary` returns `changed`, no apply.

**Cons:**
- **Hook=allow + FJH=ask edge case:** pre-patch, classifier decided; post-patch, hook-allow wins. Our current hook only emits `ask` (never `allow`), so this is dormant. But a future hook returning `allow` for safety-checked paths would auto-approve instead of consulting the classifier.

### 4.2 Option B — Narrow to auto mode (infeasible as same-length)

<details>
<summary><strong>Why this doesn't work</strong></summary>

Goal: make the bypass fire only when `mode !== "auto"`. The minimum code to express this needs a mode lookup:

```js
if(D?.behavior==="ask" && K.getAppState().toolPermissionContext.mode!=="auto") return ...;
```

That adds ~58 bytes between the existing test and the `return`. The original is a single-expression `if(X)Y;` with no slack. No useful same-length transform injects the mode check.

**Same-length alternatives don't work either:**
- Can't shorten the anchor string (string literals are codesign-stable bytes).
- Can't replace `O(_,w,K,T,$)` (5-arg) with `O(_,w,K,T,$,H)` (6-arg) — that adds bytes AND wouldn't gate on mode.

**Conclusion:** Option B requires length-changing patching (lief-based rebuild), which the project has deferred per `claude_binary_patching.py` docstring. Not recommended unless the workspace adopts length-changing patches broadly.

</details>

### 4.3 Option C — `classifierApprovable` flip (defense-in-depth candidate)

<details>
<summary><strong>Byte-level design — broad variant (all 3 safety-check emission sites)</strong></summary>

Flip `classifierApprovable: true` → `classifierApprovable: false` at each of the 3 emission sites in `deH`. Each is a 1-byte change (`!0` → `!1`).

**PatchDefs:**
```python
PatchDef(
    name='settings-safety-prompts-user',
    description='Force auto-mode classifier to skip safety-check for .claude/settings*.json (Ae_ site, SDK remote mode)',
    kind=PatchKind.TWEAK,
    anchor=b'if(Ae_($))return{safe:!1',
    old=b'classifierApprovable:!0',
    new=b'classifierApprovable:!1',
    window=200,
    min_version='2.1.90',
)

PatchDef(
    name='claude-dir-safety-prompts-user',
    description='Same for .claude/{commands,agents,skills} via TB5 (CLI mode — what our user hits)',
    kind=PatchKind.TWEAK,
    anchor=b'else if(TB5($))return{safe:!1',
    old=b'classifierApprovable:!0',
    new=b'classifierApprovable:!1',
    window=200,
    min_version='2.1.90',
)

PatchDef(
    name='sensitive-files-safety-prompts-user',
    description='Same for the sensitive-files list (jB5 site)',
    kind=PatchKind.TWEAK,
    anchor=b'which is a sensitive file',
    old=b'classifierApprovable:!0',
    new=b'classifierApprovable:!1',
    window=200,
    min_version='2.1.90',
)
```

**Effect:** `JM`'s `SPH(T.decisionReason, J => !J.classifierApprovable)` now matches the safety-check node → `A` becomes truthy → `if(A) return T` fires → JM returns the ask unchanged → GO5 sees `Y.behavior==="ask"` and calls `zKH()` → **user prompt**. Same final UX as Option A for settings.json.

**6 byte flips total** (2x per site due to `__BUN` duplicate).

</details>

**Pros vs A:**
- **No hook=allow + FJH=ask regression.** Control flow in `Ma_` is untouched. Classifier is bypassed inside `JM` by returning ask unchanged. Prompt surfaces regardless of hook behavior.
- **More version-stable** — `classifierApprovable:!0` has been stable since v2.1.90 (19+ versions of track record).

**Cons:**
- Broader effect: affects every safety-check emission, not just settings.json. `.claude/commands`, `.claude/agents`, `.claude/skills`, and the sensitive-file list all become "always prompt." For a defense-in-depth posture this is arguably *better*, not worse.
- 6 sites to patch vs 2 for Option A.
- If Anthropic ever changes `classifierApprovable` to a flag object or renames the property, more breakage surface.

### 4.4 Option D — FJH-safety-check return neutralization (not recommended)

<details>
<summary><strong>Byte-level design + why not recommended</strong></summary>

Patch the FJH ask-safety branch to never fire:

```python
PatchDef(
    name='fjh-no-safety-ask',
    description='Make FJH skip safetyCheck-based ask return, so Ma_ sees D=null and falls through',
    kind=PatchKind.TWEAK,
    anchor=b'if($?.behavior==="ask"&&(SPH($.decisionReason)||$.decisionReason?.type==="sandboxOverride"))return $',
    old=b'behavior==="ask"',
    new=b'behavior==="xsk"',
    window=100,
    min_version='2.1.90',
)
```

**Effect:** For settings.json, FJH returns null → Ma_'s `D?.behavior==="ask"` is `undefined==="ask"` → false → falls through. Since Y=="ask" (hook), reaches 6-arg default → user prompt.

**Why not recommended:**
- Broader than A: disables **all** FJH safety-check ask returns.
- FJH's safety-check-ask return line is duplicated in `EM5` (2x each = 4 sites). Anchoring only FJH is fragile; if the minifier re-orders or folds duplicates, we over/under-patch.
- **Same hook=allow+FJH=ask regression as Option A** (because Ma_ branch [5] never fires, hook=allow still falls through to branch [6]).

Essentially "Option A one layer deeper" with the same edge-case risk and no material benefit.

</details>

### 4.5 Feature-flag alternative — none exists

<details>
<summary><strong>Statsig / GrowthBook search results</strong></summary>

Statsig gates near the permission path:
- `tengu_iron_gate_closed` — controls fail-open vs fail-closed **when the classifier is network-unavailable**, not the bypass itself.
- `tengu_bash_allowlist_strip_all` — telemetry.
- `tengu_auto_mode_decision` — telemetry event name, not a gate.
- `tengu_auto_mode_denial_limit_exceeded` — telemetry.

The FJH-ask bypass in `Ma_` is unconditional — not gated by a flag we could toggle via `~/.claude.json → cachedGrowthBookFeatures` or `~/.claude.json → statsig`. **Binary patching is required.**

</details>

### 4.6 Recommendation

> [!TIP]
> **Primary: Option A.** Surgical, 1-byte flip, stable since v2.1.109, intent is obvious, scope precisely matches our gating hook.
>
> **Optional defense-in-depth: Option A + narrowed Option C (Ae_ + TB5 only).** If the user adds a future hook that returns `allow` for safety-checked paths, Option A alone would auto-approve. Pairing with Option C narrowed to the CLI mode site (TB5) and the SDK site (Ae_) — skip the sensitive-files site (jB5) — forces the classifier-skip path in `JM` so the user always prompts. Total: 2 (A) + 4 (C narrowed) = 6 byte flips, all version-stable since v2.1.109.

---

## 5. Scope & Risk Analysis

### 5.1 Effective scope of Option A

| Scenario | Pre-patch | Post-patch |
|---|---|---|
| Non-auto mode, any file | hook doesn't fire (returns 0) | unchanged |
| Auto mode, non-gated tool | hook returns 0 | unchanged |
| Auto mode, gated tool, FJH clear | prompt shown | unchanged |
| **Auto mode, gated tool, FJH=ask (settings.json)** | **classifier decides (allow)** | **prompt shown** |
| Any mode, hook deny | deny | unchanged |
| Any mode, FJH deny | deny | unchanged |
| Auto mode, gated tool, hook=allow, FJH=ask | classifier decides | **hook-allow wins (silent allow)** |

Last row is the edge case. Today: no effect (our hook only emits ask). Future-hook caveat.

### 5.2 Risk matrix

| Risk | Severity | Mitigation |
|---|---|---|
| Minifier renames vars in future version | Medium | `scan_binary` returns `changed`, patch not applied → safe fail |
| Anthropic adds new `behavior` value routed through FJH-ask | Low | New value won't match neutered `"xsk"`, effect is silent no-op |
| Future hook that emits `allow` on safety-checked paths | Low-Medium | Hook-allow wins (was: classifier decided). Option C pairing eliminates this. |
| Anthropic refactors `Ma_` | Medium | Same as rename — patch fails scan, no regression |
| Cross-version stability (v2.1.109+) | Low | Agent B verified bytes identical 109, 110, 112, 114, 116 |

---

## 6. Implementation Plan

1. Add PatchDef(s) to `cc-lib/cc_lib/claude_binary_patching.py`:
   - Primary: `hook-ask-no-override` (Option A)
   - Optional pair: `settings-safety-prompts-user` + `claude-dir-safety-prompts-user` (narrowed Option C)
2. Update module docstring: Patches section + Version Log entry for v2.1.116.
3. `claude-binary-patcher check --all` — verify `status='unpatched'` (old bytes found).
4. `claude-binary-patcher apply <name>` — same-length replace, ad-hoc resign preserves entitlements.
5. Restart Claude Code.
6. Empirical verification:
   - Edit `.claude/settings.json` under auto mode → should prompt ✓
   - Edit `protocol.py` or similar code file under auto mode → should still prompt ✓
   - Bash command under auto mode → unchanged (unaffected) ✓
7. If upstream fixes or refactors in a later version, either retarget or remove the PatchDef.

---

## 7. Open Questions

> [!QUESTION]
> **Q1:** Ship Option A alone, or pair with narrowed Option C for defense-in-depth? (Trade-off: simpler vs. more robust to future hooks.)

> [!QUESTION]
> **Q2:** Include the sensitive-files site (jB5) in the Option C pairing, or skip it to avoid broader scope? (The hook-ask bypass only affects settings.json in our observed workflow; jB5 handles a different file list we haven't empirically verified.)

> [!QUESTION]
> **Q3:** File upstream issue referencing #42797 / #51255 with our source-level analysis? Would give Anthropic a concrete path to fix, though their response velocity on this class is slow.

---

## Appendix A — Test methodology

<details>
<summary><strong>How to reproduce the bypass</strong></summary>

1. Enable auto mode.
2. Enable debug logging (via `/debug` or env var).
3. Edit `.claude/settings.json` with a trivial whitespace change.
4. Observe: no prompt surfaces, Edit succeeds silently.
5. Grep `~/.claude/debug/<session>.txt` for `"full permission pipeline"` — should see the diagnostic.
6. Compare against Edit on `protocol.py` (or any non-gated path) — should prompt.

The debug line is the smoking gun:
```
[DEBUG] Hook returned 'ask' for Edit, but ask rule/safety check requires full permission pipeline
[DEBUG] [auto-mode] new action being classified: Edit /path/to/file
[INFO]  Slow permission decision: XXms for Edit (mode=auto, behavior=allow)
```

</details>

## Appendix B — Workarounds documented in the wild

Per Agent A's landscape survey, these workarounds exist across GitHub issues. **None achieve "auto mode + inline yes/no confirmation on hook-gated operations"** — that capability doesn't exist in Claude Code today.

| Workaround | Source | Trade-off |
|---|---|---|
| Hook emits `deny` instead of `ask` | #51255, #42797, #39344 | Works in all modes. User can't approve inline — must exit Claude. |
| Hook with `exit 2` + stderr | #40641 | Hard-block, can't be overridden by any other hook/rule (even `--dangerously-skip-permissions`). Same "no inline approval" limitation. |
| `autoMode.soft_deny` prose rules | #42797 comment by `mbu-ab` | Put the gate in the auto-mode prompt itself — classifier reads it and denies with instruction to toggle auto mode off. Indirect. |
| Behavioral steering ("always ask me before X" in chat) | #42797 | Fragile — depends on model remembering. |
| Restructure commands to avoid safety heuristics | #34106 | Not applicable for file-path safetyChecks. |
| Binary patching (community-authored) | #37157 | Community patch exists for exempting paths from `IHY`/`uHY`. Has to be re-applied on every CLI update. **This is what we're doing.** |

## Appendix C — Canonical user-request quotes

<details>
<summary><strong>From #51255 (rpicatoste, Apr 20) — clearest public statement of our exact bug shape</strong></summary>

> A PreToolUse hook returning permissionDecision: "ask" is silently auto-approved when running in auto mode. The user is never prompted for confirmation. This makes it impossible to gate dangerous operations (git commit, git push) while using auto mode for everything else. [...] There is no permissionDecision value that means 'always prompt, regardless of mode'.

</details>

<details>
<summary><strong>From #42797 comment (mbu-ab, Apr 17) — classifier output space is binary</strong></summary>

> The classifier's output space appears limited to allow/deny — it can't route to 'ask' even when permissions.ask is explicitly configured. [...] Per Anthropic docs, this is by design: 'Auto mode lets Claude execute without permission prompts. A separate classifier model reviews actions before they run.' — permission-modes.md. So permissions.ask is effectively a no-op in auto mode.

</details>

<details>
<summary><strong>From #37157 (IMBurbank, Mar 22) — source-level confirmation hooks run *before* safetyCheck</strong></summary>

> PreToolUse with permissionDecision: 'allow' fires but doesn't override (runs before dN1). PermissionRequest hooks never fire in SDK subprocess mode — dN1 returns {behavior: 'ask'} before the PermissionRequest dispatch path is reached, and the SDK subprocess treats the unresolved ask as a denial.

</details>

## Appendix D — Cross-version byte stability

Counts measured across locally-available originals `2.1.86, 2.1.87, 2.1.90, 2.1.92, 2.1.109, 2.1.110, 2.1.112, 2.1.114, 2.1.116`.

| Metric | 2.1.86 | 2.1.87 | 2.1.90 | 2.1.92 | 2.1.109 | 2.1.110 | 2.1.112 | 2.1.114 | 2.1.116 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anchor `ask rule/safety check requires full permission pipeline` | 0 | 0 | 0 | 0 | 3 | 3 | 3 | 3 | 3 |
| `behavior==="ask"` in 200B before anchor | — | — | — | — | 2 | 2 | 2 | 2 | 2 |
| `classifierApprovable:!0` | 0 | 0 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| `if(Ae_($))return{safe:!1` | — | — | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `else if(TB5($))return{safe:!1` | — | — | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

**Identifier drift in the `Ma_` FJH-result variable:**
- v2.1.109 through v2.1.114: `j`
- v2.1.116: `D`

The log function varies too: `h` (older) vs `y` (2.1.116). A patch matching the full `if(j?.behavior==="ask")return h(\`...` would NOT survive the v2.1.114→v2.1.116 rename. **But `behavior==="ask"` is stable across all versions** — Option A's patch is version-stable from 2.1.109 onwards.

**`classifierApprovable:!0` is stable from 2.1.90 onwards** — Option C survives an extra ~19 versions.

## Appendix E — Reference file paths

- **Unpacked JS analyzed:** `/var/folders/8m/qjcw0jr90v1gnnqqstwz5zgm0000gn/T/claude-unpack-vew64gbj/claude-2.1.116`
- **Packed binary (live):** `/Users/chris/.local/share/claude/versions/2.1.116`
- **Historical binaries:** `/Users/chris/.claude-workspace/binary-patcher/originals/{2.1.86, 2.1.87, 2.1.90, 2.1.92, 2.1.109, 2.1.110, 2.1.112, 2.1.114, 2.1.116}`
- **Existing patcher CLI:** `/Users/chris/claude-workspace/scripts/claude-binary-patcher.py`
- **Patch definitions:** `/Users/chris/claude-workspace/cc-lib/cc_lib/claude_binary_patching.py`
- **Unpack script:** `/Users/chris/claude-workspace/scripts/claude-unpack-binary.py`
- **User's gating hook:** `/Users/chris/claude-workspace/hooks/ask-before-auto-approval.py`

## Appendix F — Key binary offsets (packed v2.1.116)

| Site | Mach-O | `__BUN` dup |
|---|---:|---:|
| `Ma_` target line start (`if(D?.behavior==="ask")...`) | 79992897 | 194599769 |
| Anchor `ask rule/safety check...` | 79992970 | 194599842 |
| `behavior==="ask"` in target condition | 79992910 | 194599782 |
| `classifierApprovable:!0` site A (`Ae_`) | 82704211 | 197311083 |
| `classifierApprovable:!0` site B (`TB5`) | 82704355 | 197311227 |
| `classifierApprovable:!0` site C (`jB5`) | 82704501 | 197311373 |