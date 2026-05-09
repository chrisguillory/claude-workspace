---
name: binary-patcher-migration
description: "Migrate the binary patcher to a new Claude Code version. Re-derive bytes for changed patches, investigate missing anchors, mark obsoleted patches, verify empirically in fresh sessions, and open a PR. Use when Claude Code releases a new version (announced or stealth-on-CDN) and the patcher needs to follow."
argument-hint: "<version> (e.g., '2.1.131')"
disable-model-invocation: true
user-invocable: true
effort: max
allowed-tools:
  - Bash
  - Read
  - Edit
  - Glob
  - Grep
  - Agent
  - WebFetch
---

# Binary Patcher Migration

Operational manual for migrating patches when Claude Code releases a new version.
Minified JS identifiers in the binary drift per release, breaking patches whose
byte sequences encode specific identifiers. This skill walks through the
six-phase migration: acquire → status → per-patch handling → apply + verify →
document → PR.

Target version: `$ARGUMENTS`

## Environment

```
Active claude:     !`readlink ~/.local/bin/claude | xargs basename 2>/dev/null`
Installed:         !`ls ~/.local/share/claude/versions/ 2>/dev/null | grep -E '^[0-9]' | sort -V | tr '\n' ' '`
Latest on CDN:     !`claude-version-manager list --remote 2>/dev/null | head -8 | tail -5`
Current branch:    !`git -C ~/claude-workspace branch --show-current 2>/dev/null`
Working tree:      !`git -C ~/claude-workspace status --short 2>/dev/null | head -10`
Latest commit:     !`git -C ~/claude-workspace log --oneline -1 main 2>/dev/null`
```

## Source-of-truth files

- **Patches**: `cc-lib/cc_lib/claude_binary_patching.py` — `PATCHES` sequence and module docstring (alphabetical Patches section, Anchor Presence Survey, Site Count Evolution table, Version Log).
- **CLI**: `scripts/claude-binary-patcher.py` — `apply` / `check` / `restore`.
- **Source mirror**: `~/claude-code-best/` — TypeScript reference for what minified IDs map to. Structural guide; binary is ground truth.
- **Pristine originals**: `~/.claude-workspace/binary-patcher/originals/<version>` — vanilla binaries kept for restore + comparison.

## Phase 0: Acquire + branch

```bash
claude-version-manager fetch $ARGUMENTS --activate

# Public changelog (fold to a subagent if extensive)
gh api repos/anthropics/claude-code/contents/CHANGELOG.md --jq '.content' | base64 -d | head -60

# GitHub tag presence — stealth releases (e.g., 2.1.127, 2.1.129) have none
gh release view v$ARGUMENTS --repo anthropics/claude-code --json tagName 2>&1 | head -3

cd ~/claude-workspace
git checkout main && git pull --ff-only
git checkout -b feat/binary-patches-$ARGUMENTS
```

Stealth releases: proceed with extra caution (Anthropic may pull the build — 2.1.127 was apparently a failed kill build). If user prefers stability, pin to the latest tagged version instead.

## Phase 1: Status check

```bash
claude-binary-patcher check --all
```

Each patch reports one of:

| Status | Meaning | Next |
|---|---|---|
| `applied` | Anchor + new bytes match | Skip |
| `unpatched (N sites)` | Anchor + old bytes match | Apply (no re-derivation) |
| `changed (anchor found, code different)` | Anchor still matches but bytes drifted | **Re-derive** (Phase 2a) |
| `missing (anchor not found)` | Anchor itself is gone | **Investigate** (Phase 2b) |
| `skipped (out of range, max_version=X)` | Patch's `[min_version, max_version]` excludes the active CC version | Skip — no action needed |

Obsolete patches (those with `max_version` set to a prior release) automatically report as `skipped`, and `apply --all` filters them out of per-patch output entirely — they appear only as a single end-of-output summary line. No manual cross-referencing required.

Categorize all patches before doing any work.

## Phase 2a: Re-derive `changed` patches

### Step 1: Get context around the anchor

```bash
PATCHED=~/.local/share/claude/versions/$ARGUMENTS

# Replace ANCHOR_REGEX with the patch's anchor (escape regex chars: parens, dots, quotes)
# Window sizes by patch shape:
#   simple gate-flip / startsWith → {120} ... {200}
#   JSX block (show-subagent etc) → {200} ... {800} (block can be 500+ bytes)
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(.{120}ANCHOR_REGEX.{200})/sg) { print "$&\n\n"; last }
' "$PATCHED"
```

### Step 2: Identify the renames

Compare the printed bytes to the patch's existing `old` field. Identifier categories that drift each release:

| Category | Examples (2.1.126 → 2.1.128 → 2.1.131) | Stability |
|---|---|---|
| Statsig accessor | `G_` → `Z_` → `G_` | **Oscillates** — back-and-forth between releases |
| Gate function | `at()` → `Me()` → `ve()` | Renamed every release |
| Constant identifier | `e76` → `EK6` (different prefix kinds in 2.1.131) | Renamed every release |
| Helper chain | `Q36(Yj_(T))` → `$D_(TQH(T))` → `RD_(WcH(T))` | Renamed every release |
| React module | `_8` → `q8` → `O8` | Renamed every release |
| JSX components | `IL5/M6/FY_/...` → `JN5/X6/Gw_/...` → `hv5/G6/bw_/...` | All renamed every release |
| String literals | `"contentArray"`, `"tengu_scratch"`, prop names | **Stable** — use as anchors |

Don't assume drift is monotonic — a name reverted in 2.1.131 may revert again. The bytes are version-specific; the strategy is forever.

### Step 3: Construct same-length new bytes

The patcher requires `len(old) == len(new)`. Verify before editing:

```bash
echo -n "Old: "; printf 'function ve(){return G_("tengu_scratch",!1)}' | wc -c
echo -n "New: "; printf 'function ve(){return!0/*scratchpad always*/}' | wc -c
```

When lengths differ, pad to match:
- Whitespace inside string literals: `wrap:"truncate"` → `wrap:"wrap"    ` (4 trailing spaces)
- Comment block extension: `if(!G_("tengu_coral_fern",!1))return[]` → `if(0/*coral_fern_gate_check*/)return[]`
- Drop redundant fields + trailing spaces: `return{...,schema:RD_(WcH(T))}` → `return{...,type:"contentArray"}   ` (3 trailing spaces; schema field dropped)

### Step 4: Verify uniqueness across `__BUN` duplicate

The JS bundle is duplicated in the `__BUN` segment from 2.1.0+, so each patch has 2 sites. Both must have identical bytes:

```bash
grep -aoE 'EXPECTED_OLD_BYTES_REGEX' "$PATCHED" | sort -u | wc -l
# Should print: 1 (one unique sequence appearing twice)
```

### Step 5: Update PatchDef + apply

Edit `cc-lib/cc_lib/claude_binary_patching.py`. Update `old`, `new`, possibly `anchor`, and bump `min_version` to the new version.

```bash
claude-binary-patcher check <patch-name>     # expect: unpatched (2 sites)
claude-binary-patcher apply <patch-name>
claude-binary-patcher check <patch-name>     # expect: applied
```

## Phase 2b: Investigate `missing` patches

Quick checks first (each ~10 seconds), subagents only when those fail.

### Quick check 1: Renamed function

If the anchor was a function header (`function NAME(){...}`), search by the surviving Statsig flag name:

```bash
grep -aoE 'function \w+\(\)\{return \w+\("FLAG_NAME"' "$PATCHED"
```

Found one match → just renamed. Update anchor + `old` bytes, retry.

### Quick check 2: Renamed identifier

If the anchor was a constant reference like `T.content.startsWith(EK6)`, search structurally:

```bash
grep -aoE 'T\.content\.startsWith\(\w+\)' "$PATCHED" | sort -u
```

One result → identifier just renamed. Update bytes.

### Quick check 3: Feature removed

```bash
ORIG=~/.claude-workspace/binary-patcher/originals/<previous-version>
echo -n "old: "; grep -ac "feature_string" "$ORIG"
echo -n "new: "; grep -ac "feature_string" "$PATCHED"
```

Zero in new → feature REMOVED → mark obsolete (Phase 4).

### Quick check 4: Renamed JSX block

For visibility patches that pivot on a JSX prop set, search by structurally-stable props:

```bash
grep -aoE 'createElement\(\w+,\{progressMessages:\w+,tools:\w+,verbose:\w+\}\)' "$PATCHED"
```

One match → component just renamed. Pull the full JSX block context with a wide perl window (`.{200}...{800}`) and update the new bytes.

### Subagent investigation

When the four quick checks all fail or you're hitting an architectural change (the 2.1.126 → 2.1.128 session-memory removal was the canonical case), spawn 3-4 parallel `unrestricted-worker` subagents on Opus across vectors:

| Vector | What to investigate |
|---|---|
| **Binary diff** | Truly removed or just renamed/relocated? Compare string sets between OLD and NEW. Empirical, ground-truth-first. |
| **GitHub** | Issues + PRs + commits between previous and current tag. `gh release view`. Community forks (Piebald-AI/tweakcc, claude-code-best). |
| **Anthropic public** | docs.anthropic.com / code.claude.com / engineering blog / changelog / Discord. Silence often noted as evidence. |
| **Community** | Reddit (r/ClaudeAI), Hacker News, peer dev blogs (claudefa.st, giuseppegurgone, simonw). |
| **Source mirror** | `~/claude-code-best/` — structural reference. Mirror lags upstream and may have its own divergent fixes. |

For routine bug-fix releases (no architectural changes signalled in the changelog), the four quick checks usually suffice. Don't burn tokens on subagents reflexively — reserve them for genuine "where did this go?" mysteries.

#### Subagent prompt template

```
Empirical investigation of [specific question].

# Background
[What was, what changed — concrete byte-level facts]

# The question
[ONE clear question with verdict format]

# Sources
1. ~/.claude-workspace/binary-patcher/originals/<old-version>
2. ~/.local/share/claude/versions/<new-version>
3. ~/claude-code-best/  (TypeScript source mirror)
4. ~/.claude/projects/<project>/<sid>.jsonl  (recent session for empirical traces)

# Output format
Under 500 words. Sections:
**Verdict**: ✅ / ⚠️ / ❌
**Empirical evidence**: quote actual bytes / function bodies, cite paths + offsets
**Open questions**: what remains uncertain

Don't speculate. Cite everything.
```

## Phase 3: Apply + empirical verify

```bash
claude-binary-patcher apply --all   # pre-flight refuses if any RequiredSetting unsatisfied
claude-binary-patcher check --all   # confirm final state
```

Verify each patch with significant runtime behavior in a **fresh `claude-exec` session** (not the current one — the running process has the pre-loaded binary in memory):

| Patch family | How to verify |
|---|---|
| `hook-ask-no-override` | `.claude/settings.json` edit in auto-mode + safetyCheck — hook ask not silently overridden |
| `mcp-array-content-to-string` | Slack-style MCP tool with content array — renders JSON instead of blank |
| `statusline` | Bottom statusline wraps multi-line, doesn't truncate with `…` |
| `inject-searching-past-context-prompt` | Ask "what do you know about my preferences from past sessions?" — Claude spontaneously `Grep`s the project memory dir |
| `scratchpad` | System prompt has "## Scratchpad Directory" section; dir created on first write |
| `show-subagent-prompt-tools-response` | Subagent in verbose/transcript mode expands to show prompt + tool calls + response |

### Vanilla-vs-patched comparison protocol

If patched behavior looks wrong, **don't assume the patch is broken**. Anthropic may have changed the underlying behavior so the patch is now inert (a no-op) or unnecessary (no bug to fix). Compare:

```bash
claude-binary-patcher restore                    # vanilla
# user runs the test in a fresh session, reports verbatim what they see
claude-binary-patcher apply --all                # back to patched
# same test in another fresh session
```

| Vanilla | Patched | Conclusion |
|---|---|---|
| Wrong / empty | Same wrong / empty | Anthropic changed upstream → mark obsolete |
| Works correctly | Wrong | Patch broke something → debug or revert |
| Wrong | Works correctly | Patch is doing its job → ship |

Real example: in 2.1.128, `reject-show-comment` looked broken because rejection rendering was empty. Vanilla comparison revealed Anthropic had silently changed rendering to be empty in vanilla too — patch became obsolete.

## Phase 4: Mark obsolete patches

When a patch's underlying feature is removed or behavior changed upstream such that the patch is no longer applicable:

1. Set `max_version='<previous-supported-version>'` on the PatchDef.
2. Update the description to explain WHY (factual, no journey residue):

```python
PatchDef(
    name='write-session-summary',
    description=(
        'Enable background extraction that writes <sid>/session-memory/summary.md for '
        'cross-session context. Obsolete in 2.1.128+ — Anthropic removed the underlying '
        'feature (tengu_session_memory and the summary.md write path) and reattached '
        '/dream to write into auto-memory typed files instead.'
    ),
    ...
    min_version='2.1.126',
    max_version='2.1.126',  # ← THIS LINE
    ...
)
```

3. Document in the Version Log entry with the upstream rationale (cite issues, postmortems, community signals).

`apply --all` silently filters obsoleted patches out of per-patch output and reports them in a single end-of-output summary line. Don't delete the PatchDef — keep it for historical reference and users on older versions.

## Phase 5: Documentation

Edit `cc-lib/cc_lib/claude_binary_patching.py` module docstring.

### Site Count Evolution table

Add a row for the new version. Only update columns for patches with multi-version history; new single-version patches go in the Version Log only.

```
    Version   statusline   mcp-array-content-to-string   write-session-summary   inject-searching-past-context-prompt   sm-compact
    2.1.128   2            2                             0 (removed)      2                                      —
    2.1.131   2            2                             0 (removed)      2                                      —
    <new>     ?            ?                             ?                       ?                                      ?
```

### Version Log entry

Add at the **top** of `Version Log::` (newest first). Date in `YYYY-MM-DD`.

For routine bug-fix releases, keep the entry terse (one line per patch):

```
    <new-version> (<YYYY-MM-DD>)
        <One-line release character — bug fixes, regressions, etc.>

        Patch updates:
        - <patch-name>: <change one-liner>
        - <patch-name>: clean apply (anchor + bytes stable since <prev-version>).
```

For releases with architectural changes, expand to a paragraph + per-patch deep notes — see the existing 2.1.128 entry as the reference (session-memory removal context).

## Phase 6: Pre-commit + commit + push + PR

```bash
# Pre-commit twice (auto-fixers may modify on first run)
uv run pre-commit run --files cc-lib/cc_lib/claude_binary_patching.py \
  || uv run pre-commit run --files cc-lib/cc_lib/claude_binary_patching.py

# Commit
git add cc-lib/cc_lib/claude_binary_patching.py
git commit -m "$(cat <<'EOF'
binary-patcher: re-derive patches for <version>

<One-paragraph summary of release character + patch changes>

Patch updates:
- <patch-name>: <change>
- <patch-name>: <change>
EOF
)"

# Push + PR
git push -u origin feat/binary-patches-$ARGUMENTS
gh pr create --title "Binary patches for Claude Code $ARGUMENTS" --body-file scratch/pr-binary-patches-$ARGUMENTS.md
```

PR description structure (real examples in PR #102, #106, #108):
- TL;DR table: status per patch
- Per-patch deep-dives for non-trivial changes
- Empirical verification checklist
- Outcome summary (N applicable patches, M sites)
- Links to investigation gists if subagents were spawned

## Cheatsheet

```bash
PATCHED=~/.local/share/claude/versions/$ARGUMENTS
ORIG=~/.claude-workspace/binary-patcher/originals/<previous-version>
```

### Find context around a known anchor

```bash
# Standard window for simple patches
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(.{120}ANCHOR_REGEX.{200})/sg) { print "$&\n\n"; last }
' "$PATCHED"

# Wider window for JSX-block patches (block can be 500+ bytes)
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(.{200}ANCHOR_REGEX.{800})/sg) { print "$&\n\n"; last }
' "$PATCHED"
```

### Find a function definition

```bash
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(function NAME\([^)]*\)\{.{0,400}?\})/sg) { print "$1\n"; last }
' "$PATCHED"
```

### Find a feature gate function by Statsig flag name

```bash
# Returns "function <NAME>(){return <ACCESSOR>("flag_name",!1)}"
grep -aoE 'function \w+\(\)\{return \w+\("FLAG_NAME"' "$PATCHED"
```

### Find a JSON.stringify wrapper (post-2.1.128 tracing pattern)

```bash
# Direct alias (older versions)
grep -aoE '\b\w+=JSON\.stringify\b' "$PATCHED" | sort -u

# Tracing-decorated wrapper (2.1.128+)
grep -aoE 'function \w+\(H[^)]*\)\{using \w+=\w+`JSON\.stringify' "$PATCHED"
```

### Find a JSX site by stable prop set

```bash
grep -aoE 'createElement\(\w+,\{progressMessages:\w+,tools:\w+,verbose:\w+\}\)' "$PATCHED"
```

### Verify byte budget

```bash
echo -n "Old: "; printf 'old_bytes_here' | wc -c
echo -n "New: "; printf 'new_bytes_here' | wc -c
```

### Verify both `__BUN` sites have identical bytes

```bash
grep -aoE 'pattern' "$PATCHED" | sort -u | wc -l
# Should print: 1
```

### Compare feature presence between versions

```bash
echo -n "old: "; grep -ac "feature_string" "$ORIG"
echo -n "new: "; grep -ac "feature_string" "$PATCHED"
```

### Find all Statsig flag references

```bash
grep -aoE '\b\w+\("flag_name"[^)]*\)' "$PATCHED" | sort -u
```

## Anchor selection rules

- **Anchor must survive the patch.** If the anchor is part of `old`, post-patch detection breaks (status reports `missing`). Use stable post-context substrings.
- **String literals are most stable** (flag names, prop names, error messages, JSX prop sets).
- **Minified function names are LEAST stable** — typically renamed every release.
- **Examples that work**:
  - `b'"contentArray"'` (string literal)
  - `b'){let Y;if(_[5]===Symbol.for("react.memo_cache_sen'` (React Compiler memoization marker, post-context)
  - `b'O8.createElement(hv5,{progressMessages:_,tools:q,verbose:K})'` (JSX prop set, structurally stable across T→K outer substitution)
- **Examples that DON'T work** (avoid):
  - `b'q.content.startsWith(UZH)'` when the patch replaces `UZH` with `"Z"` — anchor disappears post-patch
  - Bare 2-3 character minified IDs without surrounding context — too many false matches

## Patch-strategy patterns

| Strategy | When to use | Example |
|---|---|---|
| **Falsify a check** | Routing decision routes content to a wrong renderer | `T.content.startsWith(EK6)` → `T.content.startsWith("Z")` |
| **Short-circuit gate body** | Statsig gate cached false, blocking a feature | `function ve(){return G_("tengu_scratch",!1)}` → `function ve(){return!0/*scratchpad always*/}` |
| **Variable substitution** | JSX block conditionally hides info | 4× `T` → `K` in show-subagent JSX (T was isTranscriptMode, K is verbose) |
| **Discriminator preservation** | Patch breaks a related consumer | `return T` → `return q!=="ide"?EH(T):T` (preserves IDE plugin's array-shape consumer) |

## Antipatterns

- **Don't apply --all blindly without `check` first.** The 5-state table tells you what's needed.
- **Don't add NEW patches in a version-migration PR.** That's scope creep. Re-derive existing patches; new patches go in their own PR.
- **Don't skip vanilla-vs-patched comparison when behavior is ambiguous.** `reject-show-comment` looked broken on 2.1.128 but vanilla was also "broken" — Anthropic had silently changed the rendering. Saved by the comparison.
- **Don't reflexively spawn subagents for `missing` patches.** Routine bug-fix releases solve with the four Phase 2b quick checks. Reserve subagents for architectural changes.
- **Don't trust subagent verdicts without empirical follow-up.** After a subagent says "this should work," still test in a fresh session.
- **Don't trust a 100-character preview when verifying constants.** I assumed `UZH` and `EK6` were the same constant because their first 100 chars matched — they diverged at character 152 (`STOP what you...` vs `To tell you how...`). Always extract the FULL constant content before equating.
- **Don't add patches based on flawed routing analysis.** `reject-show-comment-dispatcher` was added based on misreading the dispatcher's branches. Trace the ACTUAL flow through the function, not the assumed one.
- **Don't keep dead patches "just in case."** If Anthropic removed the feature, mark obsolete with `max_version`. Don't try to "make it work" when there's nothing to fix.
- **Don't write journey residue in docstrings or PR descriptions.** Document state, not the path.
- **Don't forget pre-commit before commit.** Failed pre-commit means a wasted commit message.

## Phases checklist

- [ ] Phase 0: `claude-version-manager fetch <ver> --activate` + read changelog + branch off main
- [ ] Phase 1: `claude-binary-patcher check --all` + categorize statuses (out-of-range patches auto-report as `skipped` — no manual cross-reference)
- [ ] Phase 2a: re-derive each `changed` patch (context → renames → same-length new bytes → uniqueness check)
- [ ] Phase 2b: investigate each `missing` patch (4 quick checks → subagents if needed → rename / refactor / removed)
- [ ] Phase 3: apply + empirical-verify in fresh `claude-exec` session (vanilla comparison if ambiguous)
- [ ] Phase 4: mark obsoleted patches with `max_version`
- [ ] Phase 5: update Site Count Evolution table + add Version Log entry (terse for routine releases)
- [ ] Phase 6: pre-commit + commit + push + PR
