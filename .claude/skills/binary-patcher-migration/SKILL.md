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

# Binary Patcher Migration: Re-derive Patches for a New Claude Code Version

When Claude Code releases a new version, the minified JS identifiers in the
binary drift, breaking patches that depend on specific byte sequences. This
skill is the operational manual for migrating to the new version: identify
which patches still apply, re-derive bytes for those that drifted, investigate
patches whose anchors disappeared (renamed, refactored, or feature removed),
verify everything empirically, and open a focused PR.

Target version: `$ARGUMENTS`

## Environment

```
Active claude:     !`readlink ~/.local/bin/claude | xargs basename 2>/dev/null`
Installed:         !`ls ~/.local/share/claude/versions/ 2>/dev/null | grep -E '^[0-9]' | sort -V | tr '\n' ' '`
Latest on CDN:     !`claude-version-manager list --remote 2>/dev/null | head -8 | tail -5`
Current branch:    !`git -C /Users/chris/claude-workspace branch --show-current 2>/dev/null`
Working tree:      !`git -C /Users/chris/claude-workspace status --short 2>/dev/null | head -10`
Latest commit:     !`git -C /Users/chris/claude-workspace log --oneline -1 main 2>/dev/null`
```

## Source-of-truth files

- **Patch definitions**: `cc-lib/cc_lib/claude_binary_patching.py` — `PATCHES` sequence, plus the module docstring containing the alphabetical Patches section, Anchor Presence Survey, Site Count Evolution table, and Version Log.
- **CLI**: `scripts/claude-binary-patcher.py` — apply/check/restore commands.
- **Source mirror**: `~/claude-code-best/` — TypeScript reference for understanding what minified IDs map to. Use as a structural guide; binary is ground truth.
- **Pristine originals**: `~/.claude-workspace/binary-patcher/originals/<version>` — vanilla binaries kept around for restore + comparison.

## Phase 0: Acquire + branch

```bash
# Download the new version (verifies SHA-256), activate it
claude-version-manager fetch $ARGUMENTS --activate

# Read the public changelog — note unannounced versions warrant extra caution
gh api repos/anthropics/claude-code/contents/CHANGELOG.md --jq '.content' | base64 -d | head -60

# Check whether this version has a GitHub tag (stealth releases like 2.1.127, 2.1.129 do not)
gh release view v$ARGUMENTS --repo anthropics/claude-code --json tagName 2>&1 | head -3

# Branch off main
cd /Users/chris/claude-workspace
git checkout main && git pull --ff-only
git checkout -b feat/binary-patches-$ARGUMENTS
```

If the version is stealth (no GitHub tag, not in changelog) — proceed but be aware that Anthropic may pull the build (e.g., 2.1.127 was apparently a failed kill build). Consider also pinning to the latest tagged version if the user prefers stability.

## Phase 1: Status check — read the 4-state table

```bash
claude-binary-patcher check --all
```

Each patch reports one of:

| Status | Meaning | Next action |
|---|---|---|
| `applied` | Anchor + new bytes match | Skip |
| `unpatched (N sites)` | Anchor + old bytes match — patch ready to apply | Apply (no re-derivation needed) |
| `changed (anchor found, code different)` | Anchor still matches but bytes drifted (minified IDs renamed) | **Re-derive** — see Phase 2a |
| `missing (anchor not found)` | Anchor itself is gone | **Investigate** — see Phase 2b (rename, refactor, or feature removed) |

Categorize all patches before doing any work. The status determines the path.

## Phase 2a: Re-derive `changed` patches

For each patch reporting `changed`:

### Step 1: Get context around the anchor

```bash
PATCHED=/Users/chris/.local/share/claude/versions/$ARGUMENTS

# Replace ANCHOR with the patch's anchor string (escape regex chars as needed)
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(.{120}ANCHOR.{200})/sg) {
    print "$&\n\n";
    last;
}
' "$PATCHED"
```

### Step 2: Identify the renames

Compare the printed bytes to the patch's existing `old` field. Common renames seen in this project:

| 2.1.126 → 2.1.128 example | What changed |
|---|---|
| `G_("flag",!1)` → `Z_("flag",!1)` | Statsig accessor renamed |
| `function at(){...}` → `function Me(){...}` | Function name renamed |
| `T.content.startsWith(e76)` → `T.content.startsWith(EK6)` | Constant identifier renamed |
| `Q36(Yj_(T))` → `$D_(TQH(T))` | Helper-chain identifiers renamed |
| `_8.createElement(IL5,...)` → `q8.createElement(JN5,...)` | React module + component identifiers renamed |

### Step 3: Construct same-length new bytes

The patcher requires `len(old) == len(new)`. Use `printf | wc -c` to verify before editing the PatchDef:

```bash
echo -n "Old: "; printf 'function Me(){return Z_("tengu_scratch",!1)}' | wc -c
echo -n "New: "; printf 'function Me(){return!0/*scratchpad always*/}' | wc -c
```

If lengths differ: pad with whitespace inside string literals or extend a `/*...*/` comment block until they match. Examples:

- `wrap:"truncate"` → `wrap:"wrap"    ` (4 trailing spaces inside the string literal)
- `if(!Z_("tengu_coral_fern",!1))return[]` → `if(0/*coral_fern_gate_check*/)return[]` (comment padding)
- `return{content:T,type:"contentArray",schema:$D_(TQH(T))}` → `return{content:q!=="ide"?SH(T):T,type:"contentArray"}   ` (3 trailing spaces; schema field dropped)

### Step 4: Verify uniqueness across __BUN duplicate

The JS bundle is duplicated in the `__BUN` segment from 2.1.0+, so each patch has 2 sites. Both must have identical bytes:

```bash
grep -aoE 'EXPECTED_OLD_BYTES_REGEX' "$PATCHED" | sort -u | wc -l
# Should print: 1 (one unique sequence appearing twice)
```

### Step 5: Update the PatchDef

Edit `cc-lib/cc_lib/claude_binary_patching.py`. Update `old`, `new`, possibly `anchor`, and bump `min_version` to the new version (we use the bump strategy — old bytes preserved in git history, see CLAUDE.md decision rationale).

### Step 6: Confirm via check, then apply

```bash
claude-binary-patcher check <patch-name>
# Expect: unpatched (2 sites)

# Apply only this patch to verify it lands cleanly
claude-binary-patcher apply <patch-name>

# Confirm
claude-binary-patcher check <patch-name>
# Expect: applied
```

## Phase 2b: Investigate `missing` patches

When the anchor itself disappeared, run this decision tree:

### Decision tree

```
Anchor missing
│
├─ Was the anchor a function name (e.g., `function at(){...}`)?
│   └─ Search for the same flag/string with a different function name:
│       grep -aoE 'function \w+\(\)\{return \w+\("FLAG_NAME"' "$PATCHED"
│       └─ Found → just renamed → update anchor + old/new bytes
│
├─ Was the anchor a constant/identifier reference (e.g., `T.content.startsWith(e76)`)?
│   └─ The constant might be renamed but the structure intact:
│       grep -aoE 'T\.content\.startsWith\(\w+\)' "$PATCHED"
│       └─ One match → it's the renamed constant → update bytes
│
├─ Is the underlying feature still present?
│   └─ Compare counts of related strings:
│       grep -ac "feature_string" "$ORIG"      # In 2.1.126 originals
│       grep -ac "feature_string" "$PATCHED"   # In new version
│       ├─ Same count (or close) → feature present, just minified differently
│       │   → search for new minified names, redesign anchor
│       └─ Zero in new → feature REMOVED → see "Mark obsolete" below
│
└─ Architectural change (refactor — feature present but flow restructured)?
    └─ Spawn parallel investigation subagents (see "Investigation subagents")
```

### Investigation subagents

When the picture isn't clear from quick greps, spawn 3-4 unrestricted Opus subagents in parallel across vectors. Run them via the `Agent` tool with `subagent_type: "unrestricted-worker"` and `model: "opus"`.

| Vector | What to investigate |
|---|---|
| **Binary diff** | Truly removed or just renamed/relocated? Compare string sets between OLD and NEW binaries. Search for related Statsig flags and feature strings. Empirical, ground-truth-first. |
| **GitHub** | Search `anthropics/claude-code` issues + PRs + commits between the previous and current tag. Pull release notes via `gh release view`. Check community forks (e.g., Piebald-AI/tweakcc) for similar patches. |
| **Anthropic public** | Search docs.anthropic.com / code.claude.com / engineering blog / changelog / Discord. Often silent for stealth releases — note the silence as evidence. |
| **Community** | Reddit (r/ClaudeAI), Hacker News, peer dev blogs (claudefa.st, giuseppegurgone, simonw). |
| **Source mirror** | `~/claude-code-best/` — TypeScript reference. Use as structural guide. Note: mirror lags upstream and may have its own divergent bug fixes. |

#### Subagent prompt template

```
Empirical investigation of [specific question].

# Background
[What was, what changed — concrete byte-level facts]

# The question
[ONE clear question with verdict format]

# Sources
1. /Users/chris/.claude-workspace/binary-patcher/originals/<old-version> (vanilla old binary)
2. /Users/chris/.local/share/claude/versions/<new-version> (current binary)
3. /Users/chris/claude-code-best/ (TypeScript source mirror)
4. ~/.claude/projects/<project>/<sid>.jsonl (recent session for empirical traces)

# Output format
Under 500 words. Sections:
**Verdict**: ✅ / ⚠️ / ❌
**Empirical evidence**: quote actual bytes / function bodies, cite paths + offsets
**Open questions**: what remains uncertain

Don't speculate. Cite everything.
```

## Phase 3: Apply + empirical verify

After all `changed` and `missing` patches are addressed:

```bash
# Apply everything (pre-flight will refuse if any RequiredSetting is unsatisfied)
claude-binary-patcher apply --all

# Confirm final state
claude-binary-patcher check --all
```

For each patch with significant runtime behavior, verify in a **fresh `claude-exec` session** (not the current one — the running process has the pre-loaded binary in memory):

| Patch family | How to verify |
|---|---|
| `hook-ask-no-override` | Edit `.claude/settings.json` in a session with auto-mode + safetyCheck — hook ask should not be silently overridden |
| `mcp-array-content-to-string` | Call slack-style MCP tool with content array (no structuredContent) — should render JSON instead of blank |
| `reject-show-comment` (legacy) | Reject a Bash with a comment — comment should render (in 2.1.128+ vanilla is silent so the patch became obsolete; verify if applicable to your version) |
| `statusline` | Open a session — bottom statusline should wrap multi-line, not truncate with `…` |
| `inject-searching-past-context-prompt` | Ask "what do you know about my preferences from past sessions?" — Claude should spontaneously `Grep` the project memory dir |
| `scratchpad` | System prompt should include "## Scratchpad Directory" with a path; the dir gets created on first write |
| `show-subagent-prompt-tools-response` | Spawn a subagent with `Ctrl+R` verbose mode active — output should expand showing prompt + tool calls + response, not collapse to "Done" |

### Vanilla-vs-patched comparison protocol

If a patch's behavior looks wrong in the patched binary, **don't assume the patch is broken**. Anthropic may have changed the underlying behavior so the patch is now inert or unnecessary. Compare:

```bash
# 1. Restore vanilla
claude-binary-patcher restore

# 2. Test the same scenario in a fresh claude-exec session
# (user runs the test, reports verbatim what they see)

# 3. Re-apply patches
claude-binary-patcher apply --all

# 4. Same test in another fresh session

# 5. Compare:
#    - Both show same wrong/empty behavior   → Anthropic changed upstream → MARK OBSOLETE
#    - Vanilla works, patched doesn't        → Patch broke something → debug or revert
#    - Vanilla broken, patched fixes it      → Patch IS working, ship it
```

Real example from this project: in 2.1.128, `reject-show-comment` looked broken because rejection rendering was empty. Vanilla comparison revealed Anthropic had silently changed rendering to be empty in vanilla too — patch became obsolete (no bug to fix anymore).

## Phase 4: Mark obsolete patches

When a patch's underlying feature is removed/changed upstream such that the patch is no longer applicable:

1. Set `max_version='<previous-supported-version>'` on the PatchDef
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
    kind=PatchKind.FEATURE,
    anchor=b'function kI3(){return',
    old=b'function kI3(){return G_("tengu_session_memory",!1)}',
    new=b'function kI3(){return!0/*("tengu_session_memory")*/}',
    window=80,
    min_version='2.1.126',
    max_version='2.1.126',  # ← THIS LINE
    ...
)
```

3. Document the obsoletion in the Version Log entry with the upstream rationale (cite issues, postmortems, community signals — see real examples in the existing Version Log).

The `--all` apply will skip obsoleted patches with an informational warning. Don't delete the PatchDef — keep it for historical reference and for users on older versions.

## Phase 5: Documentation

Edit `cc-lib/cc_lib/claude_binary_patching.py` module docstring:

### Site Count Evolution table (for multi-version-history patches)

Add a new row to the table for this version. Only update columns for patches with multi-version history; new single-version patches go in the Version Log only.

```
    Version   statusline   mcp-array-content-to-string   write-session-summary   inject-searching-past-context-prompt   sm-compact
    2.1.126   2            2                             2                       2                                      —
    2.1.128   2            2                             0 (removed)             2                                      —
    <new>     ?            ?                             ?                       ?                                      ?
```

### Version Log entry (most important)

Add at the **top** of `Version Log::` (newest first). Date in `YYYY-MM-DD`. Format:

```
    <new-version> (<YYYY-MM-DD>)
        <One-paragraph summary of upstream changes that drove the migration>

        Patch updates:
        - <patch-name>: <one-line summary of what changed>
        - <patch-name>: marked obsolete via ``max_version='<prev>'``. <Why>
        - ...

        <Optional footer: links to investigations, gists, related issues>
```

Real examples are in the existing Version Log — match that style. No journey residue (don't write "I refactored X" — write what landed).

### Patches section (alphabetical)

If you renamed a patch or substantially changed its strategy, update its docstring entry. For obsolete patches, append a note explaining WHY without removing the historical record of how it worked.

## Phase 6: Pre-commit + commit + PR

```bash
# Run pre-commit twice (auto-fixers may modify on first run)
uv run pre-commit run --files cc-lib/cc_lib/claude_binary_patching.py \
  || uv run pre-commit run --files cc-lib/cc_lib/claude_binary_patching.py

# Commit
git add cc-lib/cc_lib/claude_binary_patching.py
git commit -m "$(cat <<'EOF'
binary-patcher: re-derive patches for <version>

<One-paragraph summary>

Patch updates:
- <patch-name>: <change>
- <patch-name>: <change>
EOF
)"

# Push + open PR
git push -u origin feat/binary-patches-$ARGUMENTS
gh pr create --title "Binary patches for Claude Code $ARGUMENTS" --body-file scratch/pr-binary-patches-$ARGUMENTS.md
```

PR description structure (real examples in PR #102 and PR #106):
- TL;DR table: status per patch (`✅ Clean apply`, `✅ Re-derived: <change>`, `🚫 Obsolete (max_version=...)`)
- Per-patch deep-dives for non-trivial changes
- Empirical verification checklist
- Outcome summary (N applicable patches, M sites)
- Link to investigation gists if subagents were spawned

## Cheatsheet

### Common shell idioms

```bash
PATCHED=/Users/chris/.local/share/claude/versions/$ARGUMENTS
ORIG=~/.claude-workspace/binary-patcher/originals/<previous-version>

# Patch status
claude-binary-patcher check --all
claude-binary-patcher check <patch-name>

# Find context around an anchor (perl handles binary files cleanly)
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(.{120}ANCHOR_REGEX.{200})/sg) { print "$&\n\n"; last }
' "$PATCHED"

# Find a function definition
perl -e '
my $data = do { local $/; open(my $fh, "<:raw", $ARGV[0]) or die; <$fh> };
while ($data =~ /(function NAME\([^)]*\)\{.{0,400}?\})/sg) { print "$1\n"; last }
' "$PATCHED"

# Compare feature presence between versions
echo -n "old: "; grep -ac "feature_string" "$ORIG"
echo -n "new: "; grep -ac "feature_string" "$PATCHED"

# Verify same-length byte budget
echo -n "Old: "; printf 'old_bytes_here' | wc -c
echo -n "New: "; printf 'new_bytes_here' | wc -c

# Verify both __BUN sites have identical bytes (should print 1)
grep -aoE 'pattern' "$PATCHED" | sort -u | wc -l

# Find Statsig flag references
grep -aoE '\b\w+\("flag_name"[^)]*\)' "$PATCHED" | sort -u

# Find all JSON.stringify aliases (for byte-budget scope decisions)
grep -aoE '\b\w+=JSON\.stringify\b' "$PATCHED" | sort -u
```

### Anchor selection rules

- **Anchor must survive the patch.** If the anchor is part of `old`, post-patch detection breaks (status reports `missing`). Use stable post-context substrings.
- **String literals are most stable** (flag names, prop names, error messages, JSX prop sets).
- **Minified function names are LEAST stable** (typically renamed every release).
- **Examples that work**:
  - `b'"contentArray"'` (string literal)
  - `b'){let Y;if(_[5]===Symbol.for("react.memo_cache_sen'` (React Compiler memoization marker, post-context)
  - `b'_8.createElement(IL5,{progressMessages:_,tools:q,verbose:K})'` (JSX prop set, structurally stable across T→K outer substitution)
- **Examples that DON'T work** (avoid):
  - `b'q.content.startsWith(UZH)'` when the patch replaces `UZH` with `"Z"` — anchor disappears post-patch
  - Bare 2-3 character minified IDs without surrounding context — too many false matches

### Patch-strategy patterns

| Strategy | When to use | Example |
|---|---|---|
| **Falsify a check** | Routing decision routes content to a wrong renderer | `T.content.startsWith(e76)` → `T.content.startsWith("Z")` |
| **Short-circuit gate body** | Statsig gate cached false, blocking a feature | `function at(){return G_("tengu_scratch",!1)}` → `function at(){return!0/*scratchpad always*/}` |
| **Variable substitution** | JSX block conditionally hides info | 4× `T` → `K` in show-subagent JSX (T was isTranscriptMode, K is verbose) |
| **Discriminator preservation** | Patch breaks a related consumer | `return T` → `return q!=="ide"?NH(T):T` (preserves IDE plugin's array-shape consumer) |

## Antipatterns

- **Don't apply --all blindly without `check` first.** The 4-state table tells you what's needed. Skip patches reporting `applied`; investigate `changed`/`missing`.
- **Don't add NEW patches in a version-migration PR.** That's scope creep. Re-derive existing patches; new patches go in their own PR.
- **Don't skip vanilla-vs-patched comparison when behavior is ambiguous.** Real lesson: `reject-show-comment` looked broken on 2.1.128 but vanilla was also "broken" — Anthropic had silently changed the rendering. Saved by the comparison.
- **Don't trust subagent verdicts without empirical follow-up.** Static analysis can be incomplete. After a subagent says "this should work," still test in a fresh session.
- **Don't trust a 100-character preview when verifying constants.** Real lesson: I assumed `UZH` and `EK6` were the same constant because their first 100 chars matched. They diverged after 152 chars (`STOP what you...` vs `To tell you how...`). Always extract the FULL constant content before equating.
- **Don't add patches based on flawed routing analysis.** Real lesson: `reject-show-comment-dispatcher` was added based on misunderstanding the dispatcher's branches. Always trace the ACTUAL routing through the function, not the assumed routing.
- **Don't keep dead patches "just in case."** If Anthropic removed the underlying feature, mark obsolete with `max_version`. Don't try to "make it work" when there's nothing to fix.
- **Don't write journey residue in docstrings or PR descriptions.** "I refactored X" / "Following PR review..." belong in commit messages, not in code documentation. Documents the state, not the path.
- **Don't forget pre-commit before commit.** Failed pre-commit means a wasted commit message. Run `pre-commit run --files <changed>` first, fix, then commit.

## Quick reference: phases checklist

- [ ] Phase 0: `claude-version-manager fetch <ver> --activate` + read changelog + branch
- [ ] Phase 1: `claude-binary-patcher check --all` + categorize statuses
- [ ] Phase 2a: re-derive each `changed` patch (context → renames → same-length new bytes → verify)
- [ ] Phase 2b: investigate each `missing` patch (rename / refactor / removed → spawn subagents if non-trivial)
- [ ] Phase 3: apply + empirical-verify in fresh `claude-exec` session (vanilla comparison if ambiguous)
- [ ] Phase 4: mark obsoleted patches with `max_version`
- [ ] Phase 5: update Site Count Evolution table + add Version Log entry
- [ ] Phase 6: pre-commit + commit + push + PR
