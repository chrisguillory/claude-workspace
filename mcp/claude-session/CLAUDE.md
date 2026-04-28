# Claude Session MCP - Development Guide

## Session File Analysis

Session files are JSONL at `~/.claude/projects/<encoded-path>/<session-id>.jsonl`.

### Quick Reference: Record Types

Record types found in session files (schema v0.2.9):

| Type | Purpose | Key Fields | Inherits BaseRecord |
|------|---------|------------|---------------------|
| `user` | User messages and inputs | `uuid`, `timestamp`, `sessionId`, `cwd`, `message`, `parentUuid` | Yes |
| `assistant` | Claude responses | `uuid`, `timestamp`, `sessionId`, `cwd`, `message`, `usage`, `model` | Yes |
| `summary` | Session summaries | `summary` (text), `leafUuid` | **No** (no uuid/timestamp/sessionId) |
| `system` | Internal events (see subtypes below) | `uuid`, `timestamp`, `sessionId`, `cwd`, `parentUuid`, `systemType` | Yes |
| `file-history-snapshot` | Document/file state tracking | `messageId`, `snapshot`, `isSnapshotUpdate` | **No** |
| `queue-operation` | Queue management events | `operation`, `timestamp`, `sessionId`, `content` | **No** (no uuid) |
| `custom-title` | User-defined session names (v0.1.9) | `customTitle`, `sessionId`, `timestamp` | **No** |
| `progress` | Long-running operation tracking (v0.2.4) | `data`, `parentToolUseID`, `toolUseID` | **No** (own schema) |
| `pr-link` | PR creation tracking (v0.2.9) | `prNumber`, `prUrl`, `prRepository` | **No** |
| `saved_hook_context` | Persisted hook output (v0.2.9) | `content`, `hookName`, `hookEvent`, `toolUseID` | **No** (own schema) |

**System Record Subtypes** (all have `type='system'`, differentiated by `subtype` field):

| Subtype | Purpose | Additional Fields |
|---------|---------|-------------------|
| `local_command` | Local shell/CLI operations | `content`, `level`, `slug`, `isMeta` |
| `compact_boundary` | Session compaction markers | `content`, `compactMetadata` (trigger, preTokens), `logicalParentUuid` |
| `microcompact_boundary` | Lightweight compaction markers (v0.2.4) | `content`, `microcompactMetadata` |
| `api_error` | Claude API failures | `error`, `retryInMs`, `retryAttempt`, `maxRetries`, `cause` |
| `informational` | General system notifications | `content`, `level` |
| `turn_duration` | Turn timing data (v0.2.0) | `durationMs` |
| `stop_hook_summary` | Hook execution summaries (v0.2.6) | `hookCount`, `hookInfos`, `hookErrors`, `stopReason` |
| `bridge_status` | Remote-control bridge status (v0.2.14) | `content`, `url`, `isMeta` |
| `away_summary` | Session recap for `/recap` (v0.2.22, Claude Code 2.1.108+) | `content` |

### Quick Reference: Attachment Types

Records with `type='attachment'` (discriminated by `attachment.type` field). Attachments inject non-message content into the conversation (hook output, queued commands, IDE state, file references, etc.):

| Attachment Type | Purpose | Key Fields |
|-----------------|---------|------------|
| `companion_intro` | Companion creature introduction (`/buddy` feature) (v0.2.20) | `name`, `species` |
| `mcp_instructions_delta` | MCP server instruction changes (added/removed tool descriptions) (v0.2.20) | `addedNames`, `addedBlocks`, `removedNames` |
| `deferred_tools_delta` | Deferred-tool registration changes for the tool-search system (v0.2.27, Claude Code 2.1.120+) | `addedNames`, `addedLines`, `removedNames` |
| `task_reminder` | Task-list reminder injected into conversation (v0.2.22) | `content` (Sequence[TaskReminderItem]), `itemCount` |
| `hook_success` | Hook executed successfully — stdout/exit injected (v0.2.22) | `hookName`, `toolUseID`, `hookEvent`, `content`, `stdout`, `stderr`, `exitCode`, `command`, `durationMs` |
| `hook_blocking_error` | Hook returned a blocking error; tool execution halted (v0.2.22) | `hookName`, `toolUseID`, `hookEvent`, `blockingError` |
| `hook_non_blocking_error` | Hook returned a non-blocking error; tool continued (v0.2.22) | `hookName`, `toolUseID`, `hookEvent`, `stderr`, `stdout`, `exitCode`, `command`, `durationMs` |
| `hook_additional_context` | Hook returned context to inject into the conversation (v0.2.22) | `content` (str \| Sequence[str]), `hookName`, `toolUseID`, `hookEvent` |
| `queued_command` | User command queued while Claude was working (v0.2.22) | `prompt` (str \| Sequence), `commandMode`, `imagePasteIds` |
| `dynamic_skill` | Skill discovered at a non-default location (v0.2.22) | `skillDir`, `skillNames`, `displayPath` |
| `skill_listing` | Periodic listing of available skills (v0.2.22) | `content`, `skillCount`, `isInitial` |
| `nested_memory` | Nested CLAUDE.md / memory file discovered under cwd (v0.2.22) | `path`, `content` (NestedMemoryContent), `displayPath` |
| `file` | Generic file reference attached to the conversation (v0.2.22) | `filename`, `content` (FileAttachmentContent), `displayPath` |
| `edited_text_file` | File edited externally; snippet with line numbers shown (v0.2.22) | `filename`, `snippet` |
| `opened_file_in_ide` | File opened in the connected IDE (VSCode/JetBrains) (v0.2.22) | `filename` |
| `selected_lines_in_ide` | Lines selected in the connected IDE (v0.2.22) | `ideName`, `lineStart`, `lineEnd`, `filename`, `content`, `displayPath` |
| `diagnostics` | IDE diagnostics injected into the conversation (v0.2.22) | `files` (Sequence[DiagnosticFile]), `isNew` |
| `plan_file_reference` | Plan file content referenced in the conversation (v0.2.22) | `planFilePath`, `planContent` |
| `plan_mode_exit` | User exited plan mode; whether plan file was written (v0.2.22) | `planFilePath`, `planExists` |
| `compact_file_reference` | File reference injected after a compact operation (v0.2.22) | `filename`, `displayPath` |
| `command_permissions` | Tool allow-list applied by a slash command (v0.2.22) | `allowedTools` |
| `date_change` | Date rolled over mid-session; current date injected (v0.2.22) | `newDate` |
| `auto_mode` | Auto mode activation reminder (v0.2.22, Claude Code 2.1.111+) | `reminderType` |
| `auto_mode_exit` | Auto mode exited (v0.2.22, Claude Code 2.1.111+) | (none beyond `type`) |

### Quick Reference: Message Content Types

Content blocks within `message.content` arrays (discriminated by `type` field):

| Type | Description | Key Fields |
|------|-------------|------------|
| `thinking` | Extended thinking blocks (not in context) | `thinking` (str), `signature` (str \| None) |
| `text` | Plain text from user or assistant | `text` (str) |
| `tool_use` | Tool invocations from assistant | `id`, `name`, `input` (typed or dict) |
| `tool_result` | Tool execution results from user | `tool_use_id`, `content` (str \| list \| None), `is_error` (bool \| None) |
| `image` | Base64 images from user | `source` (ImageSource with base64 `data`, `media_type`) |
| `document` | PDF/document uploads from user | `source` (DocumentSource with base64 `data`, `media_type`) |

### Field Glossary

Common fields and their meanings:

- **`uuid`**: Unique record identifier (not present on summary/file-history-snapshot/queue-operation records)
- **`timestamp`**: ISO 8601 timestamp of record creation
- **`sessionId`**: Session identifier (format: `019xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
- **`parentUuid`**: Links to preceding message; establishes conversation thread (null for root records)
- **`cwd`**: Working directory at time of record (absolute path)
- **`isSidechain`**: Message from an agent or subprocess (references `agent-{agentId}.jsonl`)
- **`agentId`**: Agent identifier; typed agents use `<type>-<hex>` format (e.g., `aprompt_suggestion-a1b2c3`)
- **`slug`**: Human-readable session identifier (e.g., "jiggly-churning-rabbit") - may change within session
- **`leafUuid`**: Most recent message in a conversation branch (used in summary records)
- **`compactMetadata`**: Session compression metadata (`trigger`: auto/manual, `preTokens`: token count before compaction)
- **`usage`**: Token consumption (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`)
- **`toolUseResult`**: Structured tool execution metadata (varies by tool type)

### Checking Token Usage

Token counts are in `message.usage` on assistant records:

```python
total_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
```

This is the **message context** only. Full context includes overhead:

| Component | Typical Size | Notes |
|-----------|--------------|-------|
| System prompt | ~3k | Base Claude Code instructions |
| System tools | ~16k | Built-in tools |
| MCP tools | 0-50k+ | Varies by configured MCPs |
| Memory files | varies | CLAUDE.md files |
| Messages | varies | Conversation history |

Run `/context` in Claude Code for authoritative breakdown.

### Finding Compaction Events

```bash
rg -n "compact_boundary" <session-file>.jsonl
```

Each has `compactMetadata.trigger` (usually "manual") and `compactMetadata.preTokens`.

### "Prompt is too long" Errors

When the API rejects a prompt, Claude Code logs a synthetic response:

```json
{
  "type": "assistant",
  "message": {
    "model": "<synthetic>",
    "content": [{"type": "text", "text": "Prompt is too long"}]
  }
}
```

This indicates context exceeded ~200k. Check preceding records for actual token counts.

### Cache Invalidation

`cache_read_input_tokens` drops significantly when:
- User edits a previous message (branches conversation)
- User goes back to retry from earlier point
- Session structure changes

This is NOT compaction - the tokens are still there, just not cached.

## Session Search

Searching session files is non-trivial due to content characteristics.

### Noise Sources

1. **Tool results**: Contain base64 images, previous search outputs, large file contents
2. **Self-referential**: Your search queries appear in session history as searchable content
3. **Nested JSON**: Content is JSON within JSONL - naive regex matches structural elements

A simple `rg "topic" ~/.claude/projects/` will match tool outputs, not just user intent.

### Layered Search Strategy

**Layer 1: Filter by record type** (eliminates 50-70% noise)
```bash
jq -r 'select(.type=="user") | .message.content' ~/.claude/projects/*/*.jsonl | rg -i "search term"
```

**Layer 2: Filter by content size** (exclude large pastes, base64)
```bash
jq -r 'select(.type=="user") | .message.content | select(type=="string" and length < 5000)' \
  ~/.claude/projects/*/*.jsonl | rg -i "search term"
```

**Layer 3: Session-level grouping** (examine matching files)
```bash
for file in $(rg -l "topic" ~/.claude/projects/*/*.jsonl 2>/dev/null); do
  echo "=== $file ==="
  jq -r 'select(.type=="user") | .message.content | if type=="string" then .[0:200] else .[0].text?[0:200] // "" end' "$file" 2>/dev/null | head -5
done
```

**Layer 4: Summary-first discovery** (fastest)
```bash
jq -r 'select(.type=="summary") | .summary' ~/.claude/projects/*/*.jsonl | rg -i "topic"
```

## Validation

```bash
./scripts/validate_models.py --fast             # Quick pass/fail
./scripts/validate_models.py --errors           # Grouped error details
./scripts/validate_models.py -e path/to/file    # Single file investigation
```

Validates all session files against Pydantic models. 100% pass rate expected.

To inspect a failing record:
```bash
cd ~/.claude/projects && find . -name "<session-id>.jsonl" -exec sed -n "<line>p" {} \; | jq .
```

## Analyzing Session Token Usage

**Always use Pydantic models** for session analysis - never raw `json.loads()` without validation. The models provide type safety, IDE autocomplete, and catch schema changes early:

```python
from pathlib import Path
from claude_session.schemas.session import validate_session_record, UserRecord, AssistantRecord, CompactBoundarySystemRecord
import json

# Parse session with Pydantic
records = []
with open(session_path) as f:
    for line in f:
        record = validate_session_record(json.loads(line))
        records.append(record)

# Find last compact_boundary
last_compact_idx = None
for i, record in enumerate(records):
    if isinstance(record, CompactBoundarySystemRecord):
        last_compact_idx = i
        print(f"Compact at {i}: preTokens={record.compactMetadata.preTokens}")

# Analyze records after compact
for record in records[last_compact_idx + 1:]:
    if isinstance(record, UserRecord) and record.message:
        for block in record.message.content:
            if block.type == 'tool_result':
                # block.content is the tool output
                pass
            elif block.type == 'text':
                # block.text is user message
                pass

    elif isinstance(record, AssistantRecord) and record.message:
        for block in record.message.content:
            if block.type == 'thinking':
                # block.thinking - NOT counted in context
                pass
            elif block.type == 'text':
                # block.text - assistant response
                pass
            elif block.type == 'tool_use':
                # block.input - Pydantic model, use .model_dump_json()
                pass
```

### Preserving Context Before Compact

Before running `/compact`, ask Claude to generate a handoff document:

> "Make a self-contained handoff artifact that the next AI can use to continue. Include executive summary, current state, pending work, and key decisions."

This produces a structured document (~12k chars) that's **larger and more useful** than the auto-generated compact summary (~9k chars). The handoff survives in session history as additional context.

The compact summary is stored in the first user message after `compact_boundary` as `message.content` (string, not list).

### Estimating Message Tokens

Claude's tokenizer differs from tiktoken, but rough estimates:

| Content Type | In Context? | Estimate |
|--------------|-------------|----------|
| Thinking blocks | NO | Generated output, not input |
| Tool results | ~45% | Truncated/compressed |
| User/assistant text | ~100% | Full content |
| Tool use inputs | ~100% | Full content |

## Operational Patterns

### Finding Session Files

Locate a session file by ID prefix:

```bash
find ~/.claude/projects -name "019b342b*.jsonl"
```

### Comparing Handoff vs Compact

Measure preservation strategy effectiveness:

```python
# Find handoff document (large assistant text before compact)
# Find compact summary (first user message after compact_boundary, stored as string)
# Compare char lengths - handoff typically ~25% larger
```

## Intercepting Claude Code API Traffic

Use mitmproxy to capture Claude Code's API traffic for observation-based schema development.

### Setup

```bash
# Install mitmproxy (if not already installed)
brew install mitmproxy

# Ensure captures directory exists
mkdir -p captures
```

### Capturing Traffic

```bash
# Kill any existing proxy
pkill -f mitmdump 2>/dev/null

# Clear old captures (now session-based directories)
rm -rf captures/*/

# Start the proxy (from repo root)
# --set stream=false ensures complete SSE capture for streaming responses
mitmdump -p 8080 -s scripts/intercept_traffic.py --set stream=false

# In another terminal, run Claude through the proxy
HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude
```

**Session Correlation**: The proxy now correlates traffic with Claude Code sessions. Captures go to `captures/<session_id>/` directories. This requires:
- claude-workspace hooks configured in `~/.claude/settings.json`
- `~/.claude-workspace/sessions.json` exists (written by hooks)

### Zero-MCP Capture (Vanilla Baseline)

To capture traffic without MCP tool overhead, temporarily disable MCP servers before running Claude through the proxy. This gives a clean baseline of Claude Code's core API usage.

### Analyzing Captures

Captures now include full metadata (headers, timing, auth info, rate limits, SSE events, session_id):

```bash
# View traffic summary (global log)
cat captures/traffic.log

# List session directories
ls -la captures/

# View a session's manifest
cat captures/<session-id>/manifest.json

# List captures for a specific session
ls -la captures/<session-id>/

# Verify session_id in captures
jq '.session_id' captures/<session-id>/req_*.json | head

# Inspect a messages request (body is nested under .body.data for JSON)
cat captures/<session-id>/req_*messages*.json | jq '{
  session_id,
  flow_id,
  method,
  path,
  model: .body.data.model,
  max_tokens: .body.data.max_tokens,
  messages_count: (.body.data.messages | length),
  tools_count: (.body.data.tools | length)
}'
```

### Output Structure

```
captures/
├── traffic.log              # Global summary log
├── <session-id-1>/          # One directory per Claude session
│   ├── manifest.json        # Session metadata from claude-workspace
│   ├── 001_req_*.json       # Request payloads (sequence-first for sorting)
│   ├── 001_resp_*.json      # Response payloads (pairs stay adjacent)
│   └── ...
├── <session-id-2>/
│   └── ...
└── unknown/                  # Traffic without session correlation
    └── ...
```

Note: Sequence-first naming (`NNN_req_*`, `NNN_resp_*`) keeps request/response
pairs adjacent when sorted alphabetically.

### Previously Discovered Endpoints

| Endpoint | Purpose |
|----------|---------|
| `api.anthropic.com/v1/messages` | Main conversation API |
| `api.anthropic.com/v1/messages/count_tokens` | Token counting (called per tool!) |
| `api.anthropic.com/api/event_logging/batch` | Telemetry |
| `api.anthropic.com/api/hello` | Health check |
| `statsig.anthropic.com` | Feature flags |

See `docs/intercepting-claude-api.md` for detailed exploration notes.

## Claude Code Binary Analysis

How to investigate Claude Code's internal behavior by examining its compiled binary. This methodology was used to discover the settings.json env allowlist, credential reader memoization, and init sequence documented in the README.

### Binary Location & Structure

Claude Code is distributed as a Node.js Single Executable Application (SEA):

| Property  | Value (v2.1.44)                                           |
|-----------|-----------------------------------------------------------|
| Symlink   | `~/.local/bin/claude`                                     |
| Binary    | `~/.local/share/claude/versions/<version>`                |
| Format    | Mach-O arm64 (macOS), ~175MB                              |
| JS bundle | Embedded at ~59.6MB offset, ~10MB minified (~7,474 lines) |

### Extraction Patterns

```bash
# Extract all readable strings
strings $(which claude) > /tmp/claude-strings.txt

# Find all env var reads
strings $(which claude) | grep -oE 'process\.env\.[A-Z_0-9]+' | sort -u

# Search for specific behavior
strings $(which claude) | grep -i 'CLAUDE_CODE_OAUTH'

# Find keychain-related code
strings $(which claude) | grep -i 'Claude Code-credentials'
```

The JS is heavily minified — function names are 2-4 character identifiers (`dB`, `b7T`, `$90`, `KTT`). These change between versions. Search by known string literals (env var names, error messages, keychain service names) rather than function names.

### Unpacking the JS Bundle

When `strings` isn't enough and you need to walk actual JS (e.g., to trace Commander option registrations, read `hideHelp` chains, inspect the router), use the wrapper script:

```bash
# Unpacks a COPY of the installed binary — installed claude is untouched
scripts/claude-unpack-binary.py

# Prints the unpacked JS path on stdout; pipe straight into analysis
scripts/claude-unpack-binary.py | xargs grep -n hideHelp
```

> [!CAUTION]
> `npx tweakcc unpack <binary>` rewrites its input **in place**. Running it directly against `~/.local/share/claude/versions/<version>` will overwrite the installed executable with extracted JS text, bricking `claude`. Always unpack a copy. The wrapper script enforces this by copying first.

If you need to run tweakcc manually for some reason:

```bash
# Copy first — never pass the installed binary path directly to tweakcc
cp "$(readlink -f "$(which claude)")" /tmp/claude-copy
npx tweakcc@latest unpack /tmp/claude-copy    # rewrites /tmp/claude-copy in place
# /tmp/claude-copy now contains the unpacked JS
```

Recovery if you clobber the installed binary: restore from `~/.claude-workspace/binary-patcher/originals/<version>`, or re-fetch via `claude-version-manager fetch <version>` (SHA-256 verified).

### Investigation Workflow

1. **Start with the symptom** — identify an observable behavior (error message, env var name, API endpoint)
2. **Search for string literals** — `strings | grep` for known constants related to the behavior
3. **Extract surrounding context** — use `grep -A5 -B5` to see code near the match
4. **Trace function calls** — follow variable references through the minified code
5. **Identify patterns** — look for `memoize`, `cache`, allowlists, and init ordering

### Key Discoveries Reference

Findings from binary analysis that affect how tools and sessions interact with Claude Code:

| Discovery                                     | Impact                                      | Details                                             |
|-----------------------------------------------|---------------------------------------------|-----------------------------------------------------|
| Credential reader is lodash-memoized          | Auth set after init is invisible            | See README: Authentication Architecture             |
| Settings.json env has security allowlist      | Credentials can't go in settings.json       | See README: Settings.json Env Allowlist             |
| Two-phase env loading                         | `/status` can show tokens that auth ignores | `$90` (filtered, early) vs `KTT` (unfiltered, late) |
| `CLAUDE_CONFIG_DIR` creates isolated keychain | Separate credentials per config dir         | SHA-256 hash suffix on keychain service name        |

### Version-Specific Notes

Minified identifiers change between versions. When updating analysis for a new version:

1. Re-extract strings: `strings ~/.local/share/claude/versions/<new-version> > /tmp/claude-strings-new.txt`
2. Search by known constants (env var names, service names), not old function names
3. Verify the allowlist hasn't changed: search for the list of allowed env var names
4. Check if credential reader memoization is still present: search for `memoize` near `CLAUDE_CODE_OAUTH_TOKEN`

## Checking Claude Code Changelog

When investigating new features or schema changes:

```bash
# Terminal summary
claude changelog

# Full changelog from GitHub
gh api repos/anthropics/claude-code/contents/CHANGELOG.md --jq '.content' | base64 -d

# Search for specific feature
gh api repos/anthropics/claude-code/contents/CHANGELOG.md --jq '.content' | base64 -d | rg -i "feature_name"
```

## MCPSearch Tool (Claude Code 2.0.73+)

MCPSearch is an undocumented built-in tool for dynamic MCP tool discovery, introduced in v2.0.73 (Dec 18, 2025) and default-on in v2.1.x.

**Schema additions (v0.2.2):**
- `MCPSearchToolInput` - Query-based tool discovery with `select:` prefix support
- `ToolReferenceContent` - Tool reference blocks in search results
- `ToolUseCaller` - Caller metadata on tool_use blocks

**Session file identification:**
```json
{"type": "tool_use", "name": "MCPSearch", "input": {"query": "select:mcp__...", "max_results": 1}}
```

**For user-facing information**, including how to disable, known issues, and accuracy concerns, see `docs/mcpsearch-guide.md`.

## Schema Validation Fix Workflow

When Claude Code updates or new session data introduces schema drift:

### 1. Validate
```bash
./scripts/validate_models.py --fast             # Quick pass/fail
./scripts/validate_models.py --errors           # Grouped error details
./scripts/validate_models.py -e path/to/file    # Single file investigation
```

### 2. Research & Inspect (parallel)

**Run these concurrently** when errors are found or `CLAUDE_CODE_MAX_VERSION` is behind `claude --version`:

**a) Research changelog delta** (launch as unrestricted-worker subagent):
- Fetch official changelog: `gh api repos/anthropics/claude-code/contents/CHANGELOG.md --jq '.content' | base64 -d`
- Fetch community changelog: `https://www.claudelog.com/claude-code-changelog/`
- Focus on versions between `CLAUDE_CODE_MAX_VERSION` (in models.py) and current `claude --version`
- Summarize: new features, new tools, behavioral changes, anything schema-relevant
- Map each validation error to its likely feature origin (e.g., "bridge_status → /remote-control feature")

**b) Inspect failing records** (in main context):
```bash
cd ~/.claude/projects && find . -name "<session-id>.jsonl" \
  -exec sed -n "<line>p" {} \; | jq .
```

The research context informs better fixes: knowing a field is from a major feature (model carefully) vs. a one-off patch (optional field) changes the modeling approach.

### 3. Fix models
Edit `claude_session/schemas/session/models.py`. Common fixes:
- New optional fields: `fieldName: Type | None = None`
- New record/tool types: add model class + wire into discriminated union + add dispatch branch in `validate_session_record()`
- Expanded Literals: add new values to existing `Literal[...]` types
- Union ordering: ensure more-specific types come before less-specific

**Rules:**
- Only model fields/values actually observed in data (no speculation)
- Always use fully typed models, never `dict` fallbacks
- Minimize churn -- don't reorganize existing fields

**Schema is the source of truth — not the session data.**

When a validation failure is caused by Claude Code (or an underlying model) emitting **incorrect data** — e.g., a hallucinated field name, a typo, a malformed value — the fix is to **patch the offending JSONL record**, not to loosen the model to accept the malformed shape. Accommodating hallucinations in the schema:

- Is a form of schema lying — the type signature no longer reflects the real tool/API contract.
- Masks future occurrences of the same hallucination (valid-looking records with bogus data).
- Degrades the "100% pass rate" signal into "100% pass after we stopped enforcing things."
- Creates bad precedent — every model misfire becomes a candidate for schema loosening.

Reserve schema change for observed **intended** structural evolution (new fields Anthropic shipped, new enum values added, new record types). If a record is failing because it's *wrong*, fix the data:

```python
# Example: one-off field-name hallucination (command→pattern in Grep input)
# surgical jq/python edit to the specific record, preserving all other fields
```

`AliasChoices` / alternate field names are reserved for legitimate cases:
- Python-keyword conflicts (`StatusChange.from_` aliasing `from`)
- Non-identifier JSON keys (`GrepToolInput.dash_n` aliasing `-n`)

They are **not** for accommodating client-side bugs in the upstream emitter.

### 4. Verify 100% pass rate
```bash
./scripts/validate_models.py --fast
```
Iterate steps 2-4 until all records validate. Multiple rounds are normal.

### 5. Bump version (ONLY after 100% pass rate)
Update the version constants in `models.py`:
```python
SCHEMA_VERSION = '0.2.XX'
CLAUDE_CODE_MAX_VERSION = '2.1.XX'
```
Add a changelog line to the module docstring header.

### 6. Commit
```bash
uv run pre-commit run --all-files
git add claude_session/schemas/session/models.py
git commit -m "Schema vX.Y.Z: Fix validation for Claude Code X.Y.Z"
```

## Model Definition Ordering

Use composite-first (top-down) ordering per root CLAUDE.md's Module Organization section. Define `SessionArchive` before its constituent types (`AgentFileEntry`, `PlanFileEntry`, etc.). Use `from __future__ import annotations` to enable forward references.

## Tool Usage Rules

### Never Silently Fall Back on Tool Failures

**CRITICAL:** When the user explicitly requests a specific tool (e.g., Perplexity for research) and that tool is unavailable, **DO NOT silently fall back to an alternative** (e.g., WebSearch). Instead:

1. **Hard fail** - Tell the user the tool is not available
2. **Ask them to enable it** - e.g., "Perplexity MCP is not connected. Please run `/mcp reconnect perplexity`"
3. **Wait for their instruction** - Do not proceed with a substitute

This applies especially to:
- `mcp__perplexity__*` tools - Never substitute with WebSearch
- Any MCP tool the user explicitly names

**Why:** The user chose a specific tool for a reason. Silent fallback wastes their time with inferior results and hides the real problem (tool not connected).