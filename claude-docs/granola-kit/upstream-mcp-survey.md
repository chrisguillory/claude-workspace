# Granola Official MCP — Survey

**Survey date:** 5/14 PT
**Survey method:** OAuth-protected MCP at `https://mcp.granola.ai/mcp` + cross-referenced public documentation
**Author:** `upstream-mcp` teammate on the `granola-kit` migration
**Account used:** Chris Guillory (paid Mainstay org plan; obscures free-tier gating)

**Confidence labels used throughout:**
- **OBSERVED** — captured from a live, authenticated session
- **DOCS** — sourced from Granola official docs or third-party doc/integrators (Scalekit, ClawHub, joelhooks/granola-cli) — verbatim where shown
- **INFERRED** — best guess, called out explicitly

> Section status: portions marked **OBSERVED** require completion of the OAuth handshake (`/mcp` from an interactive Claude Code session). Sections currently filled only from **DOCS** will be upgraded on the next survey pass after authentication.

---

## Setup

### Install path

```bash
claude mcp add granola-official --scope user --transport http https://mcp.granola.ai/mcp
```

This succeeds without auth (transport handshake only) — the MCP appears as `connected` in `claude mcp list` but is functionally `needs-auth` (only two scaffolding tools exposed). Compare to the Sentry MCP, which uses the same gated pattern.

### OAuth discovery

Verbatim `https://mcp.granola.ai/.well-known/oauth-protected-resource`:

```json
{
  "resource": "https://mcp.granola.ai/mcp",
  "authorization_servers": ["https://mcp-auth.granola.ai"],
  "bearer_methods_supported": ["header"]
}
```

Verbatim `https://mcp-auth.granola.ai/.well-known/oauth-authorization-server`:

```json
{
  "authorization_endpoint": "https://mcp-auth.granola.ai/oauth2/authorize",
  "client_id_metadata_document_supported": true,
  "code_challenge_methods_supported": ["S256"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "introspection_endpoint": "https://mcp-auth.granola.ai/oauth2/introspection",
  "issuer": "https://mcp-auth.granola.ai",
  "jwks_uri": "https://mcp-auth.granola.ai/oauth2/jwks",
  "registration_endpoint": "https://mcp-auth.granola.ai/oauth2/register",
  "scopes_supported": ["email", "offline_access", "openid", "profile"],
  "response_modes_supported": ["query"],
  "response_types_supported": ["code"],
  "token_endpoint": "https://mcp-auth.granola.ai/oauth2/token",
  "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"]
}
```

Observations:
- Full OAuth 2.1 with Dynamic Client Registration (DCR), PKCE (S256), refresh tokens. `client_id_metadata_document_supported: true` — clients identify by a metadata-doc URL rather than a registered client ID. Claude Code uses `https://claude.ai/oauth/claude-code-client-metadata`.
- Redirect URI for Claude Code: `http://localhost:3118/callback` — confirms browser-based flow.
- Scopes are stock OIDC (`email/openid/profile/offline_access`); the server does NOT advertise resource-specific scopes (e.g., `meetings:read`). All authorization happens at the resource level.
- 401 from `/mcp` includes `WWW-Authenticate: Bearer error="invalid_token", error_description="Authorization needed", resource_metadata="https://mcp.granola.ai/.well-known/oauth-protected-resource"` — textbook MCP-spec compliance.

### Server identity (OBSERVED at transport handshake)

```json
{
  "name": "granola-mcp",
  "title": "Granola",
  "icons": [{"src": "https://www.granola.ai/icon.png", "mimeType": "image/png"}],
  "version": "1.0.0",
  "websiteUrl": "https://www.granola.ai",
  "description": "The AI notepad for meetings. Connect to your meeting history and use your conversation context to get things done."
}
```

Note: official server name is literally `granola-mcp` (no `-official` suffix). The `-official` is just Chris's local registration alias to disambiguate from his existing `granola` MCP.

### Capabilities

```json
{"hasTools": true, "hasPrompts": false, "hasResources": false, "hasResourceSubscribe": false}
```

No MCP `resources` or `prompts` capability — tool-only server.

### Plan-tier observations

Per Granola docs ([DOCS](https://docs.granola.ai/help-center/sharing/integrations/mcp)):

| Capability | Free | Paid |
|---|---|---|
| Last 30 days of own notes | yes | yes |
| Notes shared with you | no | yes |
| Folder filtering (`list_meeting_folders`, `list_meetings(folder=...)`) | no | yes |
| Raw transcripts (`get_meeting_transcript`) | no | yes |
| Rate limit | undocumented | "~100 requests/minute baseline" (per a third-party summary; not verified) |

Chris is on a paid Mainstay org plan — survey results reflect the paid surface, and free-tier behavior is INFERRED from docs, not observed.

---

## Tool surface (verbatim)

> **Status: partially-OBSERVED.** Two scaffolding tools confirmed verbatim from the unauthenticated MCP session. The six substantive tools below are taken from Granola docs + third-party documentation (Scalekit, ClawHub, joelhooks/granola-cli, Granola docs) and will be upgraded to OBSERVED schema dumps on the next survey pass.

### Tools visible pre-auth (OBSERVED)

When unauthenticated, the official MCP exposes exactly two tools:

- `authenticate`
- `complete_authentication`

These are the standard Claude Code OAuth scaffolding pattern (same as Sentry). Naming convention: snake_case verbs. No schemas captured yet.

### Tools visible post-auth (DOCS, to be upgraded)

#### `query_granola_meetings`

**Description (DOCS):** "Query meeting notes using natural language. Returns a synthesized answer with inline citations to source meetings."

**Parameters (DOCS):**
- `query: string` — required. Natural-language question.
- `document_ids: uuid[]` — optional. Scope the query to specific meetings by UUID.

**Plan:** Free + Paid (free limited to last 30 days)

**Sample call (DOCS, Scalekit):**
```python
tool_input = {"query": "What decisions and follow-ups came out of last week's customer calls?"}
```

**Output:** Synthesized answer with inline citations to source meeting notes. Citation format not documented; needs OBSERVED capture.

**Doc-stated guidance:** "Prefer `query_granola_meetings` over list+get for natural language questions."

#### `list_meetings`

**Description (DOCS):** "List meetings within a time range — this_week, last_week, last_30_days, or a custom ISO date range. Returns titles and metadata."

**Parameters (DOCS):**
- `time_range: enum("this_week", "last_week", "last_30_days", "custom")` — optional. Default not documented.
- `custom_start: ISO 8601 date` — optional. Required if `time_range="custom"`.
- `custom_end: ISO 8601 date` — optional. Required if `time_range="custom"`.

For paid plans: filters by folder ID also supported (parameter name not in docs — INFERRED to be `folder_id` or similar). Includes "notes shared with you" in results.

**Plan:** Free + Paid

**Returns (DOCS):** "meeting ID, meeting title, meeting date, attendees."

#### `get_meetings`

**Description (DOCS):** "Fetch AI-generated summary, private notes, attendees, and metadata for up to 10 meetings by UUID."

**Parameters (DOCS):**
- `meeting_ids: uuid[]` — required. Max length 10.

**Plan:** Free + Paid

**Returns (DOCS):** "meeting ID, meeting title, meeting date, attendees, private notes, enhanced notes."

#### `get_meeting_transcript`

**Description (DOCS):** "Retrieve the full verbatim transcript for a specific meeting by UUID. Returns spoken content only — not summaries or notes."

**Parameters (DOCS):**
- `meeting_id: uuid` — required. **Singular** parameter (not array).

**Plan:** Paid only.

**Returns (DOCS):** "meeting ID, raw transcript."

#### `list_meeting_folders`

**Description (DOCS):** "List folders you're a member of, including folder ID, title, description, and note count."

**Parameters (DOCS):** Not documented. INFERRED: no required params.

**Plan:** Paid only.

#### `get_account_info`

**Description (DOCS):** Returns "email and active workspace for the Granola account currently connected to this MCP session."

**Parameters (DOCS):** None.

**Plan:** Free + Paid.

**Returns (DOCS):** email + active workspace. INFERRED additional fields: workspace_id, role, plan_type — but unconfirmed.

### Annotations (NOT YET OBSERVED)

MCP `ToolAnnotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) require an authenticated `tools/list` response to confirm verbatim. Pre-auth `tools/list` returns 401, so annotations are unknown. INFERRED based on tool semantics:

| Tool | INFERRED annotations |
|---|---|
| `query_granola_meetings` | `readOnlyHint=True, openWorldHint=True` |
| `list_meetings` | `readOnlyHint=True, openWorldHint=True` |
| `get_meetings` | `readOnlyHint=True, openWorldHint=True` |
| `get_meeting_transcript` | `readOnlyHint=True, openWorldHint=True` |
| `list_meeting_folders` | `readOnlyHint=True, openWorldHint=True` |
| `get_account_info` | `readOnlyHint=True, openWorldHint=False` (returns stable account identity) |

**Key observation:** the entire surface is **read-only**. No `update`, `delete`, `create`, `add_to_folder`, `remove_from_folder`. This is a deliberate design choice by Granola — agents read, agents don't mutate.

---

## Side-by-side with granola-kit plan

Cross-reference: granola-kit's 19-tool roster (per [the plan](/Users/chris/.claude/plans/reactive-foraging-wreath-clone-019e26cb.md)) vs Granola official's 6 tools.

| Concept | granola-kit | Granola official | Notes |
|---|---|---|---|
| Natural-language semantic query | — | `query_granola_meetings` | **GAP** — granola-kit has nothing equivalent. Strongest absorb candidate. |
| List meetings | `list_meetings` (rich filters) | `list_meetings` (time-range enum + ISO custom) | granola-kit's surface is materially broader: `title_contains`, `case_sensitive`, `list_id`, `created_at_gte/lte`, `source=all/owned/shared`, `include_participants`, `limit`. Official's enum is more agent-friendly. |
| Get single meeting | `get_meeting(meeting_id)` | — (covered by `get_meetings` w/ array of 1) | granola-kit adds explicit singular. |
| Get multiple meetings | `get_meetings(meeting_ids)` (no max docs) | `get_meetings(meeting_ids, max 10)` | Official caps batch size; granola-kit doesn't. Worth considering whether to mirror cap. |
| Get transcript | `download_transcript` (writes file) | `get_meeting_transcript` (returns content) | Different return discipline (see below). |
| Get AI summary / notes | `download_summary` | covered by `get_meetings` | Granola packs summary inline; granola-kit returns file path. |
| Get user private notes | `download_note` | covered by `get_meetings` (returns "private notes" inline) | Same shape difference. |
| List folders | `list_folders` | `list_meeting_folders` | Naming: granola-kit's `list_folders` is shorter; official's prefixed `list_meeting_folders` is more explicit. |
| Get single folder | `get_folder(folder_id)` | — | granola-kit-only. |
| Create folder | `create_folder` | — | granola-kit-only (private-API write). |
| Add meeting to folder | `add_meeting_to_folder` | — | granola-kit-only. |
| Remove meeting from folder | `remove_meeting_from_folder` | — | granola-kit-only. |
| List workspaces | `list_workspaces` | — (folded into `get_account_info`?) | granola-kit returns multi-workspace; official surfaces only "active workspace". |
| Create workspace | `create_workspace` | — | granola-kit-only. |
| Delete workspace | `delete_workspace` | — | granola-kit-only (destructive). |
| Update meeting | `update_meeting(title, attendees)` | — | granola-kit-only. |
| Delete meeting | `delete_meeting` | — | granola-kit-only (soft-delete). |
| Restore meeting | `restore_meeting` | — | granola-kit-only. |
| List deleted meetings | `list_deleted_meetings` | — | granola-kit-only. |
| Resolve `/t/` or `/d/` URL → doc ID | `resolve_url` | — | granola-kit-only. |
| Account info | — | `get_account_info` | **GAP** — granola-kit has no equivalent. Quick add. |

### Headline counts

- **granola-kit plan:** 19 tools (3 write-meetings, 3 folder-writes, 1 workspace-create, 1 workspace-delete, 1 meeting-update, 1 meeting-delete, 1 meeting-restore + 11 reads)
- **Granola official:** 6 tools, **all read-only**, no mutation surface
- **Overlap:** ~4 (list_meetings, get_meetings, transcript fetch, folder listing)
- **granola-kit unique:** all writes + multi-workspace + URL resolver + private-notes-explicit + soft-delete management
- **Official unique:** `query_granola_meetings` (semantic), `get_account_info`

---

## Novel concepts worth absorbing

Ranked by impact on granola-kit's agent UX.

### 1. `query_granola_meetings` — semantic search with citations [HIGH IMPACT]

**Why it's novel:** Single tool call that takes a NL query, retrieves relevant meetings, synthesizes an answer, and includes inline citations. The agent doesn't need to enumerate-then-fetch-then-synthesize manually — the server does the retrieval/synthesis loop.

**Why it matters for granola-kit:** Today, an agent answering "what did Sam say about pricing last week?" must (a) list meetings, (b) filter by participant, (c) batch-fetch, (d) read each summary/transcript, (e) synthesize. That's 4+ tool calls and a large context burn. Official's `query_granola_meetings` collapses to 1 call.

**Absorption proposal — new tool: `search_meetings`**
- Contract: `search_meetings(query: str, document_ids: list[str] | None = None, time_range_days: int | None = None) -> SearchResult` where `SearchResult { answer: str, citations: list[Citation], used_meeting_ids: list[str] }`.
- Implementation tradeoff: Granola's server presumably runs this on their own retrieval stack (they own the corpus). granola-kit can't replicate "private API + native retrieval" cheaply. Two paths:
  - **(a) Wrap, don't implement:** add a tool that delegates to a local retrieval pipeline — use document-search MCP under the hood. Requires meetings to be pre-indexed; this is the "workspace interplay" the plan defers.
  - **(b) Defer + document:** declare `search_meetings` as "future feature requires document-search index"; in v1, granola-kit's `list_meetings` + `get_meetings` is the manual fallback.
- **Recommendation:** path (b) for v1. Path (a) is a follow-up that's already on the "defer" list (transcript ingestion pipeline). Note in `docs/possible-features.md`.

### 2. Time-range enum for `list_meetings` [MEDIUM IMPACT]

**Why it's novel:** Agents reach for `last_week`/`this_week`/`last_30_days` constantly. granola-kit forces them to compute ISO dates client-side.

**Absorption proposal:** Add a `time_range` parameter to granola-kit's `list_meetings`, parallel to (not replacing) the existing `created_at_gte`/`created_at_lte`. If both are supplied, document precedence (e.g., `time_range` wins).
- `time_range: Literal["this_week", "last_week", "last_30_days", "today", "yesterday", "all"] | None = None`
- Cleaner DX without sacrificing the existing flexibility.

**Recommendation:** YES, low-cost add. ~10 lines in `services/meetings.py`.

### 3. `get_account_info` [LOW-MEDIUM IMPACT]

**Why it's novel:** Single call to confirm "which account am I authenticated as?" is a universal agent need. granola-kit's closest equivalent is `list_workspaces` — which returns N workspaces, not "the current effective identity."

**Absorption proposal:** Add `get_account_info() -> AccountInfo { email: str, active_workspace_id: str, active_workspace_name: str, plan_type: str }`. Data is already encoded in the JWT (the existing `clients/auth.py` decodes it) — implementation is trivial.

**Recommendation:** YES, trivial add. ~15 lines. Map to a `get_account_info` tool in `tools/workspaces.py` (or a new `tools/account.py`).

### 4. Batch size cap on `get_meetings` [LOW IMPACT]

**Why it's worth noting:** Granola caps `get_meetings` at 10. granola-kit's batch endpoint chunks at 200. Both fine; the cap is a deliberate UX-vs-cost tradeoff. Granola probably caps to keep response sizes bounded (each meeting includes summary + private notes + attendees).

**Recommendation:** No change. granola-kit's 200-cap chunking + per-doc cache is strictly better for batch-fetch workflows.

### 5. `get_meeting_transcript` returns content inline (not file) [LOW IMPACT, REJECT]

**Why it's a design tension:** Official returns transcript content as part of the MCP response. granola-kit writes a tempfile and returns the path.

**Why granola-kit's pattern is correct for the workspace:** Transcripts can be huge (tens of KB to hundreds of KB) and consume context window. Writing to a tempfile + emitting a path is the workspace's "filesystem-as-shared-state" pattern (matches selenium-browser's `download_resource`, document-search's index workflow, etc.). Agent can `Read` the file lazily, or pipe to other MCPs (`document-search index_documents`).

**Recommendation:** REJECT mirroring. Keep file-write discipline. Possibly add an optional `inline: bool = False` param if direct inline retrieval is ever needed.

---

## Schema/naming patterns worth mirroring or rejecting

### Mirror

- **Singular `_id` for single-resource, plural `_ids` array for batch.** Official: `get_meeting_transcript(meeting_id)` vs `get_meetings(meeting_ids)`. granola-kit's plan already does this (`get_meeting` singular + `get_meetings(meeting_ids)` plural). **VALIDATED.**
- **Snake-case tool names with verb-first.** `list_*`, `get_*`, `query_*`. granola-kit matches.
- **Caller-friendly parameter names** (`meeting_ids`, not `document_ids`). granola-kit's plan does this rename — good.

### Reject

- **`list_meeting_folders` (longer prefix).** granola-kit's `list_folders` is cleaner since folders aren't ambiguous in granola-kit's domain. KEEP `list_folders`.
- **Capping batch size at 10.** Not needed in granola-kit since we chunk + cache.
- **Returning huge content inline (transcript).** REJECT — workspace pattern is file-on-disk.
- **All-read surface.** REJECT for granola-kit — its value prop is the mutation surface (delete/restore/update/folder-management) that Granola intentionally withholds.

### Hybrid / use-with-care

- **Plan-tier gating.** Granola gates `list_meeting_folders` and `get_meeting_transcript` behind paid. granola-kit hits the private API which has no plan gating from our viewpoint. KEEP unrestricted, but document in `docs/bleeding-edge.md` that we ignore plan tiers (the private API doesn't check them per-request, though Granola might add checks later).
- **`time_range` enum.** Mirror as an ADDITION to `created_at_gte/lte`, not a replacement. Both useful: agents prefer enums, scripts prefer ISO dates.

---

## Recommendations for granola-kit's surface revision

Concrete delta from the plan's current 19-tool roster:

### ADD (3 tools)

| Tool | Domain | Rationale |
|---|---|---|
| `get_account_info()` | account (new file `tools/account.py` or fold into `workspaces.py`) | Trivial; agent-essential; absorbed from official. |
| `search_meetings(query, document_ids?, time_range?)` | meetings | DEFER to a follow-up that depends on document-search integration. Stub now or hold for the deferred "workspace interplay" session. RECOMMEND: hold — don't ship a half-baked version. Add to `docs/possible-features.md`. |
| `time_range` param on existing `list_meetings` | meetings | Convenience enum; coexists with `created_at_gte/lte`. |

### KEEP everything else

The 19 tools in the plan stand. granola-kit's mutation surface is exactly the value-add over the official MCP.

### Consider renaming (LOW priority)

| Plan name | Alt | Rationale |
|---|---|---|
| `list_folders` | (keep) | Cleaner than `list_meeting_folders`. |
| `get_folder` | (keep) | Singular pair matches `get_meeting`. |
| `download_summary` | (keep) | Distinguishes from official's `get_meetings` inline-return. |
| `download_transcript` | (keep) | Same. |

### Final tool count

- Plan: 19 tools
- Plan + this survey's recommendations: **20 tools** (+1 for `get_account_info`; defer `search_meetings`)
- Plus one parameter addition to `list_meetings` (`time_range` enum)

---

## Caveats and incompleteness

What this survey couldn't confirm yet:

1. **Verbatim `tools/list` JSON schemas.** All schema details above are reconstructed from public docs + INFERRED. The next survey pass (post-auth) will replace the DOCS-based sections with OBSERVED JSON.
2. **Tool annotations (`readOnlyHint`, etc.).** All currently INFERRED.
3. **Citation format from `query_granola_meetings`.** Public docs say "inline citations" but no format example. Could be markdown footnotes, JSON sidecar, or text-anchored.
4. **Pagination shape.** No mention of cursors/offsets in `list_meetings` docs. INFERRED: no pagination — all-results within the time range. For paid plans w/ shared meetings, this could yield large responses. Needs verification.
5. **Folder filter parameter name on `list_meetings`.** Docs imply paid users can filter `list_meetings` by folder, but the param name isn't documented. INFERRED `folder_id`.
6. **`list_meeting_folders` output shape.** Docs say "folder ID, title, description, note count" but no JSON example.
7. **Rate limits.** Third-party blog says "~100 req/min baseline" — unverified.
8. **Error envelope shape.** MCP spec applies, but Granola-specific error codes/messages unknown.
9. **Free-tier behavior.** Chris is paid; can't observe free-tier gating directly.
10. **Plan-tier checks at runtime vs registration.** Does the server hide tools (`tools/list` filters) or expose all and 403 on call? Needs OBSERVATION.
11. **`document_ids` scoping on `query_granola_meetings` — is it limit/include or strict filter?** Docs unclear.
12. **Workspace switching.** `get_account_info` returns "active workspace". Is there a way to switch the active workspace mid-session? Not mentioned in docs.

---

## Appendix A — Sources

- Granola official docs: https://docs.granola.ai/help-center/sharing/integrations/mcp
- Granola blog announcement: https://www.granola.ai/blog/granola-mcp
- Scalekit integrator profile: https://www.scalekit.com/agent-connector/granolamcp
- PulseMCP listing: https://www.pulsemcp.com/servers/granola
- ClawHub listing (bholagabbar): https://clawhub.ai/bholagabbar/granola-mcp
- joelhooks/granola-cli: https://github.com/joelhooks/granola-cli
- OAuth endpoints: `https://mcp.granola.ai/.well-known/oauth-protected-resource`, `https://mcp-auth.granola.ai/.well-known/oauth-authorization-server`

## Appendix B — Survey reproduction

To reproduce/extend this survey:

```bash
# Register MCP (idempotent)
claude mcp add granola-official --scope user --transport http https://mcp.granola.ai/mcp

# Authenticate (interactive — opens browser at mcp-auth.granola.ai/oauth2/authorize)
# From interactive Claude Code session, run: /mcp  then choose granola-official

# Probe unauthenticated state (returns 401 + WWW-Authenticate)
curl -i -X POST https://mcp.granola.ai/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'

# Verbatim tool list (post-auth — extract from a Claude Code session's MCP transport)
# Look at: ~/Library/Caches/claude-cli-nodejs/<project>/mcp-logs-granola-official/*.jsonl
```
