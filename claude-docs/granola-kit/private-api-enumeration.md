# Granola Private API — Endpoint Enumeration

**Granola.app version:** 7.220.0 (CFBundleShortVersionString)
**Bundle location:** `/Applications/Granola.app/Contents/Resources/app.asar` (~57 MB)
**Unpacked size:** ~90 MB
**Analysis date:** 5/14 2:30 PM PT

## Method

```bash
# 1. Unpack the asar archive
npx --yes @electron/asar extract \
  /Applications/Granola.app/Contents/Resources/app.asar \
  /tmp/granola-asar/

# 2. Pull all endpoint URLs from the two files that contain them
#    (dist-app/assets/schemas-62S41ciY.js is the authoritative endpoint→URL map,
#    dist-app/assets/api-Dl530PM_.js is the HTTP client with wrapper functions)
grep -rhEo 'https?://[a-z.]+\.granola\.ai/v[0-9]+/[a-zA-Z0-9_-]+' /tmp/granola-asar/ \
  | sort -u

# 3. Parse the schemas file to extract endpoint-name → full-URL mapping
#    (this is the source of truth — the schemas file holds the runtime config map)
grep -ohE '"[a-z0-9-]+":`https://[^`]+/v[12]/[a-z0-9-]+`' \
  /tmp/granola-asar/dist-app/assets/schemas-62S41ciY.js

# 4. Parse api-Dl530PM_.js for `function FOO(...) { return N(..., 'endpoint-name', {input:t})}`
#    pattern to identify which endpoints have first-class wrappers and their method/queue
```

**Source-of-truth files:**

- `dist-app/assets/schemas-62S41ciY.js` (160 KB) — runtime config schema with `api_urls` object mapping endpoint-name → full URL. **All 379 distinct endpoint names live here.**
- `dist-app/assets/api-Dl530PM_.js` (32 KB) — the HTTP client (`function N(token, endpoint_name, {input, queue, retries, signal, method, additionalHeaders})`). Contains 205 typed wrapper functions like `async function Ir(e){return N(e,"get-action-items",{})}`.
- `dist-electron/main/index.js` (9.4 MB) — Electron main process. Contains a duplicate copy of the schemas table and additional renderer-side integrations (Sentry, cache stores, OAuth callbacks). Same 379-endpoint surface.

**Auth header builder (`function k` in `api-Dl530PM_.js`):**

```js
async function k(n) {
  let r = {};
  if (n) r.Authorization = `Bearer ${n}`;
  r['X-Client-Version'] = `${PACKAGE_VERSION}${IS_WEB ? '.web' : ''}${DEV ? '.dev' : ''}`;
  r['X-Granola-Platform'] = ...;          // 'darwin', 'win32', or 'web'
  r['X-Granola-Workspace-Id'] = ...;      // active workspace UUID
  r['X-Granola-Device-Id'] = ...;         // device hash via getDeviceId()
  r['X-Granola-Os-Version'] = ...;        // via getOsVersion()
  r['X-Granola-Time-Zone'] = ...;         // IANA TZ
  r['X-Granola-Source'] = ...;            // 'desktop', 'web', etc.
  // ...
}
```

These gating headers match what was observed in Feb 2026; nothing additional has been added at the transport layer. `Content-Type: application/json` is added when there is a JSON body.

**Streaming endpoints** use a separate helper `x({input, init, ...})` that wraps `fetch` and reads `response.body.getReader()` to consume SSE-like newline-delimited JSON chunks. The chunk schema includes `text_delta`, `reasoning`, `context_sources_delta`, `citationIds`, `webSearch`, `outputs`, `stream_completed`. Used by all `stream.api.granola.ai/v1/*` endpoints.

## Subdomain topology

Granola uses **7 distinct hosts**:

| Host | Endpoints | Role (inferred) |
|---|---:|---|
| `api.granola.ai` | 294 | Core monolith — documents, lists, workspaces, sharing, OAuth, billing |
| `cinnamon.api.granola.ai` | 21 | Newer feature service — attachments, action item feedback, transcript sensitivity, follow-up emails, room device |
| `maple.api.granola.ai` | 22 | Action items + briefs + analytics + invites |
| `stream.api.granola.ai` | 15 | LLM streaming endpoints (chat, summaries, briefs, follow-up emails) |
| `chia.api.granola.ai` | 13 | Access requests, room device pairing, integrations |
| `berry.api.granola.ai` | 13 | Telemetry, dictation, ambient context, follow-up emails, list rules, derived metadata |
| `pecan.api.granola.ai` | 12 | Spaces (new container abstraction), Zoom auth, calendar auth, list rules deletion, integration config |

These are real, addressable hostnames hardcoded in the schemas table; the endpoint name → URL mapping is the source of truth, not a path on `api.granola.ai`.

## Complete endpoint inventory

**379 distinct endpoints** discovered across the 7 hosts. Breakdown:

| Category | Count | Notes |
|---|---:|---|
| **Has** (wrapped by `granola-mcp.py`) | 11 | The 14 tools collapse onto these 11 URLs |
| **Deferred** (in `POSSIBLE_FEATURES.md`) | 48 | 1 endpoint in `POSSIBLE_FEATURES.md` was removed: `/v1/sync-pull` (replaced by `sync-push` only) |
| **Novel** (in neither source) | 320 | New surface revealed by this enumeration |

Full TSV with `endpoint | host | category | has_wrapper_fn` saved at `/tmp/final-inventory.tsv`.

### Has — wrapped by granola-mcp.py (11)

| Endpoint | Host | Wrapper fn in `api-Dl530PM_.js` |
|---|---|---|
| `get-documents` | `api.granola.ai` (v1) | (legacy URL, schema also defines `get-documents-v2`) |
| `get-documents-v2` | `api.granola.ai` (v2) | `async P(e,t)` |
| `get-documents-batch` | `api.granola.ai` | (no first-class wrapper — uses `N` directly inside cache fetchers) |
| `get-document-lists-metadata` | `api.granola.ai` | (used by `get_meeting_lists`) |
| `get-document-panels` | `api.granola.ai` | `async F(e,t,{retries:2})` |
| `get-document-set` | `api.granola.ai` | (used by `get_meetings` batch) |
| `get-document-transcript` | `api.granola.ai` | `async K(e,t)` |
| `get-workspaces` | `api.granola.ai` | yes |
| `update-document` | `api.granola.ai` | `async L(e,t,{queue:"documents",retries:1})` |
| `delete-workspace` | `api.granola.ai` | yes |
| `create-workspace` (v2) | `api.granola.ai` | yes |

### Deferred — in POSSIBLE_FEATURES.md (48 still live; 1 removed)

All 48 endpoints listed in `POSSIBLE_FEATURES.md` (besides the removed `sync-pull`) still exist in 7.220.0. Of particular note for v1 priority:

- `get-user-info`, `get-user-preferences`, `update-user-preferences` — user state
- `get-feature-flags`, `set-feature-flag`, `reset-feature-flags` — feature flagging
- `get-current-subscription`, `get-subscriptions`, `get-free-trial-data` — billing surface
- `get-integrations`, `get-hubspot-integration`, `get-notion-integration`, `get-slack-integration`, `get-attio-integration`, `get-zapier-connections` — integration discovery
- `get-people`, `set-person` — people/contacts (uses `set-person` twice in code; appears live)
- `get-panel-templates`, `create-panel-template`, `update-panel-template` — template system
- `add-document-to-list`, `remove-document-from-list`, `create-document-list`, `delete-document-list`, `update-document-list`, `add-users-to-document-list` — list/folder CRUD (a v2 variant exists for most)
- `get-recipes`, `upsert-recipe`, `delete-recipe`, `track-recipe-usage` — recipes (workspace-scoped prompt templates)
- `get-workspace-members`, `add-workspace-members`, `set-workspace-roles` — team CRUD
- `upsert-document`, `create-document` — document creation
- `refresh-google-events`, `refresh-calendar-events`, `get-selected-calendars`, `set-selected-calendars` — calendar sync
- `upload-file`, `get-attachments` — file storage
- `get-entity-batch`, `get-entity-set` — entity graph
- `sync-push` — sync engine push (no pull anymore — see drift section)

## Novel endpoints (not in either source) — 320

Grouped by capability area:

### A. Action items (9 endpoints, all on `maple.api.granola.ai` except feedback on `cinnamon`/`maple`)

The action-items subsystem is **completely absent** from `granola-mcp.py` and `POSSIBLE_FEATURES.md`. This is the most prominent recent Granola feature.

| Endpoint | Wrapper | Purpose (inferred) |
|---|---|---|
| `get-action-items` | `async Ir(e){return N(e,"get-action-items",{})}` | List action items (no input — implicit user/workspace scope) |
| `update-action-item` | `async zr(e,t){return N(e,"update-action-item",{input:t})}` | Mark complete, edit text, reassign |
| `regenerate-action-items` | (no wrapper — likely IPC) | Trigger LLM regeneration for a doc's action items |
| `regenerate-action-items-preview-stream` | streaming POST, no input | Stream preview while re-deriving |
| `apply-regenerated-action-items` | `async Rr(e,t){return N(e,"apply-regenerated-action-items",{input:t})}` | Commit regenerated items |
| `create-action-item-feedback` | `async Br(e,t){return N(e,"create-action-item-feedback",{input:t})}` | RLHF — thumbs up/down |
| `update-action-item-feedback` | (no wrapper in api-Dl530PM_) | Update existing feedback |
| `get-action-item-feedback` | (no wrapper) | Read feedback history |
| `update-action-item-review` | (no wrapper) | Review workflow state |

**Recommendation: add to v1.** A `list_action_items()` + `update_action_item()` tool is high-value, read-mostly, well-shaped. Effort: small (single endpoint per tool, no streaming).

### B. Search and RAG (10 endpoints — span `api`, `stream`)

Granola has full RAG infrastructure that `granola-mcp.py` doesn't expose. The current `list_meetings` does client-side title substring filtering, which is the only option without these.

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `search-meetings-turbopuffer` | `async Or(e,t){return N(e,"search-meetings-turbopuffer",{input:t})}` | Server-side meeting search via Turbopuffer vector store |
| `turbopuffer-index-documents` | `async Er(e,t){return N(...)}` | Index docs into Turbopuffer (write — admin path?) |
| `turbopuffer-index-documents-generator` | `async Dr(e,t)` | Index generator stream |
| `search-embeddings` | (no first-class wrapper) | Embedding search (older — pre-turbopuffer?) |
| `embeddings-ada` | (no wrapper) | Direct OpenAI-ada-style embedding |
| `generate-document-embeddings` | (no wrapper) | Compute embeddings for a doc |
| `chat-with-documents` | streaming POST (desktop) | Conversational RAG — full chat with citations, web search, reasoning |
| `chat-with-documents-web` | streaming POST (web client) | Same, web variant |
| `get-wiki-page` | (no wrapper in api-Dl530PM_) | Read a workspace wiki page |
| `generate-wiki-page` | (no wrapper) | LLM-generate wiki content |

**Recommendation: `search-meetings-turbopuffer` is the single most valuable add.** Replaces brittle client-side title substring filtering with real semantic search. Effort: small if input is `{query, limit}`-shaped (need to verify); medium if it requires workspace_id and offset cursors. Read-only.

`chat-with-documents` is tempting but streaming + complex chunk schema; defer to v2.

### C. Pre-meeting briefs (4 endpoints — `maple`, `stream`)

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `get-pre-meeting-briefs` | (no wrapper in api-Dl530PM_; called from primary) | List briefs for upcoming meetings |
| `pre-meeting-brief` | streaming POST | Generate a brief on-demand for a single meeting |
| `requeue-pre-meeting-briefs` | `async Ne(e,t){return N(e,"requeue-pre-meeting-briefs",{input:t})}` | Re-enqueue brief generation |
| `share-brief-via-gmail` | (no wrapper) | Share brief link via email |

**Recommendation: `get-pre-meeting-briefs` as read-only tool.** Useful before calls. Effort: small.

### D. List rules (4 endpoints, on `berry`/`maple`/`pecan`)

Granola has automatic folder routing based on rules — this is novel and unexposed.

| Endpoint | Wrapper |
|---|---|
| `get-list-rules` | no wrapper |
| `create-list-rule` | `async kr(e,t){return N(e,"create-list-rule",{input:t})}` |
| `update-list-rule` | `async Ar(e,t){return N(e,"update-list-rule",{input:t})}` |
| `delete-list-rule` | (jr — not extracted but exists) |
| `match-document-against-rules` | `async Mr(e,t){return N(e,"match-document-against-rules",{input:t})}` |

**Recommendation: skip for v1.** Niche workflow. Could be added as a v2 tool if Chris uses folder rules.

### E. Follow-up emails (8 endpoints, span `berry`/`cinnamon`/`maple`/`stream`)

Generate & send follow-up emails after meetings.

| Endpoint | Type |
|---|---|
| `get-follow-up-emails` | read |
| `get-follow-up-email-thread` | read |
| `generate-follow-up-email` | write (non-streaming) |
| `generate-follow-up-email-stream` | write (streaming) |
| `rewrite-follow-up-email-stream` | write (streaming) |
| `send-follow-up-email` | write |
| `update-follow-up-email-outcome` | write |
| `request-follow-up-email-attachment-upload-url` | write |

**Recommendation: `get-follow-up-emails` + `get-follow-up-email-thread` read-only.** Useful for "show me drafts" workflows. Skip `send-` for v1 (write, potentially destructive). Effort: small.

### F. About-me + ambient context (4 endpoints, on `berry`/`cinnamon`)

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `get-about-me-profile` | no wrapper | Read user's "about me" auto-generated profile |
| `generate-about-me-on-demand` | no wrapper (cinnamon) | Trigger regeneration |
| `get-ambient-context` | `async Nr(e,t){return N(e,"get-ambient-context",{input:t})}` | Active-window/screen ambient signals |
| `process-ambient-context` | no wrapper (cinnamon) | Ingest ambient context server-side |

**Recommendation: `get-about-me-profile` and `get-ambient-context` as read-only tools.** Niche but unique — Granola exposes them only to its own UI. Effort: small. Privacy note: ambient context likely contains app names / window titles; safe within Chris's own data.

### G. Folder analytics + summaries (3 endpoints, all `api.granola.ai`)

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `get-folder-digest` | no wrapper | Folder-level digest (likely AI summary) |
| `get-sidebar-card` | `async vr(e){return N(e,"get-sidebar-card",{})}` | Generic "what to show in sidebar" |
| `trigger-folder-summary` | `async Rt(e,t,n){return N(e,"trigger-folder-summary",{input:{document_list_id:t,regenerate:n?.regenerate}})}` | Kick off folder summarization |

`get-workspace-analytics` (on `maple`) — `async ut(e,t){return N(...)}`. Workspace-level dashboard metrics. **Recommendation: ship as a tool.**

### H. Data export (2 endpoints)

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `create-data-export` | `async br(e,t){return N(e,"create-data-export",{input:t})}` | Request full account export |
| `download-data-export` | no wrapper (maple) | Download the resulting file |

**Recommendation: ship both as a `create_data_export` + `download_data_export` pair.** High user value for "Granola, give me everything." Read-only on data. Effort: small.

### I. Document derived metadata + status (3 endpoints)

| Endpoint | Wrapper | Notes |
|---|---|---|
| `get-document-metadata` | inline: `async W(e,t){return await N(e,"get-document-metadata",{input:t,queue:"documents"})}` | Lightweight metadata fetch (faster than `get-documents-batch` for single docs) |
| `get-document-derived-metadata` | no wrapper (berry) | LLM-derived metadata — sentiment, topics, key entities |
| `get-document-status` | no wrapper | Processing state (transcription done? notes generated?) |

**Recommendation: `get-document-derived-metadata` is interesting** if it surfaces topic/entity extraction we don't have. Verify with a live call before committing.

### J. Document access / sharing (15 endpoints, mostly `chia`/`api`)

Comprehensive access-request workflow not in `granola-mcp.py`. Includes both document and folder-level access requests.

**Recommendation: `get-shared-documents`, `check-document-access`, `get-document-inherited-access-sources` as read-only tools.** Helpful for understanding what's shared. Skip `grant-` / `request-` (write, complex permission semantics) for v1.

### K. MCP infrastructure (7 endpoints, all `api.granola.ai`)

Granola hosts its own MCP at `mcp.granola.ai/mcp` (constant found at `getAPIEndpointInWindow-DO4ZFmCm.js`). The endpoints below are the **management layer** for the official MCP, not the MCP itself.

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `mcp-registry` | `async Sr(){return N(null,"mcp-registry")}` | List available MCP servers/tools |
| `mcp-info` | `async Cr(e){return N(e,"mcp-info")}` | Metadata for the MCP service |
| `mcp-tool-execute` | `async wr(e,t){return N(e,"mcp-tool-execute",{input:t})}` | Server-side execute a tool (proxying?) |
| `manage-mcp-token` | no wrapper | Create/revoke MCP access tokens |
| `mcp-oauth-start` | `async xr(e,t){return N(e,"mcp-oauth-start",{input:t})}` | Begin OAuth for external MCP server |
| `mcp-oauth-callback` | no wrapper | OAuth callback handler |
| `mcp-server` (on `stream`) | no wrapper | The actual MCP server endpoint? |

**Recommendation: out of scope for granola-kit.** These are for Granola's own MCP product — they don't expose user data, they configure the official MCP that `upstream-mcp` teammate is investigating.

### L. People graph (3 endpoints)

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `get-people` | no wrapper in api-Dl530PM_ (pecan) | List people known to the workspace |
| `set-person` | `async oe(e,t){await N(e,"set-person",{input:t})}` | Upsert person record |
| `get-entity-batch` / `get-entity-set` | wrappers exist | Entity graph (likely people + companies + topics) |

**Recommendation: `get-people` as read-only tool.** Effort: small. Privacy-sensitive: people in workspace include attendees from past meetings.

### M. Public API keys (3 endpoints)

| Endpoint | Wrapper | Purpose |
|---|---|---|
| `create-public-api-key` | `async Dt(e,t){...}` | Mint a long-lived API key |
| `get-public-api-keys` | `async Et(e,t){...}` | List existing keys |
| `revoke-public-api-key` | `async Ot(e,t){...}` | Revoke |

**Recommendation: ship `get-public-api-keys` (read-only) for visibility.** Skip create/revoke for v1 (writes affect cross-session auth state).

### N. PII / consent / privacy (4 endpoints)

`redact-pii` (on cinnamon, no wrapper), `check-transcript-sensitivity`, `get-meeting-consent-status`, `create-affirmative-consent`. Useful context but not v1 candidates.

### O. Other significant novel endpoints

- `find-free-calendar-slots` — calendar slot finder (no wrapper; likely Calendar Assistant). Worth exposing as tool.
- `create-calendar-event`, `get-google-events` — calendar CRUD (Granola operates as a calendar client too)
- `list-user-sessions`, `revoke-session` — session management
- `get-cloud-agent-connectors`, `cloud-agent`, `cloud-agent-connector` — "cloud agent" subsystem (likely the always-on transcription/joining-bot infrastructure)
- `room-device`, `room-device-create-pairing-token`, `room-device-claim-pairing-token` — Granola Rooms (physical meeting rooms)
- `connect-meet-media`, `meet-media-auth`, `meet-media-auth-complete` — Google Meet direct integration
- `save-to-integration`, `save-to-notion`, `create-attio-note`, `search-attio-records`, `list-slack-channels`, `post-slack-message` — push-to-integration verbs (workspace-side mutations)
- `dictation-format`, `get-dictation-upload-url` — voice-note features
- `get-groq-token`, `get-deepgram-token`, `get-transcription-auth-token` — third-party tokens (sensitive, skip)
- `phone-*` (10 endpoints) — phone integration (Granola Voice)
- `update-config` — generic key/value config update (broad surface)

### P. Out-of-scope novel categories

- **Webhooks** (`*-webhook`, `*-webhook-handler` — 11 endpoints): inbound from Stripe, WorkOS, Twilio, Knock, Loops, Slack, transcription. Not callable from clients.
- **OAuth callbacks** (`*-oauth-callback` — 8 endpoints): not callable from clients.
- **WebSocket signaling** (`websocket-*`, `public-websocket-*` — 6 endpoints): Lambda WebSocket handlers, not useful from MCP.
- **Test endpoints** (`test-400`, `test-500`, `test-exception`, `test-call-sns`): for backend health checks.
- **Login/auth flow internals** (`login-check`, `login-complete`, `auth-handoff-complete`, `workos-auth-complete`): client uses these for the OAuth dance; granola-mcp leverages the resulting token.

## Drift vs. POSSIBLE_FEATURES.md

**What's new since Chris's Proxyman work (Oct 2025):**

- The entire **action items subsystem** (`get-action-items`, `update-action-item`, `regenerate-action-items*`, `create-action-item-feedback`, etc.) is post-Proxyman.
- The entire **follow-up email subsystem** (8 endpoints) appears to be post-Proxyman.
- **Turbopuffer search** (`search-meetings-turbopuffer`, `turbopuffer-index-documents*`) is post-Proxyman. Server-side semantic search is now a real option.
- The **flavored subdomains** (`berry`, `chia`, `cinnamon`, `maple`, `pecan`, `stream`) didn't show up in `POSSIBLE_FEATURES.md` — Chris's earlier capture was 100% `api.granola.ai`. Granola has rolled newer features onto dedicated services.
- **Streaming endpoints on `stream.api.granola.ai`** (15 endpoints): all new, including `chat-with-documents` with full RAG (citations, web search, multi-step reasoning).
- **Spaces** (`create-space`, `get-space-for-invite-code`, `create-space-invite-link`, etc. — 11 endpoints on `pecan`): new container abstraction, separate from workspaces.
- **Room device** (3 endpoints) — Granola Rooms feature.
- **Pre-meeting briefs** (4 endpoints) — automated briefing system.
- **List rules** (4 endpoints) — folder auto-routing rules.
- **MCP management** (7 endpoints) — Granola's own MCP product surface.
- **Public API keys** (3 endpoints) — the user can now mint long-lived API keys via Granola UI (so `granola-kit` could authenticate via mint-and-store flow rather than scraping `supabase.json` only).
- **Cloud agent** (3 endpoints) — likely backs the always-on meeting joiner that Granola announced as a feature.

**What's gone:**

- `/v1/sync-pull` is no longer in the schema. Only `sync-push` remains. Sync is now write-only from the client; the desktop app receives state via `get-documents-delta`, `get-document-set`, websocket pushes, or polling.

**What's changed shape:**

- `/v1/get-documents` — the v1 path still exists but the schema also defines a `get-documents-v2` key (URL still `https://api.granola.ai/v2/get-documents`). The wrapper function `P` in `api-Dl530PM_.js` calls `get-documents-v2`. `granola-mcp.py` already uses the v2 URL, so this is consistent.
- `/v1/create-document-list-v2`, `/v1/delete-document-list-v2`, `/v1/update-document-list-user-v2`, `/v1/add-users-to-document-list-v2`, `/v1/remove-users-from-document-list-v2`, `/v1/batch-update-document-lists-v2` — newer list-CRUD endpoints with v2 suffixes coexist with v1 originals. The original v1 versions still exist (probably for older clients).
- `/v1/get-document-lists` exists (POSSIBLE_FEATURES had `get-document-lists-metadata` only) and a `get-document-lists-v2` is referenced too.
- The official MCP entry point is `https://mcp.granola.ai/mcp` (confirmed via string constant in `getAPIEndpointInWindow-DO4ZFmCm.js`).

## Recommendations for granola-kit's surface

Reading the plan at `/Users/chris/.claude/plans/reactive-foraging-wreath-clone-019e26cb.md`, granola-kit is scoped to 19 tools. Based on this enumeration, here are concrete deltas to consider.

### High-value v1 additions (read-only, small effort)

| Proposed tool | Endpoint(s) | Why |
|---|---|---|
| `search_meetings` | `search-meetings-turbopuffer` | Replaces client-side title-substring filter with proper server-side semantic search. Eliminates the biggest known limitation in `granola-mcp.py`. |
| `list_action_items` | `get-action-items` | First-class access to Granola's action-item subsystem (entirely missing today). |
| `update_action_item` | `update-action-item` | Lets the user mark items complete from chat. Pairs naturally with `list_action_items`. |
| `get_pre_meeting_briefs` | `get-pre-meeting-briefs` | Surface briefs before calls. |
| `get_about_me_profile` | `get-about-me-profile` | Read user's Granola-generated self-profile. |
| `list_people` | `get-people` | Workspace people directory. |
| `list_follow_up_emails` | `get-follow-up-emails` | Surface follow-up drafts. |
| `get_workspace_analytics` | `get-workspace-analytics` | Metrics for the workspace. |
| `create_data_export` + `download_data_export` | `create-data-export`, `download-data-export` | "Give me everything." |
| `get_user_info` + `get_user_preferences` | (deferred list) | Trivially small, useful. |
| `get_integrations` | `get-integrations`, `list-available-integrations` | Show what's connected. |

That's ~12 new tools, mostly small. If you want to stay near 19 total, drop existing duplicates that the action-items / search additions subsume.

### v1 additions worth considering (with caveats)

- `list_meetings_v2(query=...)` — if you can drop the current client-side filtering entirely, do. The Turbopuffer endpoint should accept `{query, limit, workspace_id?}` (need to confirm payload shape via runtime trace).
- `get_document_metadata` and `get_document_derived_metadata` — lighter than full document fetch, possibly includes Granola's LLM-derived signals (sentiment, topics).
- `find_free_calendar_slots` — useful if Granola's calendar integration is a primary surface for the user.

### Out of scope for v1

- All streaming endpoints (`chat-with-documents`, generate/rewrite stream variants) — wait for granola-kit to have streaming machinery.
- All write endpoints to third-party integrations (`save-to-notion`, `post-slack-message`, `create-attio-note`, `send-follow-up-email`) — too easy to misuse.
- MCP management endpoints (`mcp-*`) — those belong to the official MCP, not the user data layer.
- Phone, room-device, cloud-agent — niche features, defer.
- Anything with `*-oauth-callback`, `*-webhook`, `test-*`, `websocket-*` — not callable from clients anyway.

### Schema changes worth considering

The current `granola-mcp.py` uses strict `extra='forbid'` Pydantic models. Given Granola's drift, a few notes:

1. **The `delete-workspace` wrapper exists today, but in v7.220.0 there is no `delete-space` corresponding wrapper for the old workspace concept.** Spaces are a new abstraction (11 endpoints on `pecan`) that coexist with workspaces. granola-kit should pin to workspaces for now; spaces appear to still be opt-in.
2. **Many endpoints return paginated structures with `cursor` rather than `offset`.** The `get-documents-delta` and Turbopuffer endpoints likely use cursors. Test before assuming offset-based.
3. **The header `X-Granola-Source` is new.** The Feb 2026 gating headers are now: `X-Client-Version`, `X-Granola-Platform`, `X-Granola-Workspace-Id`, `X-Granola-Device-Id`, `X-Granola-Os-Version`, `X-Granola-Time-Zone`, `X-Granola-Source`. granola-kit should send these; the existing `granola-mcp.py` only sends `Authorization: Bearer`. Anecdotally requests still work without them, but adding them future-proofs against gating becoming required.

## Local DB encryption (March 2026)

Granola 7.220.0 bundles `better-sqlite3-multiple-ciphers@^12.9.0` (per `package.json`) and `cipher`/`encrypt` strings appear ~321 times across `dist-electron/main/index.js`. The local SQLite cache at `~/Library/Application Support/Granola/cache-v3.db` is encrypted at rest. **Implication for granola-kit:** the project's spec wisely already targets the **HTTP API** (via `supabase.json` token) rather than the local DB, which is the correct call. Reading the encrypted SQLite would require recovering the key, which Granola intentionally hides. Stay on the API.

## Caveats and incompleteness

- **All JS is minified.** Wrapper function bodies are short and well-typed (the input shape is always `t` passed as `input:t`), but the exact field names of the request payloads can only be verified by:
  - Setting up a Proxyman / mitmproxy intercept and clicking through the feature
  - Reading the cache-store code in `cacheStore-CJU6AFeP.js` which often serializes the input
  - Calling the endpoint live with a test payload (read-only ones only)
- **The 175 "no wrapper" endpoints** are referenced elsewhere (renderer components calling fetch directly, or IPC code in the Electron main process). They are reachable; they just don't have first-class typed wrappers in `api-Dl530PM_.js`. Examples: `get-pre-meeting-briefs`, `redact-pii`, `find-free-calendar-slots`. Confirming their shapes requires runtime tracing.
- **I did not call any endpoint.** Everything here is static analysis of the legally-licensed app bundle on Chris's machine. Verifying response schemas for novel endpoints requires either runtime trace (Proxyman) or careful client-side instrumented calls during follow-up work.
- **Endpoint name → URL mapping is the schema table.** I did not verify that all 379 endpoints are actually deployed live; some may be 404s with stub registration. Calling each one (read-only ones) would confirm.
- **Subdomain assignment is hard-coded.** Granola may move endpoints between subdomains in future releases; granola-kit should use the schema-table-style "endpoint name to host" mapping if it ever needs to call many of these.
- **I did not look at `dist-electron/preload/preload.js`.** Could contain IPC channels relevant to endpoints that the renderer doesn't call directly. Probably not load-bearing for HTTP API enumeration.

## Reproducibility

All artifacts saved to `/tmp/`:

- `/tmp/granola-asar/` — unpacked Electron bundle
- `/tmp/granola-endpoints-full-urls.txt` — 289 `api.granola.ai` URLs
- `/tmp/granola-flavored-endpoints.txt` — 96 flavored-subdomain URLs
- `/tmp/endpoint-url-tsv.txt` — 379 endpoint → host → URL
- `/tmp/endpoint-wrappers-detail.tsv` — 205 wrapper function signatures
- `/tmp/final-inventory.tsv` — full classification (`has` / `deferred` / `novel`)
- `/tmp/has-table.md`, `/tmp/deferred-table.md`, `/tmp/novel-table.md` — per-category markdown tables

All grep-derived; re-runnable against the unpacked bundle.
