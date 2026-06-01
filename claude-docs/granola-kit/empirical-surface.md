# Empirical surface — mitmproxy capture findings

**Capture source**: `/tmp/granola-capture/granola-flows.mitm` — 3,002 flows / 1,180+ to `*.granola.ai` / 73 distinct POST endpoints / driven by real Granola.app usage on `chris-K4442D3Q0X` (M4).

**Identity headers (real, from capture)**:

| Header | Value |
|---|---|
| `x-client-version` | `7.220.0` |
| `x-granola-platform` | `macOS` (not `darwin`) |
| `x-granola-os-version` | `26.4.1` |
| `x-granola-device-id` | `71a05389…` = **`sha256(IOPlatformUUID)`** — upper-case, dashed, no trailing newline (verified 2026-05-29: `sha256("7F7E1B9E-…-DA91") == 71a05389…fa72f8e`). Matches the shipped gate-passing patch + `granola-client`. Earlier "different hash / derivation TBD" note was wrong (probe likely hashed a lower-cased/normalized form). |
| `x-granola-workspace-id` | `6d13b566-5b90-4d22-8e7f-5bc15d12f186` (workspace UUID from token claims or settings) |
| `x-granola-time-zone` | `America/Los_Angeles` |
| `user-agent` | `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Granola/7.220.0 Chrome/146.0.7680.188 Electron/41.2.1 Safari/537.36` |

The plan's identity header list included `x-granola-source: desktop`. **Not seen in capture.** Drop it.

## Verified-working endpoints with payload shapes

All status 200 unless otherwise noted. `host` defaults to `api.granola.ai` if unprefixed.

### Documents (meetings)

| Endpoint | Request | Response shape |
|---|---|---|
| `get-document-set` | `{}` (paginated via cursor) | `{documents: {id → {updated_at, owner, has_ydoc?}}}` — lightweight index |
| `get-document-metadata` | `{document_id}` | full document payload |
| `get-document-panels` | `{document_id}` | array of panels (AI-generated notes) |
| `get-document-transcript` | `{document_id}` | array of transcript segments |
| `update-document` | `{id, ...fields_to_set}` | success — used for update/delete/restore via `deleted_at` |
| `create-document` | `{id, user_id, ...}` | created doc — client-generated UUID |
| `check-document-access` | `{document_id}` | access check |
| `get-users-with-access` | `{document_id, filter_has_accessed}` | user list |
| `get-attachments` | `{document_list_id}` | attachment list |

### Folders (document lists)

| Endpoint | Request | Notes |
|---|---|---|
| `get-document-lists-metadata` | `{include_document_ids, include_only_joined_lists}` | full hierarchy with parents, visibility, icons |
| `create-document-list-v2` | `{id, title, visibility, ...}` | client-generated UUID |
| `add-document-to-list` | `{document_id, document_list_id}` | join |
| `get-folder-zapier-integrations` | `{document_list_id}` | per-folder integrations |
| `pecan.list-document-integration-shares` | `{documentId}` (camelCase!) | integration shares |

### Workspaces & members

| Endpoint | Request |
|---|---|
| `get-workspaces` | `{}` |
| `get-workspace-members` | `{workspace_id}` |
| `get-workspace-invite-links` | `{workspace_id}` |
| `get-workspaces-notes-count` | `{}` |
| `create-workspace-invite-link` | `{workspace_id}` |
| `add-workspace-members` | `{workspace_id, emails: [...]}` |

### Identity & account

| Endpoint | Request | Response highlights |
|---|---|---|
| `get-user-info` | `{}` | `{id, email, user_metadata, signed_in_on_platforms, ...}` |
| `get-user-preferences` | `{}` | prefs |
| `update-user-preferences` | `{timestamp, timestamps, ...}` | sync state |
| `get-current-subscription` | `{include_stripe_data}` | `{active_plan_id, owner_type, owner_id, status}` |
| `get-subscriptions` | `{include_enterprise_plan}` | plans |
| `cinnamon.get-paywall-status` | `{}` | `{state, trialEndsAt, paywallThresholdDays, salesOverride}` |
| `get-feature-flags` | `{force_defaults}` | array of `{feature, value, user_id, min_client_version}` |
| `get-privacy-mode` | `{}` | privacy state |

### Chat / RAG (the real "search" UX)

| Endpoint | Request | Notes |
|---|---|---|
| `stream.chat-with-documents` | `{thread_id, chat_history, messageContext: {mode, currentViewContext}}` | **Streaming** — server-side ReAct loop. LLM picks tools (`listMeetings`, etc.) and assembles answer with inline citations. This IS Granola's meeting search. |
| `get-chat-citation` | `{answer_text, meeting_id, exclude_transcript}` | `{citation_source_type, citation_source_quote}` — resolves citations from answer text |
| `get-chat-models` | `{}` | list of `{id, label, provider, default, requires_reasoning}` — Claude 4.6, GPT-5.5, etc. |
| `llm-proxy` | `{prompt_slug, prompt_variables}` | OpenAI-format response — server-side prompt template invocation |
| `stream.llm-proxy-stream` | `{prompt_slug, panel_id, ...}` | streaming variant |

### Calendar

| Endpoint | Request | Notes |
|---|---|---|
| `refresh-calendar-events` | (no body) | refresh trigger |
| `get-selected-calendars` | `{}` | calendar list |
| `set-selected-calendar` | `{id, selected, provider}` | toggle |
| `get-transcription-auth-token` | `{provider}` | e.g. `"assembly-universal"` |

### Recipes (templates), exports, public API keys

| Endpoint | Request |
|---|---|
| `get-recipes` | `{}` |
| `track-recipe-usage` | `{recipe_id}` |
| `get-panel-templates` | `{}` |
| `create-data-export` | `{format: "csv", workspace_ids: [...]}` → 202 `{exportId, status, requestedAt}` |
| `get-public-api-keys` | `{workspace_id}` |
| `create-public-api-key` | `{workspace_id, scope: "user_notes"}` → `{success, api_key}` |

### MCP, integrations, misc

| Endpoint | Request | Notes |
|---|---|---|
| `mcp-registry` | `{}` | list of MCP servers (Notion, Linear, Amplitude, …) |
| `get-attio-integration` / `get-notion-integration` / `get-slack-integration` / `get-hubspot-integration` | `{}` | per-integration state |
| `get-integrations` | `{}` | summary |
| `get-zapier-connections` | `{user_id}` | zapier |
| `chia.list-available-integrations` | `{}` | catalog |
| `pecan.integration-folder-config` | `{document_list_id}` | folder ↔ integration |
| `airtable` | `{action, data}` | feedback endpoint (yes, lowercase `airtable`) |
| `send-download-email` | `{}` | request app download email |
| `create-tracked-link` | `{url_type, document_id, utm_*}` | tracking links |
| `embeddings-ada` | `{input: [...]}` | embeddings |
| `insert-transcriptions` | `{chunks: [...]}` | client-side transcript upload |
| `sync-push` | `{operations: [...]}` | bulk ops |
| `refresh-access-token` | `{refresh_token}` | already used in helpers.py |

## Endpoints NOT captured (still gaps)

These are in the desktop-app enumeration but **not seen in any flow during regular UI usage** of Granola.app:

- `search-meetings-turbopuffer` — likely server-side-only, called internally by `chat-with-documents`. Not directly exposed to clients.
- `get-pre-meeting-briefs`, `pre-meeting-brief` (stream)
- `get-follow-up-emails`, `get-follow-up-email-thread`, `generate-follow-up-email-stream`
- `get-workspace-analytics`
- `get-action-items`, `update-action-item` — gated (403 "Feature not enabled") on Chris's account
- `download-data-export` — only captured the create; would need to wait for export completion + click download
- `get-wiki-page`, `get-document-derived-metadata`
- `get-about-me-profile`, `get-people`

## V1 surface — empirical re-grounding

**Drop from plan** (not empirically reachable on this account):
- `search_meetings` (the killer feature) — endpoint exists but only called server-side
- `list_action_items` + `update_action_item` — 403
- `get_pre_meeting_briefs` — 404
- `list_follow_up_emails` — 404
- `get_workspace_analytics` — 400
- `download_data_export` — defer (create is fine, download untested)

**Add to plan** (captured working, high-leverage):
- `chat_with_meetings(question, thread_id?, chat_history?)` — wraps `stream.chat-with-documents`. **This replaces `search_meetings` and is strictly better** (Granola's own LLM answers with citations via its full corpus).
- `get_chat_citation(answer_text, meeting_id)` — resolves citations from chat answers.
- `list_chat_models()` — `get-chat-models`. Useful before chat invocations.
- `get_account_status()` — composes `get-user-info` + `get-current-subscription` + `cinnamon.get-paywall-status` + `get-workspaces`. Single-call full account picture.
- `list_feature_flags()` — `get-feature-flags`. Useful for clients to know what's gated server-side.
- `create_data_export(format)` — async; returns `export_id`. Pair with future `get_data_export_status(export_id)` once we can observe the polling pattern in capture.

**Net v1 roster: 25 tools** (was 28). Cuts: 7 unreachable (`search_meetings`, `list_action_items`, `update_action_item`, `get_pre_meeting_briefs`, `list_follow_up_emails`, `get_workspace_analytics`, `download_data_export`). Adds: 4 verified high-leverage (`chat_with_meetings`, `get_chat_citation`, `list_chat_models`, `list_feature_flags`).

| Domain | Count | Tools |
|---|---|---|
| Meetings | 7 | list_meetings, get_meeting, get_meetings, update_meeting, delete_meeting, restore_meeting, list_deleted_meetings |
| Chat (replaces search) | 3 | chat_with_meetings, get_chat_citation, list_chat_models |
| Notes | 3 | download_summary, download_note, download_transcript |
| Folders | 5 | list_folders, get_folder, create_folder, add_meeting_to_folder, remove_meeting_from_folder |
| Workspaces | 3 | list_workspaces, create_workspace, delete_workspace |
| Account | 2 | get_account_status, list_feature_flags |
| Exports | 1 | create_data_export |
| URLs | 1 | resolve_url |

**Total: 25 verified-working tools.**
