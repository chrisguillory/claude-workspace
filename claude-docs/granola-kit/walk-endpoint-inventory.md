# Walk endpoint inventory — Granola 7.277.1 full CDP breadth walk

Captured 143 non-OPTIONS flows across 47 unique endpoints (3 novel vs the 7.220 capture). Date 2026-05-29.

## api.granola.ai
- `check-for-update/latest-mac.yml` [GET] (302:11) ×11 **NEW** — ``
- `create-document-panel` [POST] (200:3) ×3 — `{"id":"331d9ba3-d2f0-46ef-a7cc-15c2cac80b99","created_at":"2`
- `create-referral-link` [POST] (200:1) ×1 — `{"config_slug":"referral_program_v1"}`
- `get-attio-integration` [POST] (200:2) ×2 — ``
- `get-chat-models` [POST] (200:2) ×2 — ``
- `get-cloud-agent-connectors` [POST] (200:2) ×2 — ``
- `get-current-subscription` [POST] (200:3) ×3 — `{"include_stripe_data":false}`
- `get-document-lists-metadata` [POST] (200:2) ×2 — `{"include_document_ids":true,"include_only_joined_lists":fal`
- `get-document-set` [POST] (200:2) ×2 — `{}`
- `get-document-transcript` [POST] (200:2) ×2 — `{"document_id":"6ee32f01-9e5a-43b0-9017-46114421d98f"}`
- `get-documents-batch` [POST] (200:8) ×8 **NEW** — `{"document_ids":["7AC92069-29B2-489E-A4D7-BABEF0DDDDB0"]}`
- `get-entity-set` [POST] (200:12) ×12 — `{"entity_type":"chat_thread"}`
- `get-feature-flags` [POST] (200:9) ×9 — `{"force_defaults":false}`
- `get-hubspot-integration` [POST] (200:2) ×2 — ``
- `get-integrations` [POST] (200:1) ×1 — ``
- `get-invite-list` [POST] (200:3) ×3 — `{"include_external_people":false,"include_granola_users":tru`
- `get-knock-user-token` [POST] (200:1) ×1 — ``
- `get-notion-integration` [POST] (200:2) ×2 — ``
- `get-offer` [POST] (200:4) ×4 — ``
- `get-panel-templates` [POST] (200:4) ×4 — ``
- `get-privacy-mode` [POST] (200:2) ×2 — ``
- `get-recipes` [POST] (200:3) ×3 — ``
- `get-selected-calendars` [POST] (200:1) ×1 — ``
- `get-sidebar-card` [POST] (200:1) ×1 — ``
- `get-slack-integration` [POST] (200:2) ×2 — ``
- `get-subscriptions` [POST] (200:1) ×1 — `{"include_enterprise_plan":true}`
- `get-user-info` [POST] (200:3) ×3 — ``
- `get-user-preferences` [POST] (200:1) ×1 — ``
- `get-users-with-access` [POST] (200:2) ×2 — `{"document_id":"6ee32f01-9e5a-43b0-9017-46114421d98f","filte`
- `get-workspace-invite-links` [POST] (200:6) ×6 — `{"workspace_id":"6d13b566-5b90-4d22-8e7f-5bc15d12f186"}`
- `get-workspace-members` [POST] (200:6) ×6 — `{"workspace_id":"6d13b566-5b90-4d22-8e7f-5bc15d12f186"}`
- `get-workspaces` [POST] (200:6) ×6 — `{}`
- `get-workspaces-notes-count` [POST] (200:2) ×2 — `{}`
- `get-zapier-connections` [POST] (200:1) ×1 — `{"user_id":"2b38a55c-f77b-46c0-93c3-8e0524b965a5"}`
- `list-slack-channels` [POST] (200:1) ×1 — ``
- `llm-proxy` [POST] (200:1) ×1 — `{"prompt_slug":"chat-thread-title","prompt_variables":{"mess`
- `mcp-registry` [POST] (200:1) ×1 — ``
- `refresh-access-token` [POST] (200:1) ×1 — `{"refresh_token":"Vq8eo0ZtLT9D3LlS9IVZZwqIg"}`
- `refresh-calendar-events` [POST] (200:8) ×8 — ``
- `sync-push` [POST] (200:3) ×3 — `{"operations":[{"type":"add","workspace_id":"6d13b566-5b90-4`
- `update-document-panel` [POST] (200:4) ×4 — `{"last_viewed_at":"2026-05-29T13:19:25.836Z","id":"0e2c8446-`
- `upsert-document` [POST] (200:5) ×5 **NEW** — `{"id":"981f311e-2ce9-42bb-9222-779d9dcdb118","created_at":"2`
- `upsert-integrations` [POST] (200:1) ×1 — `{"affinity_domain":null}`

## chia.api.granola.ai
- `list-available-integrations` [POST] (200:1) ×1 — ``

## cinnamon.api.granola.ai
- `get-paywall-status` [POST] (200:2) ×2 — ``

## pecan.api.granola.ai
- `list-document-integration-shares` [POST] (200:1) ×1 — `{"documentId":"6ee32f01-9e5a-43b0-9017-46114421d98f"}`

## stream.api.granola.ai
- `chat-with-documents` [POST] (200:1) ×1 — `{"thread_id":"7491f793-1e72-46d8-a3e8-df9136808252","chat_hi`

## Coverage conclusions & insights (full breadth walk)

Walked every navigable surface via CDP: Home, My notes, note-detail, all folders, Chat, Search, Shared-with-me, Recipes, People, Companies, Trash, workspace menu, and all 9 Settings tabs (Preferences/Profile/Calendar/Notifications/Connectors/General/Team/Billing/Referrals). Net: **~47 endpoints = the production desktop read surface**, and it's stable — only 2 genuinely-new *functional* endpoints vs the 7.220 capture.

**Key new endpoints:**
- **`get-documents-batch`** `{"document_ids":[...]}` → 200. The real **batch meeting fetch**. `get_meetings(meeting_ids)` should map here, not to singular `get-document-metadata`.
- **`upsert-document`** → 200. The document write endpoint (note edits / `update_meeting`).

**Hash-routing map** (use for CLI deep-links / tests): `#/` Home · `#/list/__section__private` My notes · `#/list/<folderId>` folder · `#/meeting/<id>` · `#/shared-with-me` · `#/recipes` · `#/people` · `#/companies` · `#/settings/{preferences,profile,calendar,notifications,integrations,workspace,team,billing,referrals}`.

**Works at the API but NOT wired into the desktop read UI** (so unobservable by driving; reachability known only from probes):
- `get-pre-meeting-briefs` → 200 `{"briefs":[]}` via probe, but **not fetched** when opening an upcoming meeting in the desktop app. Real but UI-dormant on this account.
- `get-workspace-analytics` → 400 (callable), no desktop surface.
- `get-people` / companies → not fired; people/companies views render client-side from synced data.

**Confirmed deferred (consistent across asar + probe + walk):** `search-meetings-turbopuffer` (dev-tools-only, `Cmd+Shift+D`), `get-action-items` / `get-follow-up-emails` (403 gated on this plan).

**Other insights:**
- `get-entity-set` is a generic entity store, called with `entity_type` ∈ {`chat_thread`,`chat_message`,`list_rule_suggestion`} — Granola's client-side entity abstraction.
- `get-chat-models` lists `claude-opus-4-8` "Opus 4.8" `is_new:true` — current.
- Production **semantic search = Chat** (`chat-with-documents`); the `Cmd+K` search is purely client-side.
- Every chat request ships `messageContext.directoryContext.myNotes[]` (the folder list) as grounding context.
