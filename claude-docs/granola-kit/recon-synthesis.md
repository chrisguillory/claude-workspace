# Recon synthesis — granola-kit plan pressure-test (2026-05-29)

Capstone of the pre-implementation pressure-test (Opus 4.8 re-validating Opus-4.7-era recon). Ties together `asar-diff-7277.md`, `reprobe-findings.md`, `buried-insights.md`, `capture-session-7277.md`, `chat-stream-sample.json`, `walk-endpoint-inventory.md`, and the live official-MCP exercise.

## Verdict: **GO** — plan is sound; fold in the corrections below

The plan's foundations all held up: bleeding-edge private-HTTP approach (not the local encrypted SQLite, not the official MCP), layered architecture, dual CLI+MCP. **(Update 2026-06-01: a subsequent validate-plan pass found 2 blocking-class errors this verdict missed — the chat parser spec (B1) and the `SubsetModel`/`OpenModel` drift-base strategy (B2) — both since corrected. See `validate-plan-findings.md`.)** The pressure-test (a) corrected several 4.7-era factual errors — most importantly device-id — (b) refined the tool roster against what's actually reachable on this account/build, and (c) produced exact specs for the central `chat_with_meetings` tool. Net effort is roughly unchanged: device-id scope removed offsets people/companies added.

## Corrections by area (each grounded in a verified finding)

### Auth / identity — `clients/auth.py`, `clients/identity.py`
1. **Device-id = `sha256(IOPlatformUUID)`** (upper-case dashed form from `ioreg -d2`). Verified black-box (`sha256("7F7E…DA91")==71a05389…`) *and* white-box (asar) *and* matches the shipped patch + `granola-client`. **DELETE the planned `identity.json` + `granola-kit identity capture` CLI + "background research on derivation" — that apparatus solved a non-problem.** (Scope reduction.)
2. **Read `X-Client-Version` from `Info.plist` at runtime** (`CFBundleShortVersionString`, currently 7.277.1) — not a hardcoded `7.220.0`. Auto-tracks upgrades.
3. **6 identity headers**, not 7: `X-Client-Version`, `X-Granola-Platform: macOS`, `X-Granola-Os-Version`, `X-Granola-Device-Id`, `X-Granola-Workspace-Id`, plus `Authorization`. **No `X-Granola-Source`.** `X-Granola-Time-Zone` is sent **per-call on auth endpoints only**, not as a global gating header.
4. **Drop the hardcoded `User-Agent`** — not gating; Chromium sends one anyway. One less stale string.
5. **Detect the "Unsupported client" gate: it's `HTTP 200` with body `{"message":"Unsupported client"}`** — `raise_for_status()` will NOT catch it. The client must inspect for the `message` envelope and raise `GranolaUnsupportedClientError`, else a `SubsetModel` swallows it into an empty result.
6. **Token refresh is mandatory + must be async.** Granola desktop no longer writes refreshed tokens back to `supabase.json`; the file's `workos_tokens` is a **JSON string inside JSON** (parse twice). Refresh via `POST /v1/refresh-access-token {refresh_token}` (+ identity headers, no Authorization). Current code uses sync `httpx`/`subprocess` in an async stack — make it async.
7. **Top future break-risk: `supabase.json` encryption.** 7.277.1 ships `safeStorage`/Keychain wrapping behind `encrypted_supabase_storage` (defaults TRUE); the file is plaintext today only because it predates the update. `auth.py` must detect-and-degrade (and we may later need a Keychain read path).

### Tool roster — final v1
**Verified-working, ship:**
- Meetings: `list_meetings` (+`time_range` enum: this_week/last_week/last_30_days/custom), `get_meeting`, **`get_meetings(meeting_ids)` → `get-documents-batch` `{document_ids:[...]}`** (the real batch endpoint, not singular `get-document-metadata`), `update_meeting` (`update-document`/`upsert-document`), `delete_meeting`, `restore_meeting`, `list_deleted_meetings`.
- **Chat (primary semantic): `chat_with_meetings`** — full spec below. Plus `get_chat_citation`, `list_chat_models`.
- Notes: `download_summary`, `download_note`, `download_transcript`.
- Folders: `list_folders`, `get_folder`, `create_folder`, `add_meeting_to_folder`, `remove_meeting_from_folder`.
- Workspaces: `list_workspaces`, `create_workspace`, `delete_workspace`.
- Account: `get_account_status`, `list_feature_flags`.
- Exports: `create_data_export`. URLs: `resolve_url`.
- **Briefs: `get_pre_meeting_briefs`** — works (200). NOTE `pre_meeting_brief_surface_v2=chat`: briefs are delivered through chat, so this may be largely redundant with `chat_with_meetings`. Ship as a thin tool; revisit.

**Promoted from deferred (discovered viable in the walk):**
- **`list_people`** → `get-people` (pecan), direct. Returns `{name, job_title, company_name, email, hd, links, avatar}`. (`{}` returned only self — full-attendee params TBD at impl.)
- **`list_companies`** → **derived** client-side by grouping `get-people` on `company_name` (no `get-companies` endpoint exists; we replicate the desktop's own rollup — fits "bring the best into the fold").

**Deferred (empirical reasons, not guesses):**
- `search_meetings` / `search-meetings-turbopuffer` — flag-enabled + structurally client-callable, but **403 Forbidden (feature-gated on this plan)** with the correct raw shape [corrected by validate-plan M5; the earlier "no client path / 400 not-gated" was a wrong-shape misread]. Chat is the semantic primary.
- `get_workspace_analytics` — flag-enabled but **no desktop surface** fires it (web/admin-only).
- `list_action_items`, `list_follow_up_emails` — **403 "feature not enabled"** on this plan (`get-follow-up-emails` is on `berry`, not cinnamon).
- `download_data_export` — create works; download never observed.

### Implementation specs
- **`chat_with_meetings` / `ChatService`** (see `chat-stream-sample.json`): `POST stream.api.granola.ai/v1/chat-with-documents`, body `{thread_id, chat_history:[{role:"USER", text, messageContext:{mode:"all", currentViewContext, includeTranscripts:false, directoryContext:{myNotes:[{id,name,noteCount}]}}}]}`. Response = chunks joined by literal `-----CHUNK_BOUNDARY-----`; types: `output_delta`/`text_with_citations` (answer in `plain_text` + `response_lines[].answer_text`/`.citations`), `output_delta`/`tool_call` (server ReAct, e.g. `listMeetings`), terminal `stream_completed`. **Final answer = `stream_completed.responseText`** (verified byte-identical to the last `text_with_citations.plain_text`); citations = flatten `response.response_lines[].citations` (tolerate `null`); tool trace = `toolCalls[]`. Use the terminal chunk, not the deltas; do not reassemble `argsDelta` (invalid JSON). [corrected 2026-06-01 by validate-plan B1 — the earlier "not `stream_completed`" was backwards]
- **Preserve the dual-cache's surgical `_invalidate_caches_for_document` eviction** (load-bearing for `list_meetings` perf). **Lift `analyze_markdown_metadata`** too (second pure util alongside `prosemirror.py`).
- **Host map fixes:** `update-action-item` maple→**chia**; `get-follow-up-emails`→**berry**; `get-people`→**pecan**; `get-pre-meeting-briefs`→maple. Otherwise the plan's host map is 100% accurate vs 7.277.1.

### Official MCP validation (now OBSERVED, not inferred)
Authenticated + exercised all 6 tools live: `get_account_info`, `list_meetings` (`time_range` enum + `folder_id`), `get_meetings` (`meeting_ids`, max 10), `get_meeting_transcript` (`meeting_id`), `list_meeting_folders`, `query_granola_meetings` (NL synthesis + `[[n]](url)` citations). **Confirms** the plan's `meeting_id`/`meeting_ids` naming, `time_range` enum, folder model, and `chat-as-primary` direction (the vendor's own MCP leads with `query_granola_meetings`).

## Production surface reality
The full CDP breadth walk (every view + all 9 settings tabs) exercised **47 endpoints** — the stable production read surface. The asar's 380 are mostly writes/streams/oauth/webhooks/admin not hit by reads. New functional endpoints found: `get-documents-batch`, `upsert-document`. Hash-routing map recorded in `walk-endpoint-inventory.md`.

## Recommended next steps
1. Apply these deltas to the plan; re-publish the gist.
2. (Optional) one adversarial validate-plan pass on the revised plan.
3. Begin implementation at the data+transport layer (schemas + `clients/`), since auth is now fully specified.

## Write endpoints — live-confirmed (2026-05-29, reversible throwaway-folder test)

Created + deleted a throwaway folder (nothing persists). All 200:
- `create-document-list-v2` — `{id, title, visibility, workspace_id}` → 200 (returns the list object)
- `add-document-to-list` — `{document_id, document_list_id}` → 200 `{"message":"Document added to list successfully","attioSync":null}`
- `remove-document-from-list` — `{document_id, document_list_id}` → 200 `{"message":"Document removed from list successfully"}`
- `delete-document-list-v2` — `{id}` → 200 `{"success":true}`

→ `create_folder` / `add_meeting_to_folder` / `remove_meeting_from_folder` / folder-delete tool payloads are ground-truth confirmed. Bodies are sent **raw** (not `{input:…}`-wrapped) despite the asar wrapper signature. Still asar-only (low priority, heavier to live-test): `create-workspace` / `delete-workspace`. `update_meeting`/`delete_meeting`/`restore_meeting` use `update-document` (`deleted_at` toggle), captured earlier.
