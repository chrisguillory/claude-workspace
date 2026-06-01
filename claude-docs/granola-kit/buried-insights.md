# Buried Insights — mined from the 13-compaction session history

**Source:** `transcript-miner` agent over the main granola session (`019e26cb`, 103 MB, 30,732 lines, 13 compactions) + ~80 subagent transcripts, cross-checked against the current plan + docs. Captured 2026-05-29.

**Corpus-hygiene note:** the `019e26cb` session was reused for **unrelated** work (the Mainstay "direct-listings marketplace" / TXN-1875 effort — `DirectListing` APIs, `brokerage_service.py`, ECR/CDK/Terraform). That is NOT Granola and is excluded here. Some search hits that look like "user pushback" (full_address immutability, withdraw response shape) belong to that other project.

---

## Category 1 — Design decisions DEBATED then REVERSED

**1.1 — `search_meetings` (Turbopuffer): killer feature → shipped in v1 → dropped.** The desktop-app enumeration called it "the single most valuable add… replaces brittle client-side title-substring filtering with real semantic search." An intermediate revision shipped it; the mitm capture reversed it (`search-meetings-turbopuffer` never called by the client → server-side only → replace with `chat_with_meetings`). Deliberate, capture-grounded — don't let a future pass "restore the killer feature." Input shape was never verified.

**1.2 — Tool count: 19 → 28 → 25.** Original 19; enumeration pushed ~28 (adding gated domains); capture cut to 25 (drop 7 gated/server-side, add 4 verified chat/account). `private-api-enumeration.md` and `upstream-mcp-survey.md` still say "19-tool roster" throughout — stale.

**1.3 — Action items / briefs / follow-ups: "add to v1" → dropped as gated.** Enumeration pushed them; capture killed them (403 "Feature not enabled" / 404). User drove this (see 2.1).

**1.4 — Token refresh: "just open Granola.app" → "refresh flow is mandatory in-code."** Empirically reversed: Granola desktop no longer writes refreshed tokens back to `supabase.json` (post-March DB-encryption likely moved token storage; file observed untouched for 2 days while token 44h expired). Refresh is the ONLY way to get a usable token — not a nicety.

---

## Category 2 — User corrections / pushback / strong preferences

**2.1 — "Why the fuck would we write something for something that we can't even use."** Pushback on adding `list_action_items` while 403-gated. → core principle: *ship only verified-working tools; gated/server-side endpoints are deferred, not shipped half-working.*

**2.2 — "don't we have real traffic? You can't analyze any of that real traffic…"** Pushback on probing with guessed payloads instead of mining the real capture. → empirical-surface.md is "ground truth"; new endpoint shapes come from capture, not speculation.

**2.3 — "I don't know why you keep calling something M1."** Machine is **M4** (`chris-K4442D3Q0X`, work mac). Avoid reintroducing "M1."

**2.4 — "Bleeding edge, full feature set, no backcompat" (said twice) + "the value of your fork IS the writes + private notes + workspace mgmt — don't mirror the official MCP's read-only 6."** Settled design constraint. Don't trim to parity with the official surface or cap batch sizes.

**2.5 — Safety-classifier hits (TWICE), operational constraint:** (a) probing the API with spoofed client signatures was flagged as safety-check bypass; (b) relaunching Granola through mitmproxy to intercept live HTTPS was flagged as credential-interception unless explicitly authorized in-session. The plan's "periodic reconnaissance" + `granola-kit identity capture` (sniff a mitm flow) will hit these walls without explicit per-session user authorization.

---

## Category 3 — Recon-subagent findings summarized away

**3.1 — `granola-client` PyPI (MIT, Anjor Kanekar, `github.com/anjor/granola-py-client`) dissection — highest-value buried artifact.** Its source comments: identity headers "must mirror what the live Granola Electron app sends, because the API gates on them and returns 'Unsupported client' otherwise." Details:
- **Device-ID = `sha256(IOPlatformUUID)`**, derived live via `ioreg -d2 -c IOPlatformExpertDevice` (macOS) / `MachineGuid` (Windows). Omitted if lookup fails; lib treats it as best-effort. **Contradicts the plan (see 5.2).**
- **Version read dynamically** from `Info.plist:CFBundleShortVersionString` (fallback chain), bare semver, no `Granola/` prefix. **Plan hardcodes `7.220.0` (see 5.3).**
- **No `User-Agent` override** — httpx default; moved to `X-Client-Version`. **Plan lists a full Electron UA (see 5.6).**
- Auth: `supabase.json → workos_tokens.access_token`; refresh POSTs `/v1/refresh-access-token` `{refresh_token}` with no auth header; caches `access_token + expires_in + obtained_at`.
- MIT-licensed → copyable with attribution (matches workspace "fork or patch" principle).

**3.2 — `chat-with-documents` stream contract (from the renderer JS).** Final answer is **`stream_completed.response`** (the full response REPLACES the deltas) when `contentType==='text'` — NOT accumulated `text_delta`. Chunk types: `error`, `text_delta(.delta)`, `context_sources_delta(.numTotalSources/.scannedDocuments/.citationIds/.webSearch)`, `context_limit_reached`, `stream_completed(.reasoning/.response/.contentType/...)`, `outputs`, `output_delta(.index/.output)`. Citations accrue in `context_sources_delta.citationIds` (Set-dedup'd), not one field. NOTE: the live capture showed the inter-chunk wire delimiter as the literal `-----CHUNK_BOUNDARY-----` — reconcile with the JS (which splits on a minified var). Both can hold: split on the wire delimiter, parse each chunk's `type`, final = `stream_completed.response`.

**3.3 — `get-document-set` returns ALL docs, NO pagination** (working `models.py` docstring). **Contradicts empirical-surface.md** which says "paginated via cursor" (see 5.8).

**3.4 — Existing granola-mcp internals that must survive the refactor:**
- **Dual cache** — session-lifetime doc cache + per-doc dict with **surgical `_invalidate_caches_for_document` eviction after writes** (commit `12a9eb2`), "load-bearing for `list_meetings` performance." `services/cache.py::DocumentCache` must preserve surgical post-write invalidation.
- **Sync-in-async hazard** — `helpers.py` uses sync `httpx.post` (refresh) + sync `subprocess.run` (`ioreg`/`sw_vers`) inside an async stack. `clients/auth.py` should make refresh async / thread the sync bits.
- **`analyze_markdown_metadata`** — a second pure, liftable utility (plan only lifts `prosemirror.py`).

**3.5 — `GranolaDocument` drift fields:** `subscription_plan_id`, `privacy_mode_enabled` → `| None` (the model-relax fix); newer fields `was_trashed`, `is_primary_event_note`, `ydoc_state`, `ydoc_version`, `zoom_rtms_permission`, `document_user_role`, `is_scratchpad`; `DocumentSetEntry.has_ydoc`/`has_notes_ydoc`. Good seeds for `test_schemas_drift.py`.

**3.6 — `POSSIBLE_FEATURES.md` provenance:** Oct 2025 / Proxyman / Granola v6.298.0, when traffic was "100% `api.granola.ai`." Action-items, follow-ups, Turbopuffer, flavored subdomains, Spaces, Rooms, MCP-management, public-API-keys, cloud-agent are all post-Proxyman additions.

---

## Category 4 — Explicit deferrals

**4.1 — Official-MCP OAuth was never completed in the original session** → `upstream-mcp-survey.md` schemas are DOCS/INFERRED, never OBSERVED. (RESOLVED 2026-05-29: OAuth now complete; the 6 tools were exercised verbatim this session — survey should be upgraded from inferred to observed.)

**4.2 — `download_data_export` deferred** — create works (202), download never captured.

**4.3 — Local-RAG `query_meetings_with_synthesis` deferred** (needs document-search index) — the owned-end-to-end complement to server-side `chat_with_meetings`.

**4.4 — First interplay deliverable narrowed** to ProseMirror lift + manual document-search smoke test. Auto-index pipeline deferred. No skill this round.

**4.5 — Spaces (`pecan`, 11 endpoints) deferred** — coexists with workspaces, opt-in; no `delete-space` wrapper.

---

## Category 5 — Findings that CONTRADICT or are MISSING from the plan / empirical-surface.md

**5.1 — STALE:** dropped `search_meetings`/Turbopuffer + "19-tool roster" still live in `private-api-enumeration.md` and `upstream-mcp-survey.md`. Reconcile to the 25-tool reality or annotate as point-in-time.

**5.2 — CONTRADICTION (highest priority): device-id. → RESOLVED EMPIRICALLY 2026-05-29.** Plan/empirical-surface said "NOT `sha256(IOPlatformUUID)` — derivation TBD" + proposed `identity.json` + capture-CLI + background research. Verified directly: `sha256("7F7E1B9E-3912-5C2E-B80A-0B2F6CACDA91") == 71a05389…fa72f8e` (the captured device-id), i.e. **device-id = `sha256(IOPlatformUUID)` in the upper-case dashed form `ioreg` returns, no trailing newline.** The "different hash" conclusion was a 4.7 error (hashed a normalized form). **Action taken: `empirical-surface.md` corrected; plan must default device-id to `sha256(IOPlatformUUID)` and DELETE the `identity.json`/capture-CLI/background-research apparatus (scope removed).**

**5.3 — CONTRADICTION: version hardcoded vs dynamic plist read.** Plan: `'X-Client-Version': '7.220.0'` literal. Patch + library read `CFBundleShortVersionString` at runtime. Dynamic read auto-tracks upgrades (and matches the workspace `CCVersion` philosophy).

**5.4 — MISSING rationale: why token refresh is mandatory** (desktop app stopped writing refreshed tokens to `supabase.json`). Also: open question whether identity gating headers are required on `/v1/refresh-access-token` (likely yes).

**5.5 — MISSING: the "Unsupported client" gate returns HTTP 200**, body `{"message":"Unsupported client"}` — a top-level error envelope, not 4xx. `raise_for_status()` won't catch it; a SubsetModel could swallow it into an empty `list_meetings`. `clients/`/`exceptions.py` must detect the 200-with-`message` envelope and raise (e.g. `GranolaUnsupportedClientError`).

**5.6 — CONTRADICTION: `User-Agent` listed as a header to send, but it's not gating** and the gate-passing patch omits it. The capture shows a UA only because Chromium always sends one. Lean toward omitting (one less stale-drift string).

**5.7 — CHAT FIXTURE:** the plan's `test_services_chat.py` references `-----CHUNK_BOUNDARY-----`. That value DID appear in the live capture (so it's real, not fabricated), but the parser spec must key on chunk `type` + final = `stream_completed.response`, and the fixture must be a real captured stream. Confirm the delimiter against the current 7.277.1 renderer.

**5.8 — CONTRADICTION: `get-document-set` pagination.** empirical-surface.md says paginated-via-cursor; working `models.py` says no pagination. Shipped behavior wins unless re-capture proves otherwise.

**5.9 — Host-map assumptions to verify:** `remove-document-from-list` host is assumed (not in capture) though `remove_meeting_from_folder` ships v1; `get-chat-citation`/`get-chat-models` host assignment should be double-checked against the schema table.

---

## Net actionable deltas (priority order)

1. **Reverse 5.2** — default device-id to `sha256(IOPlatformUUID)`; demote `identity.json`/capture-CLI/background-research to fallback. (Removes scope.)
2. **Add 5.5** — detect the 200-`{"message":"Unsupported client"}` envelope as a raised exception; document that `raise_for_status()` won't catch it.
3. **Fix 5.3 + 5.6** — read `X-Client-Version` from the plist dynamically; drop the hardcoded Electron `User-Agent` (or mark non-gating).
4. **Rewrite the chat-stream spec (3.2 + 5.7)** — real captured fixture, `type`-keyed parser, final answer = `stream_completed.response`.
5. **Preserve `_invalidate_caches_for_document` surgical eviction (3.4)**; make token refresh async.
6. **Record 5.4 rationale** (desktop app stopped refreshing `supabase.json`).
7. **Reconcile stale 19-tool/`search_meetings` text (5.1)**; upgrade the official-MCP survey from INFERRED to OBSERVED (4.1, now unblocked).
