# Re-probe findings — dropped endpoints, correct identity (7.277.1)

**Date:** 2026-05-29. **Method:** authenticated read-only probes against prod, using the proven `granola-mcp/helpers.py` identity (verified `sha256(IOPlatformUUID)` device-id, `X-Client-Version` from plist = 7.277.1, `X-Granola-Platform: macOS`, real workspace-id). Auth path validated end-to-end: the May-12 token auto-refreshed via `/v1/refresh-access-token`; sanity `get-document-set` → 200. Hosts/payload shapes from the 7.277.1 asar `schemas-*.js` map + `api-*.js` wrappers.

This overturns 4.7's "dropped" decisions, which were made with malformed headers/hosts.

| Endpoint | Host (verified) | Body | Status | Verdict |
|---|---|---|---|---|
| `get-pre-meeting-briefs` | `maple.api.granola.ai` | `{}` | **200 `{"briefs":[]}`** | ✅ **WORKS** — reopen for v1. 4.7's 404 was wrong host/headers. |
| `search-meetings-turbopuffer` | `api.granola.ai` | **raw** `{searchType,keywords[],limit,includeShared}` | **403 Forbidden** (stable 3/3, verified 2026-06-01) | ⛔ **Feature-gated on this plan** (same class as action-items/follow-ups). [CORRECTED 2026-06-01: the original `{input:}`-wrapped probe got 400 and this row wrongly read "callable, not gated" — the RAW shape reveals the 403 gate.] Stays deferred. |
| `get-workspace-analytics` | `maple.api.granola.ai` | `{"input":{…}}` | **400 Bad Request** (3 field variants tried) | 🔓 **Callable, not gated** — reopen candidate; payload TBD. |
| `get-action-items` | `maple.api.granola.ai` | `{}` | **403 "This feature is not enabled"** | ⛔ Genuinely gated on this plan — correctly deferred. |
| `get-follow-up-emails` | `berry.api.granola.ai` | `{}` | **403 "This feature is not enabled"** | ⛔ Genuinely gated — was misdiagnosed as 404 (4.7 used `cinnamon`; real host is `berry`, returns 403). |

## Net effect on the v1 roster

- **+`get_pre_meeting_briefs`** — empirically working; promote into v1.
- **`search_meetings` (turbopuffer)** — **[CORRECTED 2026-06-01, validate-plan M5]** the raw keywords shape returns **403 Forbidden** (feature-gated on this plan); structurally client-callable but authz-blocked. Stays **deferred** (NOT "reopen"). The earlier "client-callable, not gated" was a wrong-shape (400) misread.
- **`get_workspace_analytics`** — reachable; reopen candidate pending payload.
- **action items + follow-up emails** — genuinely feature-gated (403) on this workspace/plan; stay deferred to `possible-features.md`. The "ship only verified-working tools" principle holds — these are correctly out.

## Payload shapes recovered from the asar (for the reopened endpoints)

- `search-meetings-turbopuffer` wrapper: `M(e,'search-meetings-turbopuffer',{input:t})` — but the **wire body is raw `t`**, not `{input:}`-wrapped. The raw `{searchType:'keywords', keywords:[…], limit, includeShared}` shape → **403 Forbidden** (gated); the `{input:}`-wrapped shape → 400. The **gate**, not the shape, is the blocker. (Confirms the general raw-vs-`{input:}` wire invariant.)
- `get-workspace-analytics` wrapper: `M(e,'get-workspace-analytics',{input:t})` — `t` fields unknown; capture needed.
- `get-follow-up-emails`: `M(e,'get-follow-up-emails',{})` (no input) — host `berry`.
- `get-pre-meeting-briefs`: `{}` works at `maple`.
