# Live capture session — Granola 7.277.1 (CDP-driven)

**Date:** 2026-05-29. **Method:** mitmdump on :8080; Granola relaunched through it and driven via CDP/Playwright. Capture at `~/granola-capture/granola-7277.mitm`. Stream fixture at `docs/chat-stream-sample.json`.

## The relaunch recipe that works (reproducible)

The single-instance lock + tray means a scripted `pkill` does NOT fully quit Granola — a flagged relaunch then hands off to the surviving instance and the flags never apply (no proxy, no CDP). The working sequence:

1. **User** quits via the tray menu → **"Quit Completely"** (not plain "Quit"). A scripted quit is insufficient.
2. Relaunch the binary **directly** (not `open -a`, which drops `--args`):
   ```bash
   HTTPS_PROXY=http://127.0.0.1:8080 HTTP_PROXY=http://127.0.0.1:8080 \
     /Applications/Granola.app/Contents/MacOS/Granola \
     --ignore-certificate-errors --proxy-server=127.0.0.1:8080 --remote-debugging-port=9222 &
   ```
   - `HTTPS_PROXY` env is what actually routes Granola's traffic (the 4.7-proven method; a prior compaction summary had this backwards re `--proxy-server`).
   - After a *full* quit, the direct launch DOES honor `--remote-debugging-port` → **CDP works** (Granola does NOT strip it in prod; the earlier "CDP stripped" conclusion was an artifact of the single-instance handoff).
3. Attach: Playwright `connect_over_cdp("http://127.0.0.1:9222")` → `contexts[0].pages[0]` is the `app://ui/` window.

## chat-with-documents — the v1 primary (`chat_with_meetings`)

**Host:** `stream.api.granola.ai/v1/chat-with-documents`. **Trigger:** the in-app **Chat** (NOT the `Cmd+K` search, which is client-side).

**Request shape (live, 7.277.1):**
```jsonc
{
  "thread_id": "<uuid>",
  "chat_history": [
    {
      "role": "USER",
      "text": "<question>",
      "messageContext": {                    // NOTE: messageContext, NOT chat_context (asar-diff guess was wrong)
        "mode": "all",
        "currentViewContext": {"view":"global","numTotalDocuments":N,"attendedNotes":N,"sharedNotes":N},
        "includeTranscripts": false,
        "directoryContext": {"myNotes": [{"id","name","noteCount"}, ...]}   // folder list + counts
      }
    }
  ]
}
```

**Response stream:** chunks joined by the literal delimiter `-----CHUNK_BOUNDARY-----` (real — earlier "fabricated" claim was wrong). For the captured query, 67 chunks:

| chunk `type` | `output.type` | count | meaning |
|---|---|---|---|
| `output_delta` | `text_with_citations` | 47 | answer deltas — `output.plain_text` (full replace each delta) + `output.response_lines[].answer_text` / `.citations` |
| `output_delta` | `tool_call` | 18 | server-side ReAct (`name:"listMeetings"`, args, `id:"toolu_…"`) |
| `stream_completed` | — | 1 | terminal marker |

**Parser spec for `ChatService` (corrected 2026-06-01, validate-plan B1):** split on `-----CHUNK_BOUNDARY-----`, skip the one trailing empty part, and read the single terminal **`stream_completed`** chunk — it is the authoritative result: answer = **`responseText`** (verified byte-identical to the last `text_with_citations.plain_text`), citations = flatten **`response.response_lines[].citations`** (tolerate `null` — 2 of 8 lines were null), tool trace = **`toolCalls[]`** (`{id,name,args,result}`, fully resolved). The `output_delta` chunks (`text_with_citations`, `tool_call`) are incremental-UI deltas an MCP collect-then-return tool ignores. Do **not** reassemble `tool_call.argsDelta` — the 18 fragments do not form valid JSON. (Supersedes the earlier claim that the answer is NOT in `stream_completed`; it is, as `responseText`.)

**Companions captured:** after the chat, the app calls `llm-proxy {prompt_slug:"chat-thread-title"}` (gpt-5-mini) to auto-title the thread; `get-chat-models` lists models incl. `claude-opus-4-8` "Opus 4.8" `is_new:true`.

## turbopuffer search — 403 FEATURE-GATED (corrected 2026-06-01, validate-plan M5)

The in-app `Cmd+K` search is **client-side** (fired zero server search calls). `search-meetings-turbopuffer` is reachable + structurally client-callable, but with the correct **raw** `{searchType:'keywords',keywords[],…}` shape it returns a **stable HTTP 403 `Forbidden`** — **feature-gated on this plan**, same class as action-items/follow-ups. (The original probe sent the `{input:}`-wrapped shape, got 400, and mis-read it as "reachable, not gated"; the renderer's dev-tools "Turbopuffer tab" exists but is moot given the 403.) The production semantic primary is **chat-with-documents** (`chat_with_meetings`). **Deferred — feature-gated, not v1.**

## workspace-analytics — not desktop-exposed

No Analytics/Insights surface in the desktop sidebar. `get-workspace-analytics` is reachable (400, needs payload) but appears web-/admin-only — not capturable by driving the desktop app. Defer; revisit if a desktop or web surface that calls it is found.

## Net v1 surface impact

- **In (newly confirmed):** `chat_with_meetings` (fully spec'd here) + `get_pre_meeting_briefs` (works, 200).
- **Out / deferred (better-understood reasons):** `search_meetings` (turbopuffer **403 feature-gated** — raw shape; chat covers the use case), `workspace_analytics` (not desktop-exposed), `list_action_items`/`list_follow_up_emails` (403 gated), `download_data_export` (not observed).
