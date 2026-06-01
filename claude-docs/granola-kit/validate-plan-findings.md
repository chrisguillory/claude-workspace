# validate-plan synthesis — granola-kit plan (2026-06-01)

Three independent Opus validators empirically attacked the revised plan (auth/transport · tool-roster/reachability · chat-spec/schemas/architecture). All ran read-only live probes + parsed the real capture/asar. Aggregated, deduplicated, severity-ranked below.

## Verdict

**Conditional GO — architecture confirmed solid; revise the blocking + major items, then build.** The scaffolding is verified-correct: tool count (28), endpoint→host map (26/26 byte-exact in asar), device-id = `sha256(IOPlatformUUID)` (3 ways), all read endpoints 200, deferred-403s, dual-cache + pure utils real, all 6 cc-lib integration points exist, module split justified by precedent. But the plan's self-declared **"GO, no blocking flaws" capstone was overstated** — two validators independently found blocking-class errors at the two most novel surfaces (chat parser, drift schema), both reproduced against live data.

## BLOCKING (fix before implementing the affected step)

**B1 — Chat parser spec is wrong on its central claim.** (chat-arch, empirically reproduced on the 67-chunk live stream)
- `stream_completed.responseText` is **byte-identical** (1234 chars, `==`) to the last `text_with_citations.plain_text`. The plan's "final answer ≠ `stream_completed`" is false.
- The plan names a non-existent field (`.response` is a dict `{response_lines}`; the answer string is `.responseText`), and misses `stream_completed.toolCalls` (resolved tool trace) + `.response.response_lines[].citations` (flat ID list).
- The mandated delta-accumulation path is the *harder* one and its `tool_call.argsDelta` fragments **don't concatenate to valid JSON**.
- **Fix:** `ChatService` = collect body → split on `-----CHUNK_BOUNDARY-----` → use the single `stream_completed` chunk: `answer=responseText`, `citations=flatten(response_lines[].citations)` (tolerate `null`), `tool_calls=toolCalls[]`. Drop the delta path; `ChatStreamChunk` delta schema becomes unnecessary for v1. Tolerate the trailing empty chunk.
- **Doc corrections:** the error originated in `capture-session-7277.md:51` and propagated to `recon-synthesis.md:42` + the plan. Fix all three.

**B2 — Drift base-class strategy is unimplementable + fail-closed in this workspace.** (chat-arch, reproduced with the real env)
- `SubsetModel` (`extra='ignore'`) has `model_extra is None` → **no hook to log dropped fields**, so "SubsetModel + logged warning" is impossible.
- `OpenModel` for ProseMirror **crashes** here: `~/.claude/settings.json` sets `CC_OPEN_MODEL_EXTRA_FORBID=1` globally → fail-closed on the most drift-prone schema.
- `test_schemas_drift.py` "warn not crash" is unsatisfiable as written.
- **Fix:** custom base = `extra='allow'` + `model_validator(mode='after')` logging `self.model_extra` keys, hardcoded `extra='allow'` independent of the env toggle (or the migration explicitly unsets it for granola-kit). `SubsetModel` stays only where we deliberately read a subset and never want the rest.

## MAJOR

**M1 — Identity headers are NOT "gating."** (auth, proven: fresh token + ZERO identity headers → 200 with real data on both api + pecan; roster: existing code sends 5 headers and works) — the headers are what the desktop *sends*, not what the server *requires*. Re-label "6 gating headers" → "identity headers the desktop mirrors; server tolerates absence with a valid token." **Only token refresh is load-bearing.**

**M2 — Drop `X-Granola-Workspace-Id` + `_active_workspace_id()`.** (auth) Non-gating, non-functional for reads (byte-identical data with/without), and its only local source (`user-preferences.json`) is already frozen by the `.enc` migration → would read stale today, `None` tomorrow. If a write needs workspace scoping, pass it in the request body.

**M3 — The 200-`{"message":"Unsupported client"}` guard must match on VALUE, not key presence, and should refresh-and-retry.** (auth) 243 legitimate 200s in the capture carry a top-level `message` (`"Refreshed 1 user."`) → a presence check false-positives. The envelope means *stale/invalid token sent without identity*, so: on exact value `"Unsupported client"` → refresh token + retry once → raise `GranolaUnsupportedClientError` only if it persists. Pair the test fixture with a negative (`"Refreshed 1 user."`) that must NOT raise.

**M4 — Encryption migration is a near-term outage risk, not hypothetical.** (auth) Plaintext `supabase.json` froze 2026-05-12; `supabase.json.enc` updates daily; on-disk token is ~20 days expired. granola-kit reads `refresh_token` from the plaintext file — once Granola stops writing it (or the refresh token rotates), the server is **dead**, not degraded. Elevate to #1 operational risk; **consider implementing the `safeStorage`/Keychain decrypt path in v1** rather than deferring. Also: persist the rotated `refresh_token` to the token-cache (WorkOS may rotate single-use).

**M5 — Turbopuffer drop-reason is wrong; a shipped doc is now empirically false.** (roster) Correct raw shape → stable **403 Forbidden** (authz-gated, same class as action-items), NOT "dev-only/no-client-path." `reprobe-findings.md` lines 10/18 say "400, callable, not gated, drop rationale false" — now wrong (it sent the `{input:}`-wrapped shape and saw 400). **Fix:** plan reason → "403 feature-gated"; correct/retract reprobe-findings.md; keep dropped. (buried-insights §1.1 already warns against "restoring the killer feature.")

**M6 — `download_summary`/`download_note`/`download_transcript` mis-annotated as writes.** (roster) They call `get-document-panels`/`get-document-transcript` — pure reads. Set `readOnlyHint=True` (drop `readOnlyHint=False, idempotentHint=True`). The local file write is a local-FS side effect, not a server mutation.

**M7 — `list_people`/`list_companies` return only self.** (roster, 5 param variants all → 1 row) "Verified-working" at HTTP, near-useless in practice on the testable account. Strong defer-to-v1.1 candidates, or ship with an explicit "may return only your own profile" caveat.

**M8 — `create_workspace`/`delete_workspace` ship unverified — contradicts the plan's own "every tool empirically verified" contract.** (roster) Concrete risk: the asar wrapper is `{input:t}` but every live-tested sibling took a **raw** body → shipped shape is a coin-flip. Either live-test with a throwaway workspace at impl, or defer to v1.1.

**M9 — `aiocache` dependency silently dropped.** (chat-arch) The dual-cache that "must survive" runs on `aiocache`; `pyproject.toml` omits it. Decide: reimplement dict-only (recommended — the set-index is one entry) or add the dep.

**M10 — `directoryContext` population is an unspecified hidden dependency.** (chat-arch) Each chat carries 14 folders + a `team` block as RAG scope. `ChatService` must build `directoryContext.myNotes` per call — implying a `list_folders` call per chat (latency/cost), or send minimal and accept degraded recall. Unaddressed.

**M11 — Missing migration step: extract the real chat fixture.** (chat-arch) `chat_stream_chunks.jsonl` doesn't exist; the on-disk `chat-stream-sample.json` is 6 of 67 chunks with NO `tool_call` chunk. Add a step to write the full 67-chunk stream from the `.mitm` to the fixture; `test_services_chat.py` can't run without it.

## MINOR / NIT (apply on touch)

- Deferral taxonomy inconsistent: 403s (turbopuffer/action-items/follow-ups) vs 400/no-path (analytics) — normalize wording. (roster)
- `get-people` + `get-feature-flags` are **top-level JSON arrays** → `RootModel[list[...]]`, not dict containers. (roster)
- Raw-vs-`{input:}` is a **wire invariant**: send bodies raw; the asar `{input:t}` is the JS arg, not the wire body (proven: `get-chat-citation` raw→200 / wrapped→400). Document in `granola_api.py`. (roster + chat-arch + auth, convergent)
- Claim "get_meetings→get-documents-batch" is framed as a delta but existing `granola-mcp.py` already does it (`_get_documents_by_ids`) — no-op migration. (roster)
- `chat_with_meetings` creates+titles a persisted thread server-side → `readOnlyHint=True` questionable; `idempotentHint=False`. (roster)
- `Content-Type` is a per-request concern (httpx sets it for `json=`), not an identity header. (auth)
- `from_pid_walk` requires an active session + codesign; fine for MCP lifespan, would fail from CLI. (chat-arch)
- prosemirror lift must include `process_list_item` + `extract_text`; `extract_text` has a redundant inner `import re`; `analyze_markdown_metadata -> dict` is untyped (repo rule) → make it a Pydantic model on lift. (chat-arch)
- `test_tools_meetings.py`/`test_tools_account.py` are happy-path passthroughs — cut or make them assert annotations/error-translation. (chat-arch)
- Effort estimate ~20.75 hr is optimistic (greenfield chat + spec rework + missing fixture step) → ~22–25 hr. (chat-arch)
- Request snippet omits `messageContext.additionalContext` + `directoryContext.team` present in the real capture. (chat-arch)

## Convergent meta-findings (multiple validators, independently)

1. **The "GO, no blocking flaws" capstone was overstated** (chat-arch + roster).
2. **Raw-vs-`{input:}` wire invariant** surfaced from 3 angles (get-chat-citation, argsDelta, header builder).
3. **Recon docs now contain errors to correct:** `reprobe-findings.md` (turbopuffer 400→403, "not gated"→gated), `capture-session-7277.md`+`recon-synthesis.md` (chat parser spec), `recon-synthesis.md` (the capstone claim). *Improve project health / fix-the-source: correct them, don't ship known-wrong intel.*

## Alternatives the validators surfaced

- **Leaner v1 (24 tools):** defer `create_workspace`/`delete_workspace` (unverified writes) + `list_people`/`list_companies` (self-only) to v1.1 → every v1 tool demonstrably useful + verified. (roster; ~23 if briefs also deferred)
- **stream_completed-only `ChatService`** (~6 lines) + defer the `ChatStreamChunk` delta schema. (chat-arch)
- **Minimal-auth transport / Keychain-native decrypt in v1** (ideal-state "fix the source" vs reading the dying plaintext file). (auth)
- **No new drift base** — the existing `extra='forbid'` fails loud (how all 8 current drift fields were found); add `extra='allow'`+logging only to the 3 upstream doc classes, not a 3-way taxonomy. (chat-arch)

## Judgment calls for the user

1. **v1 scope:** keep 28 (fix the verification debt — live-test workspace writes, caveat people/companies/briefs) vs leaner 24 (defer the 4 weak tools to v1.1).
2. **Auth source:** implement Keychain/`safeStorage` decrypt in v1 (survives the active migration) vs plaintext-read + degrade (plan's current; known expiry).
3. Everything else (B1, B2, M1–M3, M5, M6, M9–M11, all MINOR/NIT + doc corrections) is empirically-settled — apply directly.

## PR #3 integration + superset (2026-06-01)

`chrisguillory/granola-mcp` **PR #3** ("Fix 401: read Granola's live encrypted credential store") — the M3 AI independently hit and fixed the exact failure `validator-auth` flagged as the #1 risk (M4). Read the diff; the implementation is correct.

**Resolves M4 + answers the auth judgment call.** granola-kit's `clients/auth.py` ports PR #3's chain (macOS):
```
Keychain 'Granola Safe Storage' (security find-generic-password -w -s ...)
  -> safeStorage v10 decrypt storage.dek  (AES-128-CBC, key=PBKDF2-HMAC-SHA1(pw,'saltysalt',1003,16B), IV=16 spaces, PKCS7)
  -> base64 -> 32-byte DEK
  -> AES-256-GCM(nonce=blob[:12], ct+tag=blob[12:]) decrypt supabase.json.enc -> json -> workos_tokens (double-parse)
```
Encrypted-store-first, **plaintext fallback** (older installs / non-macOS). Add `cryptography` to deps (alongside the `aiocache` decision in M9). `get_auth_token()`/refresh logic unchanged — just receives live tokens. The "self-persist rotated refresh_token" idea (M4 sub-point) becomes **moot** — the live store IS the source of truth; read it each cold start.

**Cross-machine reality (superset):** on **M3** the plaintext file is fully dead (refresh_token rotated → self-refresh 401s → encrypted store required); on **M4** the plaintext refresh still works (our validators refreshed fine). The migration is **per-machine and in-flight**. Encrypted-first/plaintext-fallback correctly handles both — but see M12.

**M12 — NEW (mesh/headless Keychain constraint).** The encrypted-store read needs macOS Keychain access via `security`, which requires the **login keychain unlocked** + a one-time GUI "Always Allow" approval for the calling process. The plan's Step-13 **mesh smoke** (`crb execute -t M2,M3 'granola-kit list-meetings'`) will hit this: a headless/locked-keychain mesh node can't read the key → falls back to plaintext → **dead on a migrated node**. Neither PR #3 (M3-interactive) nor the validators (M4-local) surfaced this. The plan must add a per-machine keychain-approval step (or document the constraint) for mesh use; the bleeding-edge "works across mesh" claim depends on it.

**M13 — NEW (write-tool ownership).** PR #3 verification: `update_meeting` returns **403 on meetings owned by a different account** (shared/collaborator = read-only). `update_meeting`/`delete_meeting`/`restore_meeting` only succeed on **owned** meetings — annotate + document this constraint (composes with M6's annotation fixes).

**Minor PR critiques (for the port):** lenient PKCS7 unpad (doesn't verify all pad bytes — harmless here); broad `except Exception: return None` masks a real decrypt bug silently — per the repo's *actionable errors at boundaries*, log the exception before falling back; no handling of a locked/denied Keychain (returns None → plaintext → dead on migrated machines — ties to M12).

**Action:** recommend **merging PR #3** into granola-mcp so the source `helpers.py` carries the encrypted-store read; granola-kit then ports the merged version (the migration's "lift helpers.py → clients/auth.py" should lift the post-merge file).
