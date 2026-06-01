# Granola asar diff — 7.220.0 → 7.277.1 (adversarial re-analysis)

**Prior baseline:** `private-api-enumeration.md` (Granola.app **7.220.0**, analyzed 5/14)
**This analysis:** Granola.app **7.277.1** (`CFBundleShortVersionString`), asar dated 5/27, re-analyzed 5/29
**Bundle:** `/Applications/Granola.app/Contents/Resources/app.asar` (~56 MB), extracted to `/tmp/granola-asar-7277/`
**Method:** static ripgrep + Python brace-matching over minified JS; **device-id and platform claims verified empirically against this machine.**

## Source-of-truth files (hashes changed vs 7.220.0)

| Role | 7.220.0 | 7.277.1 |
|---|---|---|
| Endpoint→URL schema map | `schemas-62S41ciY.js` | `dist-app/assets/schemas-COPueaY7.js` (162 KB, **380 endpoints**) |
| HTTP client (`O` header builder, `M` request fn, wrappers) | `api-Dl530PM_.js` | `dist-app/assets/api-ZMfGOsS8.js` (33 KB) |
| Electron main (dup schema + device-id + token store) | `index.js` | `dist-electron/main/index.js` (9.5 MB) |
| Largest renderer chunk (turbopuffer caller, agent tool map) | (n/a) | `dist-app/assets/primary-C2f84e4V.js` (5.2 MB) |

---

## TL;DR — what the prior recon got wrong, what changed

1. **Device-ID derivation: prior analysis FAILED, and so did `empirical-surface.md`.** The header `X-Granola-Device-Id` **IS** `sha256(IOPlatformUUID)`. I proved it on this machine: `sha256("7F7E1B9E-3912-5C2E-B80A-0B2F6CACDA91")` = `71a05389cc0df4c3…`, **byte-matching the captured `71a05389…`**. `empirical-surface.md`'s "NOT sha256(IOPlatformUUID)" is **wrong**. granola-kit can compute the device-id at runtime — **kill the `identity.json` one-time-capture workaround.**
2. **`search-meetings-turbopuffer` IS called client-side.** The plan's load-bearing assumption ("server-side only, cannot call directly → drop `search_meetings`") is **false**. There is a typed wrapper *and* a live caller in the production renderer that POSTs it with the user's `access_token`. Input shape recovered. (Whether it's account-gated to 200 is unverified here — capture never exercised it — but it is structurally client-callable.)
3. **`X-Granola-Time-Zone` is NOT a global identity header.** It is attached *per-call* via `additionalHeaders`, only on `workos-auth-complete` and `refresh-access-token`. The prior enumeration listing it as a transport-layer gating header is **wrong**. The capture saw it only because the auth flow ran.
4. **All 30 task-critical endpoints (22 host-map + 7 "dropped" + chat/account extras) are STILL PRESENT in 7.277.1.** None removed. One host **moved**: `update-action-item` maple → **chia**.
5. **`X-Granola-Platform` is a lookup-table value, confirmed `macOS`** (table `{darwin:"macOS", win32:"Windows", web:"web"}`). Prior enum's `'darwin'` claim was wrong; capture's `macOS` was right.
6. **NEW latent risk: `supabase.json` is now encrypted-at-rest behind a flag that defaults TRUE** (`encrypted_supabase_storage`, Electron `safeStorage`/Keychain). The on-disk file is *still plaintext today* (not rewritten since the update), but a future re-auth could encrypt it and break granola-kit's `auth.py`. **This is the single biggest version-drift threat to the plan.**

---

## 1. VERSION-DRIFT DIFF (endpoints)

### Method limitation (state plainly)

The 7.220.0 schema file and the prior run's `/tmp/*.tsv` artifacts are **deleted** (asar re-downloaded 5/27). I **cannot** produce a byte-exact NEW/REMOVED diff over all ~380 endpoints — I lack the baseline set. CDN recovery of the old asar was not pursued (uncertain, token-expensive). What follows is rigorous for the **30 task-named endpoints** (verified directly against the 7.277.1 schema) plus host-count deltas and capability-keyword scans.

### 1a. The 22 plan host-map endpoints + chat/account — all PRESENT, hosts confirmed

Every endpoint in the plan's `_ENDPOINT_HOST` map exists in 7.277.1 at the **same host** the plan assigned. Verified from `schemas-COPueaY7.js`:

| Endpoint | Host (7.277.1) | Ver | Plan host | Match? |
|---|---|---|---|---|
| `get-document-set` | api.granola.ai | v1 | api | ✓ |
| `get-document-metadata` | api.granola.ai | v1 | api | ✓ |
| `get-document-panels` | api.granola.ai | v1 | api | ✓ |
| `get-document-transcript` | api.granola.ai | v1 | api | ✓ |
| `update-document` | api.granola.ai | v1 | api | ✓ |
| `create-document` | api.granola.ai | v1 | api | ✓ |
| `get-document-lists-metadata` | api.granola.ai | v1 | api | ✓ |
| `create-document-list-v2` | api.granola.ai | v1 | api | ✓ |
| `add-document-to-list` | api.granola.ai | v1 | api | ✓ |
| `remove-document-from-list` | api.granola.ai | v1 | (commented) | ✓ present |
| `get-workspaces` | api.granola.ai | v1 | api | ✓ |
| `create-workspace` | api.granola.ai | v1 | api | ✓ |
| `delete-workspace` | api.granola.ai | v1 | api | ✓ |
| `chat-with-documents` | **stream**.api.granola.ai | v1 | stream | ✓ |
| `get-chat-citation` | api.granola.ai | v1 | api | ✓ |
| `get-chat-models` | api.granola.ai | v1 | api | ✓ |
| `get-user-info` | api.granola.ai | v1 | api | ✓ |
| `get-current-subscription` | api.granola.ai | v1 | api | ✓ |
| `get-paywall-status` | **cinnamon**.api.granola.ai | v1 | cinnamon | ✓ |
| `get-feature-flags` | api.granola.ai | v1 | api | ✓ |
| `create-data-export` | api.granola.ai | v1 | api | ✓ |
| `refresh-access-token` | api.granola.ai | v1 | api | ✓ |

**The plan's endpoint→host map is 100% accurate against 7.277.1. No remap needed for the shipped 25 tools.**

Also confirmed present (v2 sibling keys point to `/v2/` URLs):
`create-workspace-v2` → `api.granola.ai/v2/create-workspace`; `get-documents-v2` → `/v2/get-documents`; `get-document-lists-v2` → `/v2/get-document-lists`. The v1 originals (`get-documents`, `get-document-lists`) coexist.

### 1b. The 7 "dropped" endpoints — all STILL PRESENT, one host MOVED

From `schemas-COPueaY7.js`, exact URL lines:

| Endpoint | Host (7.277.1) | Prior-doc host (7.220.0) | Drift |
|---|---|---|---|
| `search-meetings-turbopuffer` | **api**.granola.ai/v1 | (prior recommended it; host ambiguous in prose) | present; on `api`, not maple/stream |
| `get-action-items` | maple.api.granola.ai/v1 | maple | none |
| `update-action-item` | **chia**.api.granola.ai/v1 | **maple** (prior §A: "all on maple except feedback") | **HOST MOVED maple → chia** |
| `get-pre-meeting-briefs` | maple.api.granola.ai/v1 | maple | none |
| `get-follow-up-emails` | berry.api.granola.ai/v1 | berry/cinnamon/maple (prose span) | on `berry` |
| `get-workspace-analytics` | maple.api.granola.ai/v1 | maple | none |
| `download-data-export` | maple.api.granola.ai/v1 | maple | none |

**None of the 7 was removed.** They were dropped from the plan for *runtime* reasons (403/404/400/server-side), not because the schema dropped them — and that classification is unchanged in 7.277.1. The one structural drift is `update-action-item` moving to **chia**; if those are ever revived, the endpoint→host map must reflect chia, not maple.

Request/response **shape** changes for these 7 cannot be verified statically beyond wrapper signatures (all are `(token, input)` shaped). Runtime gating (403/404) requires a live probe, out of scope here.

### 1c. Host topology — 7 hosts unchanged; counts shifted

Whole-bundle host scan confirms **no new API host**. The 7 are identical: `api / maple / cinnamon / berry / stream / pecan / chia .granola.ai`. (Non-API hosts `go.granola.so`, `notes.granola.ai`, `mcp.granola.ai`, `amp.granola.ai`, `meet.granola.ai`, `recipes.granola.ai` are link/web surfaces, same as before.)

Endpoint counts per host (schema map):

| Host | 7.220.0 | 7.277.1 | Δ |
|---|---:|---:|---:|
| api.granola.ai | 294 | 283 | −11 |
| maple | 22 | 22 | 0 |
| cinnamon | 21 | 19 | −2 |
| berry | 13 | 16 | +3 |
| stream | 15 | 14 | −1 |
| pecan | 12 | 13 | +1 |
| chia | 13 | 13 | 0 |
| **total** | **379** | **380** | **+1** |

The net +1 with shifts across hosts means endpoints were **added, removed, and moved between flavored services** between releases — but I cannot enumerate exactly which without the 7.220.0 baseline set. The `api` count dropping 11 while flavored hosts net +12 is consistent with Granola continuing to peel features off the monolith onto flavored services. **Implication for granola-kit:** keep the endpoint→host map per-endpoint (the plan already does); never assume a path on `api.granola.ai`.

### 1d. NEW feature domains visible in 7.277.1 (not characterized in prior doc)

Capability-keyword scan surfaced these (presence in 7.220.0 unconfirmable, but absent from prior prose):

- **Salesforce CRM** — `get-salesforce-integration`, `delete-salesforce-integration`, `salesforce-oauth-callback` (new CRM beyond HubSpot/Notion/Attio).
- **Plain (support tool)** — `create-plain-feedback-thread`, `upsert-plain-customer`, `plain-customer-cards`, `create-plain-attachment-upload-url`.
- **Referral program** — `create-referral-link`, `get-referral-link`, `get-referral-link-public`, `send-referral-email`.
- **Artifact generation** — `generate-artifact` (stream), `generate-summary` (stream), `generate-transcript`.
- **Multipart audio upload** — `initiate-multipart-audio-upload`, `complete-multipart-audio-upload`, `request-audio-upload-url`, `delete-transcription-chunks` (chunked audio ingestion path).
- **Summary A/B experiments** — `get-summary-comparison-experiment`, `get-summary-comparison-assignment`.
- **`auth-soft-check`**, **`store/retrieve-device-session`** — new auth/session endpoints (referenced only in schema tables, not in the header builder; not gating).
- **`validate-account-access`** (chia), **`check-document-access-for-user`** (maple), **`trigger-meeting-classification`** (pecan).

None of these is in the 25-tool plan and none needs to be; they confirm Granola's surface keeps growing. The plan's `SubsetModel` drift-tolerance posture is the right hedge.

---

## 2. IDENTITY HEADERS (7.277.1) — the exact set

The REST header builder is `async function O(n)` in `api-ZMfGOsS8.js` (renderer) with a **byte-identical** twin in `dist-electron/main/index.js` (main process). Verbatim:

```js
async function O(n){
  let r={};
  if(n&&(r.Authorization=`Bearer ${n}`),
     r[`X-Client-Version`]=`${s.PACKAGE_VERSION}${s.IS_WEB?`.web`:``}${s.DEV?`.dev`:``}`,
     s.IS_WEB ? r[`X-Granola-Platform`]=t.web
              : x.platform&&(r[`X-Granola-Platform`]=e(x.platform)?t[x.platform]:t.darwin),
     x.activeWorkspaceId&&(r[`X-Granola-Workspace-Id`]=x.activeWorkspaceId),
     x.getDeviceId){let e=await x.getDeviceId().catch(()=>void 0);e&&(r[`X-Granola-Device-Id`]=e)}
  if(x.getOsVersion){let e=await x.getOsVersion().catch(()=>void 0);e&&(r[`X-Granola-Os-Version`]=e)}
  return w&&(r[`X-Force-Default-Flags`]=`true`),
         T&&(r[`X-Feature-Overrides`]=JSON.stringify(T)),
         r
}
```

**The complete REST identity-header set is 6 headers:**

| Header | Value | Source |
|---|---|---|
| `Authorization` | `Bearer <token>` | only if token present |
| `X-Client-Version` | **`7.277.1`** | `PACKAGE_VERSION` (+`.web`/`.dev` only on those builds; desktop release = bare `7.277.1`) |
| `X-Granola-Platform` | **`macOS`** | table `t` = `{darwin:"macOS", win32:"Windows", web:"web"}` (var `gn` in schemas) keyed by `process.platform` |
| `X-Granola-Workspace-Id` | active workspace UUID | `x.activeWorkspaceId` (from `getPreference('activeWorkspaceId')`) |
| `X-Granola-Device-Id` | `sha256(IOPlatformUUID)` hex | `await x.getDeviceId()` — see §3 |
| `X-Granola-Os-Version` | e.g. `26.4.1` | `sw_vers -productVersion` |

Plus two **debug-only** headers (off in normal use): `X-Force-Default-Flags: true` and `X-Feature-Overrides: <json>`.

**`X-Client-Version` confirmed:** `7.277.1` (the constants module carries both the placeholder `0.0.0` and the injected `7.277.1`; the main process logs `app_version:"7.277.1"`).

**`X-Granola-Platform` confirmed:** `macOS` — empirically matches the capture; **prior enum's `'darwin'` was wrong.**

**`X-Granola-Source`: ABSENT from the REST API surface — confirmed.** It appears exactly **once** in the entire bundle, in a *different* header builder used for the **WebSocket** transport (adjacent to the `ws` library's `nodebuffer/arraybuffer/fragments` constants), with a *different* User-Agent format:

```js
// WebSocket header builder — NOT the REST builder. Irrelevant to granola-kit.
{ "User-Agent":`Granola/${T} (${process.platform}; ${process.arch})`,
  "X-Granola-App-Version":T, "X-Granola-Connection-Id":S, "X-Granola-Source":C }
```

So the plan/empirical "no `X-Granola-Source`" stance is **correct for REST**. Keep omitting it.

**`X-Granola-Time-Zone` is per-call, NOT a global header.** It is added only via `additionalHeaders` on two endpoints:

```js
// workos-auth-complete:
additionalHeaders: t?.timeZone ? {"X-Granola-Time-Zone": t.timeZone} : undefined
// refresh-access-token:
additionalHeaders: T.timeZone ? {"X-Granola-Time-Zone": T.timeZone} : undefined
```

The capture saw it because the auth/refresh flow ran. **Prior enum listing it as a standard gating header is wrong.** granola-kit may safely send it on every call (harmless), but it is not required for normal reads/writes.

**NEW gating header check:** None. The request core `M` merges exactly `{...O(token), ...additionalHeaders, ...(body?{'Content-Type':'application/json'}:{})}`. No new required `X-*` appeared. Other `X-*` literals in the bundle (`X-Granola-Client: companion-cli`, `X-Client-Info`, `X-Client-Sample-Rate`, `X-Client-Library`) belong to the companion-CLI feature and Sentry/telemetry SDKs, not the user-data REST API.

**User-Agent caveat:** the full Electron UA from the capture (`Mozilla/5.0 … Granola/7.220.0 Chrome/146… Electron/41.2.1 …`) is **not a hardcoded string** — Electron generates it at runtime (Chrome UA + `Granola/<version>` appended). It drifts with each release's bundled Chromium/Electron. granola-kit's UA, copied from the 7.220.0 capture, will be stale (`7.220.0`, `Chrome/146`, `Electron/41.2.1`) but the API does not appear to validate it. Optionally bump `Granola/<version>` to match `X-Client-Version`; the Chrome/Electron parts are cosmetic.

---

## 3. DEVICE-ID DERIVATION — solved and empirically proven

`X-Granola-Device-Id` = `await x.getDeviceId()`, injected in the main process as:

```js
// dist-electron/main/index.js — the api context construction:
getDeviceId: async () => hashedDeviceId ?? null,
getOsVersion: async () => osVersion ?? null,
```

So the header value is **`hashedDeviceId`**, computed once at startup:

```js
function hashDeviceId(S){ if(S) return (0,node_crypto.createHash)(`sha256`).update(S).digest(`hex`) }
var hashedDeviceId = hashDeviceId(deviceId);
```

And `deviceId` = `getDeviceId()` (the *raw* extractor — note this is a different fn than the injected async closure):

```js
function getDeviceId(){
  try{
    switch((0,os$1.platform)()){
      case `darwin`:{
        let S=(0,child_process.execSync)(`ioreg -d2 -c IOPlatformExpertDevice`).toString().split(`\n`);
        for(let C of S) if(C.includes(`IOPlatformUUID`)){
          let S=C.split(`"`);
          for(let C=0;C<S.length;C++) if(S[C]===`IOPlatformUUID`) return S[C+2]   // value two slots after the key
        }
        break
      }
      case `win32`:{
        let S=/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i,
            C=(0,child_process.execSync)(`reg.exe query HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography /v MachineGuid /reg:64`,{encoding:`utf8`});
        return S.exec(C)?.[0] ?? void 0
      }
    }
  }catch(S){ logToBackend(`error`,`get-device-id-failed`,{error:serializeError(S)}) }
}
```

### The exact macOS algorithm

1. `execSync('ioreg -d2 -c IOPlatformExpertDevice')`, split on `\n`.
2. Find the line containing `IOPlatformUUID` (looks like `    "IOPlatformUUID" = "7F7E1B9E-…"`).
3. Split that line on `"`. Array ≈ `["    ", "IOPlatformUUID", " = ", "<UUID>", ""]`. Find index where element `=== "IOPlatformUUID"`, return `parts[i+2]` — the UUID, **exactly as ioreg prints it: uppercase, hyphenated, no quotes.**
4. `X-Granola-Device-Id = sha256(<that UUID string>).hexdigest()`.

Windows: regex-extract `MachineGuid` from the registry, then the same sha256.

### Empirical proof (this machine)

```
ioreg IOPlatformUUID         = 7F7E1B9E-3912-5C2E-B80A-0B2F6CACDA91
sha256(that, utf-8)          = 71a05389cc0df4c31714f759d2068ff639a2bb1a78ff7b06d04f15b48fa72f8e
captured X-Granola-Device-Id = 71a05389…   ← MATCH (first 8 chars and full hash)
```

**Verdict:** `empirical-surface.md` and the plan's "NOT `sha256(IOPlatformUUID)`; derivation TBD; read from a one-time captured value" is **flatly wrong**. The derivation is `sha256(IOPlatformUUID)`. Reference Python (drop-in for `identity.py`):

```python
import subprocess, hashlib
def granola_device_id() -> str:
    out = subprocess.check_output(['ioreg','-d2','-c','IOPlatformExpertDevice']).decode()
    for line in out.split('\n'):
        if 'IOPlatformUUID' in line:
            parts = line.split('"')
            for i, p in enumerate(parts):
                if p == 'IOPlatformUUID':
                    return hashlib.sha256(parts[i+2].encode()).hexdigest()
    raise RuntimeError('IOPlatformUUID not found')
```

**No persisted file backs this** — it is derived fresh each launch from `ioreg`; there is no device-id stored on disk. (`store/retrieve-device-session` endpoints are unrelated — they're for room-device pairing sessions, not this client identity.) **Recommendation: delete the planned `identity.json` device-id capture/`granola-kit identity capture` machinery; compute it.** The same goes for `X-Granola-Os-Version` (`sw_vers -productVersion`) and `X-Granola-Platform` (constant `macOS`) — all three are computable, none needs capture. Only `X-Granola-Workspace-Id` (the active workspace UUID) is account state, and that comes from the token/user-info, not capture.

---

## 4. `search-meetings-turbopuffer` — CLIENT-callable (plan assumption is wrong)

### It has a typed wrapper

`api-ZMfGOsS8.js`:

```js
function Er(e,t){ return M(e,`search-meetings-turbopuffer`,{input:t}) }   // (token, input)
// exported: at→Er (public "Er"), and Er→mr (public "mr") — the turbopuffer fn is public export `mr`
```

### It has a live client caller in the production renderer

Exactly one file imports the public `mr`: `dist-app/assets/useTodayAndTomorrowEvents-DY5E19Qe.js` (`import {…, mr as o, …}`). That file defines and **exports** the search function (public `C` → impl `w`), and it is imported by **`primary-C2f84e4V.js`** (the 5.2 MB main renderer bundle — production, not dev-only) and by `DevToolsPanel`.

The search builder (`w`), verbatim:

```js
async function w(e,t){
  let n=u.getState().activeWorkspaceId,
      r={ limit:t.limit??30, includeShared:t.includeShared??!1,
          ...t.fromDate?{fromDate:t.fromDate}:{}, ...t.toDate?{toDate:t.toDate}:{},
          ...n?{activeWorkspaceId:n}:{} },
      i = t.searchType===`people_and_companies`
            ? {...r, searchType:`people_and_companies`, people:t.people??[], companies:t.companies??[], strictParticipantMatch:t.strictParticipantMatch??!1}
            : {...r, searchType:`keywords`, keywords:t.keywords??[]},
      a;
  try{ a = await o(e,i) }catch(e){ throw e instanceof Error?e:Error(`Search failed`) }
  if(a.error) throw Error(a.error);
  let s={ results:a.results, searchType:t.searchType, query:y(t) };
  return /* store */ s
}
```

And the invocation in `primary` (a deep-link command handler, gated on a logged-in token):

```js
case `turbopuffer-search`:
  if(!n||!o) break;
  if(!n.access_token){ X.error(`Please log in to run Turbopuffer search`); break }
  Oue(n.access_token, {
    searchType:t.data.searchType, people:…, companies:…, keywords:…,
    strictParticipantMatch:t.data.strictParticipantMatch, limit:…
  }).then(()=> r(`/internal/turbopuffer`,{replace:!0})) …
```

### Recovered request/response shape

**POST `search-meetings-turbopuffer`** (host `api.granola.ai/v1`), body:

```jsonc
{
  "searchType": "keywords",          // OR "people_and_companies"
  "keywords": ["..."],               // when searchType=keywords
  "people": ["..."],                 // when searchType=people_and_companies
  "companies": ["..."],              // "
  "strictParticipantMatch": false,   // "
  "limit": 30,
  "includeShared": false,
  "fromDate": "...",                 // optional
  "toDate": "...",                   // optional
  "activeWorkspaceId": "<uuid>"      // injected from active workspace
}
```

Response: `{ "results": [...], "error"?: "..." }`.

### Verdict

The plan states (Reconnaissance finding #1): *"`search-meetings-turbopuffer` is server-side only — never called by the client … Drop `search_meetings`."* **The static evidence contradicts the "cannot call directly" half.** There is a public wrapper *and* a production-renderer caller that POSTs the endpoint with the user's `access_token`, with a clear two-mode input contract. It is **structurally client-callable**.

**Caveat (be unflinching):** `empirical-surface.md` reports it was *never observed in the mitm capture* during normal use, and granola-kit cannot confirm a live 200 without a probe. There are two distinct facts here: (a) **it is server-internal to the chat ReAct loop** — the chat agent's tools `searchMeetingsByKeywords` / `searchMeetingsByPeopleAndCompanies` (in the `Nbt` tool→category map in `primary`) name the *same capability*, executed server-side by `chat-with-documents`; (b) **it is ALSO directly client-callable** via the `/internal/turbopuffer` deep-link path. The deep-link caller suggests it may be a dev/internal surface (hence not seen in ordinary capture), but the auth model (user bearer token, `activeWorkspaceId`) is exactly what granola-kit has.

**Recommendation:** reopen `search_meetings` as a **v1.1 candidate pending one live probe**. Before shipping, call it once with `{"searchType":"keywords","keywords":["test"],"limit":3}` + the 6 headers and confirm a 200 (not 403/404). If it 200s, it is strictly cheaper than `chat_with_meetings` for pure retrieval (no LLM round-trip) and gives structured hits. If it gates, keep it deferred. Either way, **the plan's stated reason for dropping it ("cannot be called directly") is not supported by the binary.**

---

## 5. Other adversarially-relevant findings for the 25-tool plan

### 5a. `supabase.json` encryption — HIGH-PRIORITY latent break

The token file is now wrapped by an encrypted storage layer, gated on a flag that **defaults TRUE**:

```js
// dist-electron/main/index.js
encryptedSupabaseStorageFlag = { id:`encrypted_supabase_storage`, defaultValue:!0 };   // TRUE
supabaseStorage = isFeatureEnabled(`SUPABASE_STORAGE_PROCESS`)
  ? createStorage({ file:`supabase.json`, encrypted:true,
                    getEncryptionEnabled: () => !!getFeatureFlag(encryptedSupabaseStorageFlag) })
  : createStorageDummy({ cognito_tok… });
```

Encryption backend is **Electron `safeStorage`** (`safeStorage.encryptString` / `safeStorage.decryptStringAsync`) → macOS **Keychain** ("Granola Safe Storage" key). There is also `encrypted_preferences_storage` (defaultValue TRUE) for `activeWorkspaceId` and prefs.

**Current on-disk reality (verified):** `~/Library/Application Support/Granola/supabase.json` is **still plaintext JSON** (size 2879 B, mtime **5/12**, i.e. *before* the 5/27 update), with top-level keys `{workos_tokens, session_id, user_info}` where `workos_tokens` is a JSON string holding the WorkOS JWT. So **granola-kit's "read supabase.json + JWT decode" works today.** But the file has not been rewritten since the update; the next time Granola re-auths or rotates, the flag-gated path may write a `safeStorage`-encrypted blob, and the plaintext read will fail.

**Recommendations:**
- `auth.py` must **detect encryption**: try `json.loads`; on failure, treat the file as a `safeStorage` blob.
- Decrypting `safeStorage` from Python on macOS means reading the Keychain item and applying AES — non-trivial. Cheaper fallback: capture the live bearer token (already what granola-kit ultimately uses) and cache it, refreshing via `refresh-access-token`.
- Add a **regression-style guard**: if `supabase.json` is unparseable, emit an actionable error ("Granola token file is encrypted (safeStorage). Re-capture token or implement keychain decrypt"). Do not silently fail.
- This belongs in `docs/possible-features.md` / `CLAUDE.md` author-notes as the top drift risk.

### 5b. Auth/refresh flow — unchanged in essentials

```js
function refreshAccessToken(S,C,T){
  return makeAPICall(S, `refresh-access-token`,
    { input:{refresh_token:C}, retries:3, signal:T.signal,
      additionalHeaders: T.timeZone ? {"X-Granola-Time-Zone":T.timeZone} : undefined })
}
```

Token schema (Zod) in main: `{ access_token, expires_in, refresh_token, token_type, obtained_at?, session_id?, sign_in_method? }`. URL: `https://api.granola.ai/v1/refresh-access-token`. **granola-kit's existing refresh helper remains valid.** `auth-soft-check` exists but is only in schema tables, not the refresh path.

### 5c. Core request core `M` — signature unchanged, behavior worth noting

```js
async function M(e,t,{input:n,queue:r,retries:i=0,signal:a,method:o,additionalHeaders:s}={}){
  // headers: {...await O(e), ...s, ...(n?{"Content-Type":"application/json"}:{})}
  // method: o ?? "POST";  url: x.getAPIEndpoint(t);  body: n?JSON.stringify(n):null
  // reads response header x-server-time for clock-skew sampling
  // retry backoff: [0,1000,2000,4000,8000,16000,32000,64000,128000,256000] ms, n<=retries
}
```

Same shape as the prior `N`. No new required params. Errors throw a typed error carrying `responseStatus`, `responseText`, and `apigw-requestid` — useful for granola-kit's actionable-error boundary (capture `apigw-requestid` from response headers when surfacing failures).

### 5d. `chat-with-documents` streaming — confirmed for `chat_with_meetings`

The streaming wrapper picks `chat-with-documents-web` vs `chat-with-documents` by `IS_WEB`, POSTs `body:JSON.stringify(t)` with the 6 identity headers + `Content-Type`, `timeoutMS:30000`, `retryCount:2`, newline-delimited chunks consumed via `body.getReader()` with per-chunk `successCallback`. The request body uses a top-level **`chat_context`** field (seen in the error path: `chat_context:t.chat_context`) — note the empirical capture reported `messageContext`/`currentViewContext`; the binary's error logging shows `chat_context`. **Both may coexist**; granola-kit's `ChatService` should send what the capture observed (it 200'd) but be aware the field naming has variants. Chunk schema is heavy on `toolCall`/`tool_call`/`outputs` (the server-side ReAct loop emits tool-call traces) — matches the plan's "collect stream → final answer + tool trace".

### 5e. The in-app agent tool map (context for `chat_with_meetings`)

`primary-C2f84e4V.js` carries `Nbt`, the agent tool→UI-category map — the actual tool names the `chat-with-documents` server loop exposes:

```
web_search, web_fetch, getUpcomingCalendarEvents, listMeetings, exploreMeetings, readMeetings,
readTranscript, readFile, listFolderFiles, generateImage, generateExcalidrawDiagram,
get_action_items, render_action_items, editMeetingNotes, rewriteMeetingNotes, fetchMeetings,
getMeetings, queryMeetingContext, searchMeetingsByPeople, searchMeetingsByPeopleAndCompanies,
searchMeetingsByKeywords, getCurrentTime, searchHelpArticles, code_execution
```

This is a labeling map only (renders tool-call chips), not an executor — but it documents the full capability the chat agent wields. Two takeaways: (1) `chat_with_meetings` transitively reaches turbopuffer search, action items, transcripts, calendar, web search, and even `code_execution` — it is broad; (2) `get_action_items` and `render_action_items` are agent-internal even though the direct `get-action-items` REST endpoint is 403-gated on this account, i.e. the chat loop can surface action items the direct endpoint won't.

### 5f. Local SQLite still encrypted (unchanged)

`package.json` still bundles `better-sqlite3-multiple-ciphers@^12.9.0`. The plan's HTTP-only stance remains correct.

---

## Net recommendations to the plan

1. **Device-ID (correct a wrong finding):** drop `identity.json` device-id capture and `granola-kit identity capture`; **compute** `sha256(IOPlatformUUID)` at runtime (§3). Same for `X-Granola-Os-Version` (`sw_vers -productVersion`) and `X-Granola-Platform` (`macOS`).
2. **`X-Granola-Time-Zone` (correct a wrong finding):** it is *not* one of "6 gating headers"; it's per-call on auth/refresh only. The 6 REST headers are `Authorization, X-Client-Version, X-Granola-Platform, X-Granola-Workspace-Id, X-Granola-Device-Id, X-Granola-Os-Version`. Sending Time-Zone everywhere is harmless but unnecessary.
3. **`search_meetings` (correct a wrong assumption):** the binary proves it is client-callable with the user token; the plan dropped it for the wrong reason. Reopen as v1.1 pending a single live 200 probe.
4. **NEW top drift risk:** `supabase.json` encryption (`safeStorage`/Keychain, flag defaults TRUE). Make `auth.py` detect-and-degrade; document as the leading break vector (§5a).
5. **Host map:** plan's 22-entry map is exactly right for 7.277.1 — no changes for shipped tools. If action items are ever revived, `update-action-item` is on **chia**, not maple.
6. `X-Client-Version` → bump to `7.277.1`; consider auto-deriving from the installed app's `Info.plist` rather than hardcoding, since it drifts every release.

## Reproducibility

- Extraction: `npx @electron/asar extract /Applications/Granola.app/Contents/Resources/app.asar /tmp/granola-asar-7277`
- Endpoint map TSV: `/tmp/ep-7277.tsv` (endpoint · host · version · url-tail; 380 rows)
- Device-id proof: re-run the Python in §3 — output must start `71a05389…` on this machine.
- All header/wrapper claims: grep `dist-app/assets/api-ZMfGOsS8.js` and `dist-electron/main/index.js`.
