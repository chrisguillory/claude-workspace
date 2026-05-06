# grok-kit

MCP server + CLI for mirroring grok.com conversations into the workspace. Pulls
the user's chat history (including Tesla in-car Grok sessions synced via X SSO)
through grok.com's reverse-engineered REST API.

## Architecture

Three layers, three responsibility regions:

```
api-spec/openapi.yaml    Hand-authored from HAR observations. Source of truth
                         for the API surface. LLM-judgment layer.
       │
       ▼
grok-kit-sdk/            Speakeasy-generated Python SDK. Pydantic v2 models,
                         sync+async clients, hard-fail on 4XX/5XX. Mechanical
                         codegen layer; regenerated when the spec changes.
       │
       ▼
grok_kit/                Hand-written consumer. Auth bootstrap, transport
                         pluggability, business logic, MCP tools, CLI commands.
                         Only references SDK fields it actually uses.
```

**Why this layering**: when xAI changes their API, the diff is contained to the
spec layer (LLM updates the YAML) and the regenerated SDK absorbs it
mechanically. The consumer is insulated — it only breaks if a field IT references
changes shape, not if grok.com adds new fields elsewhere.

## API spec

`api-spec/openapi.yaml` is hand-authored OpenAPI 3.1 describing 5 endpoints
empirically observed via HAR capture against an authenticated grok.com session:

| Endpoint | Purpose |
|----------|---------|
| `GET /rest/app-chat/conversations` | Paginated list of conversation summaries |
| `GET /rest/app-chat/conversations_v2/{id}` | Single conversation metadata |
| `GET /rest/app-chat/conversations/{id}/response-node` | Message tree (parent-pointer linked) |
| `POST /rest/app-chat/conversations/{id}/load-responses` | Full message bodies for given responseIds |
| `GET /rest/app-chat/share_links?conversationId={id}` | Share-link list |

The spec is not authoritative; it reflects what we observed. xAI may evolve
the surface silently. When that happens: re-capture HAR → update spec → regen.

## Auth

Cookie-based via X SSO. Five load-bearing cookies (`sso`, `sso-rw`, `x-userid`,
`cf_clearance`, `__cf_bm`). The spec models them as a single `Cookie:` header
because Speakeasy's Python runtime doesn't support `apiKey in: cookie` security
schemes (verified gap, file `grok-kit-sdk/src/grok_kit_sdk/utils/security.py`
line 151 — un-interpolated stub). Wire-correct: cookies travel as one Cookie
header anyway. The auth layer in `grok_kit/auth.py` formats the cookie string
from individual cookies extracted via the selenium-browser MCP's
`save_profile_state` path.

Cookie expiry: 12-72h for `cf_bm`/`cf_clearance`. Refresh path: detect 401/403
from the SDK → re-bootstrap via selenium-browser.

## Regenerating the SDK

```bash
cd mcp/grok-kit/grok-kit-sdk
speakeasy run --skip-versioning --skip-upload-spec
```

`--skip-upload-spec` keeps the spec out of Speakeasy's hosted registry (we
describe a third party's API, not ours). Without it, the spec uploads to a
workspace-private location at `registry.speakeasyapi.dev/<workspace>/...`.

If `speakeasy auth login` is needed first, you'll be prompted.

## Speakeasy config rationale

Mirrors the monorepo SDK convention exactly (`gen.yaml`, `workflow.yaml`):

- `enumFormat: union` → `Literal` types, not Enum classes
- `clientServerStatusCodesAsErrors: true` → non-2xx → typed errors (hard-fail
  per workspace philosophy)
- `asyncMode: both` → sync + async surface
- `flattenGlobalSecurity: true` → single-field security hoisted into the SDK
  constructor (`GrokKit(cookie_header=...)` ergonomics)

## Layer interaction

```python
# Consumer-side usage:
from grok_kit_sdk import GrokKit  # Speakeasy-generated typed client
from grok_kit.auth import format_cookie_header  # consumer-side auth helper

cookie_header = format_cookie_header(load_cookies_from_browser())
with GrokKit(cookie_header=cookie_header) as sdk:
    convs = sdk.conversations.list_conversations(page_size=60)  # typed response
    for c in convs.conversations:
        print(c.title, c.modify_time)  # typed fields, IDE completion
```

The consumer's `service.py` wraps SDK calls in workspace-shaped operations
(syncing, pagination, retry-on-401, etc.). MCP tools and CLI commands consume
`service.py`, not the SDK directly.

## Status

Scaffolded. Auth bootstrap, transport, service, MCP tools, and CLI commands are
stubs awaiting implementation. SDK is fully working end-to-end against live
grok.com (verified via spike).
