# Session Persistence Deep Dive

Complete reference for browser session export/import, including manual workflows, automation research, and real-world testing.

## Overview

### The Problem

Chrome profile loading (`profile="Default"`) doesn't work reliably for session reuse:
- Chrome locks profiles when running (can't share with Selenium)
- Even when Chrome is closed, Selenium doesn't read profiles consistently
- All-or-nothing approach (entire profile, not domain-scoped)
- No portability between machines or browsers

### The Solution

Export browser storage state (cookies + localStorage) to a portable JSON format, then import when starting new Selenium sessions.

**Two workflows:**

| Workflow | When to Use |
|----------|-------------|
| **Selenium-native** | Login via Selenium → `save_storage_state()` → later `storage_state_file` |
| **Chrome-to-Selenium** | Login in Chrome proper → manual export → build JSON → `storage_state_file` |

The Chrome-to-Selenium workflow is valuable when:
- Sites with bot detection block Selenium login but accept Chrome-established sessions
- 2FA/MFA is easier to complete in your regular browser
- You want to reuse existing authenticated sessions from daily browsing

## Playwright storageState Format

We use Playwright's storageState JSON format for cross-tool compatibility. Files can be used with Playwright, Puppeteer, or edited manually.

### Schema

```json
{
  "cookies": [
    {
      "name": "sessionId",
      "value": "abc123",
      "domain": ".example.com",
      "path": "/",
      "expires": -1,
      "httpOnly": true,
      "secure": true,
      "sameSite": "Lax"
    }
  ],
  "origins": [
    {
      "origin": "https://www.example.com",
      "localStorage": [
        {"name": "authToken", "value": "eyJ..."}
      ]
    }
  ]
}
```

### Cookie Fields

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | Cookie name |
| `value` | string | Cookie value |
| `domain` | string | Include leading `.` for domain-wide cookies |
| `path` | string | Usually `/` |
| `expires` | number | `-1` for session cookies, Unix epoch seconds for persistent |
| `httpOnly` | boolean | If true, not accessible via JavaScript |
| `secure` | boolean | If true, only sent over HTTPS |
| `sameSite` | string | `"Strict"`, `"Lax"`, or `"None"` (must capitalize first letter) |

### Origin Fields

| Field | Type | Notes |
|-------|------|-------|
| `origin` | string | Full origin: `https://www.example.com` (no trailing slash) |
| `localStorage` | array | `[{name, value}, ...]` pairs |

### Import Behavior

When `storage_state_file` is provided to `navigate()`:

1. **Cookies set via CDP BEFORE navigation** - Sent with the initial HTTP request
2. **localStorage restored AFTER navigation** - Requires origin context (page must load first)

This sequencing matters: cookies authenticate the request, localStorage may be needed for client-side auth state.

## Manual Chrome Export

When Selenium can't complete login (bot detection, complex MFA), export from your regular Chrome.

### Step 1: Export localStorage

Open DevTools Console on the authenticated page:

```javascript
copy(JSON.stringify(
  Object.entries(localStorage).map(([k, v]) => ({name: k, value: v})),
  null, 2
))
```

Paste into a temporary file (e.g., `localStorage.json`).

### Step 2: Export Cookies

1. DevTools → Application → Cookies → select the site
2. Select all cookies: `Cmd+A` (Mac) or `Ctrl+A` (Windows)
3. Copy: `Cmd+C` / `Ctrl+C`
4. Paste into a text file

This produces a **tab-separated table** with columns:
```
Name    Value    Domain    Path    Expires/Max-Age    Size    HttpOnly    Secure    SameSite    ...
```

### Step 3: Parse Cookie Table

The tab-separated format needs conversion. Key transformations:

| Chrome Column | storageState Field | Transformation |
|--------------|-------------------|----------------|
| `Expires / Max-Age` | `expires` | `"Session"` → `-1`, date → epoch seconds |
| `HttpOnly` | `httpOnly` | `"✓"` → `true`, empty → `false` |
| `Secure` | `secure` | `"✓"` → `true`, empty → `false` |
| `SameSite` | `sameSite` | `"strict"` → `"Strict"`, etc. (capitalize) |

**Example Python parser:**

```python
import json
from datetime import datetime

def parse_cookie_line(line: str) -> dict:
    """Parse a single tab-separated cookie line from Chrome."""
    parts = line.split('\t')
    if len(parts) < 8:
        return None

    name, value, domain, path, expires_str, size, http_only, secure = parts[:8]
    same_site = parts[8] if len(parts) > 8 else "Lax"

    # Parse expires
    if expires_str == "Session":
        expires = -1
    else:
        try:
            # Chrome format: "2025-12-27T02:24:40.000Z" or similar
            dt = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
            expires = dt.timestamp()
        except:
            expires = -1

    # Normalize sameSite
    same_site_map = {"strict": "Strict", "lax": "Lax", "none": "None"}
    same_site = same_site_map.get(same_site.lower(), "Lax")

    return {
        "name": name,
        "value": value,
        "domain": domain,
        "path": path,
        "expires": expires,
        "httpOnly": http_only == "✓",
        "secure": secure == "✓",
        "sameSite": same_site,
    }
```

### Step 4: Build Storage State JSON

Combine cookies and localStorage:

```python
storage_state = {
    "cookies": parsed_cookies,
    "origins": [
        {
            "origin": "https://www.example.com",
            "localStorage": localStorage_items
        }
    ]
}

with open("auth.json", "w") as f:
    json.dump(storage_state, f, indent=2)
```

### Step 5: Test Import

```python
navigate(
    "https://www.example.com/account",
    fresh_browser=True,
    storage_state_file="auth.json"
)
```

## Real-World Testing: Marriott.com

Extensive testing with Marriott Bonvoy revealed patterns applicable to other complex authentication systems.

### What We Tried

| Attempt | Result | Issue |
|---------|--------|-------|
| Direct Selenium login | ❌ Blocked | Akamai bot detection ("Access Denied") |
| Chrome profile loading | ❌ Failed | Profile lock, inconsistent state |
| Partial cookie export (10 cookies) | ❌ Failed | Missing critical Akamai cookies |
| Full cookie export (83 cookies) | ✅ Success | All cookies including `bm_*` required |

### Critical Findings

**1. Akamai Bot Management Cookies**

Marriott uses Akamai for bot detection. These cookies are critical:

| Cookie | Purpose |
|--------|---------|
| `bm_so` | Session origin fingerprint |
| `bm_ss` | Session state |
| `bm_sz` | Session size/checksum |
| `_abck` | Akamai bot challenge cookie |
| `ak_bmsc` | Bot management session cookie |

Without these, even with valid auth tokens, requests are blocked.

**2. Authentication Token Architecture**

Marriott uses a hybrid approach:

| Storage | Token | Purpose |
|---------|-------|---------|
| Cookie | `authStateToken` (JWT) | API authorization |
| Cookie | `UserIdToken` (JWT) | User identity |
| Cookie | `sessionID` | Session tracking |
| localStorage | `mi-session-store` | Client-side auth state, UI personalization |

**3. Device Fingerprinting**

Three cookies establish device identity:

```
deviceId: 42e7ca76-56e2-4ed7-91e0-b86a954ba1b2
devicePrivateKey: ME4CAQAwEAYHKoZIzj0CAQYFK4EEACIENzA1AgEB...
devicePublicKey: MHYwEAYHKoZIzj0CAQYFK4EEACIDYgAEJGJo7S4...
```

These are bound to the browser on first login. New browsers get new keys on login.

**4. Token Lifetime**

Marriott tokens are **extremely short-lived** (~30 minutes):

```json
{
  "expiresIn": "2025-12-27T02:24:40.281874334",
  "userIdTokenExpiresIn": "1766802461"
}
```

**Implication:** Export and test immediately after login. Sessions cannot be saved for later use without token refresh.

**5. localStorage Overwrite Issue**

Marriott's JavaScript creates a new session on page load:

```javascript
// Our imported sessionToken
"C6CF3E8E-0B1C-5674-8EA8-D592C384B8FA"

// After page JS runs - NEW token!
"736486B8-1721-54E5-9E18-2B8472A30B32"
```

**Why it still worked:** The cookies (especially `authStateToken` JWT) are the source of truth. The localStorage `mi-session-store` is for client-side UI state. When cookies are valid, the page recognizes the session regardless of localStorage.

### Success Pattern

What ultimately worked:

1. Login in Chrome proper with "Remember this device" option
2. Export ALL cookies (83 total), not just obvious auth ones
3. Export localStorage for UI state
4. Import immediately (within token lifetime)
5. Navigate to authenticated page
6. Verify via ARIA snapshot (shows "Hello, Christopher", points, status)

## Automation Research

Manual export is tedious. Here are researched automation approaches.

### Approach 1: CDP Remote Debugging

**How it works:**
1. Start Chrome with `--remote-debugging-port=9222`
2. Connect via WebSocket to CDP endpoint
3. Use `Network.getCookies` for all cookies (including HttpOnly)
4. Use `Runtime.evaluate` for localStorage
5. Build and save storageState JSON

**Implementation sketch:**

```python
import websocket
import json
import requests

# Get WebSocket URL from Chrome
resp = requests.get("http://localhost:9222/json")
ws_url = resp.json()[0]["webSocketDebuggerUrl"]

# Connect
ws = websocket.create_connection(ws_url)

# Get cookies
ws.send(json.dumps({"id": 1, "method": "Network.getCookies"}))
result = json.loads(ws.recv())
cookies = result["result"]["cookies"]

# Get localStorage (in page context)
ws.send(json.dumps({
    "id": 2,
    "method": "Runtime.evaluate",
    "params": {"expression": "JSON.stringify(localStorage)"}
}))
result = json.loads(ws.recv())
local_storage = json.loads(result["result"]["result"]["value"])
```

**Pros:**
- No extension needed
- Full access to HttpOnly cookies
- Python-native, could integrate into MCP server
- Works with existing Chrome (just needs restart with flag)

**Cons:**
- Requires restarting Chrome with `--remote-debugging-port=9222`
- User must close all Chrome windows first
- Security implications of exposed debugging port

**Complexity:** Medium (WebSocket handling, CDP protocol)

**User Experience:** Poor (requires Chrome restart)

### Approach 2: Chrome Extension

**How it works:**

Manifest V3 extension with:
- `chrome.cookies.getAll({})` - all cookies including HttpOnly
- Content script to read `localStorage`
- Export button that saves storageState JSON

**Manifest:**

```json
{
  "manifest_version": 3,
  "name": "Session Exporter",
  "version": "1.0",
  "permissions": ["cookies", "storage", "activeTab"],
  "host_permissions": ["<all_urls>"],
  "action": {
    "default_popup": "popup.html"
  }
}
```

**Popup script:**

```javascript
async function exportSession() {
  // Get current tab's URL for origin
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  const url = new URL(tab.url);

  // Get all cookies for this domain
  const cookies = await chrome.cookies.getAll({domain: url.hostname});

  // Convert to storageState format
  const storageCookies = cookies.map(c => ({
    name: c.name,
    value: c.value,
    domain: c.domain,
    path: c.path,
    expires: c.expirationDate || -1,
    httpOnly: c.httpOnly,
    secure: c.secure,
    sameSite: c.sameSite === "strict" ? "Strict" :
              c.sameSite === "lax" ? "Lax" : "None"
  }));

  // Get localStorage via content script
  const [{result: localStorage}] = await chrome.scripting.executeScript({
    target: {tabId: tab.id},
    func: () => Object.entries(localStorage).map(([k,v]) => ({name: k, value: v}))
  });

  const storageState = {
    cookies: storageCookies,
    origins: [{origin: url.origin, localStorage}]
  };

  // Download as file
  const blob = new Blob([JSON.stringify(storageState, null, 2)], {type: "application/json"});
  const downloadUrl = URL.createObjectURL(blob);
  chrome.downloads.download({url: downloadUrl, filename: "session.json"});
}
```

**Pros:**
- Works with normal Chrome (no restart)
- One-click export
- Good UX (popup with button)
- Could add domain filtering, session naming

**Cons:**
- Must build and install extension
- Extension permission warnings may concern users
- Manifest V3 service worker quirks

**Complexity:** Medium (extension packaging, Chrome Web Store optional)

**User Experience:** Good (once installed)

### Approach 3: MCP Server Integration

**How it works:**

New tool `import_from_chrome()` that:
1. Checks if Chrome is running with `--remote-debugging-port`
2. If yes, connects via CDP and exports
3. If no, provides instructions for restarting Chrome

**Tool definition:**

```python
@mcp.tool()
async def import_from_chrome(
    domain: str | None = None,
    output_file: str = "session.json",
) -> ImportResult:
    """
    Import session from running Chrome (requires --remote-debugging-port=9222).

    If Chrome isn't running with debugging enabled, returns instructions.
    """
    try:
        # Try to connect
        resp = requests.get("http://localhost:9222/json", timeout=1)
        targets = resp.json()
    except:
        return ImportResult(
            success=False,
            message="Chrome not running with debugging. Restart with:\n"
                    "open -a 'Google Chrome' --args --remote-debugging-port=9222"
        )

    # Find page matching domain
    target = next((t for t in targets if domain in t.get("url", "")), targets[0])

    # Connect and export
    ws = websocket.create_connection(target["webSocketDebuggerUrl"])
    # ... CDP calls ...

    return ImportResult(
        success=True,
        path=output_file,
        cookies_count=len(cookies),
        origins_count=1
    )
```

**Pros:**
- Integrated workflow (no context switching)
- Could cache sessions by domain
- Natural tool discovery

**Cons:**
- Still requires Chrome restart with flag
- Adds complexity to MCP server
- Edge cases (multiple Chrome windows, profiles)

**Complexity:** Medium-High (robust CDP handling, error cases)

**User Experience:** Medium (seamless when Chrome is ready, friction otherwise)

### Approach 4: Bookmarklet

**How it works:**

JavaScript bookmarklet that exports current page's session:

```javascript
javascript:(function(){
  const cookies = document.cookie.split(';').map(c => {
    const [name, value] = c.trim().split('=');
    return {name, value, domain: location.hostname, path: '/',
            expires: -1, httpOnly: false, secure: location.protocol === 'https:',
            sameSite: 'Lax'};
  });
  const ls = Object.entries(localStorage).map(([k,v]) => ({name: k, value: v}));
  const state = {cookies, origins: [{origin: location.origin, localStorage: ls}]};
  const blob = new Blob([JSON.stringify(state, null, 2)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = location.hostname + '_session.json'; a.click();
})();
```

**Pros:**
- Zero installation
- Works anywhere (drag to bookmarks bar)
- Instant export

**Cons:**
- **Cannot access HttpOnly cookies** (JavaScript limitation)
- Only gets `document.cookie` visible cookies
- May miss critical auth cookies

**Complexity:** Low

**User Experience:** Good (but incomplete data)

### Recommendation Matrix

| Approach | HttpOnly Cookies | UX | Setup Effort | Completeness |
|----------|-----------------|-----|--------------|--------------|
| CDP Remote Debugging | ✅ | Poor | Medium | Full |
| Chrome Extension | ✅ | Good | Medium | Full |
| MCP Integration | ✅ | Medium | High | Full |
| Bookmarklet | ❌ | Good | None | Partial |

**For maximum reliability:** Chrome Extension or CDP approach (both get HttpOnly cookies).

**For quick one-off exports:** Manual process documented in README (proven to work).

## Site-Specific Patterns

Different sites use different authentication mechanisms. Here's what we've observed.

### Simple Cookie Auth

Sites that primarily use cookies for auth:
- Set `authToken` or `sessionId` cookie on login
- Cookie is HttpOnly, Secure
- No complex client-side state

**Export complexity:** Low - just cookies needed
**Example sites:** Many traditional web apps

### JWT + localStorage

Modern SPAs using JWTs:
- JWT stored in localStorage (not cookies)
- Sent in `Authorization: Bearer` header
- May have short expiration with refresh tokens

**Export complexity:** Medium - need localStorage, may need to handle token refresh
**Example sites:** React/Vue/Angular SPAs

### Hybrid (Marriott Pattern)

Complex enterprise sites:
- Multiple cookies (auth, session, device, bot detection)
- localStorage for UI state
- Short token lifetimes
- Device fingerprinting

**Export complexity:** High - need ALL cookies, immediate use
**Example sites:** Airlines, hotels, banks

### OAuth/SSO

Sites using third-party authentication:
- Auth state spans multiple domains
- Tokens issued by identity provider
- May use cross-origin cookies

**Export complexity:** Very High - multi-domain cookie capture needed
**Current support:** Limited (single origin localStorage)

## Future Enhancements

### Near-Term

| Enhancement            | Description                                       | Approach                                     |
|------------------------|---------------------------------------------------|----------------------------------------------|
| sessionStorage capture | Add to storageState export                        | CDP `DOMStorage` with `isLocalStorage=false` |
| Multi-origin IndexedDB | Capture/restore IndexedDB for all visited origins | Same lazy restore pattern as localStorage    |

### Longer-Term

| Enhancement                | Description                                       | Challenges               |
|----------------------------|---------------------------------------------------|--------------------------|
| Token expiration detection | Parse JWTs, warn on expired tokens before restore | JWT library, edge cases  |
| Chrome extension export    | Export storage state from regular Chrome          | Extension API, user flow |

### CDP Implementation Notes

Key discoveries from multi-origin localStorage implementation:

**CDP DOMStorage asymmetry:**
- `getDOMStorageItems` CAN read any origin (no frame required)
- `setDOMStorageItem` REQUIRES active frame for target origin

**Lazy restore pattern (validated by Perplexity research):**
1. Store pending storage state in memory
2. Restore each origin's data only when navigating there
3. Track restored origins to avoid double-restore
4. This matches Playwright's internal approach

**For multi-origin IndexedDB:**
- CDP IndexedDB domain can READ any origin's databases
- CDP IndexedDB has NO write API - must use JavaScript
- Same lazy restore pattern applies: inject JS to restore when visiting each origin

## Security Considerations

Storage state files contain authentication credentials. Handle appropriately:

- **Never commit to version control** - Add to `.gitignore`
- **Encrypt at rest** - For long-term storage
- **Delete when done** - Remove after use
- **Scope minimally** - Only export cookies for needed domain
- **Treat as passwords** - Same handling as credential files

## References

- [Playwright storageState documentation](https://playwright.dev/docs/api/class-browsercontext#browser-context-storage-state)
- [Chrome DevTools Protocol - Network.getCookies](https://chromedevtools.github.io/devtools-protocol/tot/Network/#method-getCookies)
- [Chrome Extensions - cookies API](https://developer.chrome.com/docs/extensions/reference/cookies/)