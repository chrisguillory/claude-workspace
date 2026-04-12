# Browser Data Taxonomy & Ecosystem Guide

Understanding browser data organization and how automation tools approach state management.

---

## Overview

This document provides a comprehensive reference for understanding:
1. How browsers organize user data internally
2. How existing automation frameworks handle state persistence
3. Why this MCP server uses `profile_state` terminology
4. Format compatibility considerations

---

## Chrome Data Hierarchy

Chrome organizes data in a strict hierarchy. Understanding this structure clarifies what we capture and why our terminology choices matter.

```
BROWSER (Chrome Application)
│
└── USER DATA DIRECTORY
    │   Location: ~/Library/Application Support/Google/Chrome/  (macOS)
    │             %LOCALAPPDATA%\Google\Chrome\User Data\       (Windows)
    │             ~/.config/google-chrome/                      (Linux)
    │
    │   Contains: Global settings, crash reports, Safe Browsing data
    │
    └── PROFILE (Default/, Profile 1/, Profile 2/, etc.)
        │
        │   A complete user identity container. Each profile is isolated.
        │
        ├── BROWSING DATA
        │   │
        │   │   Clearable via "Clear browsing data" dialog.
        │   │   Chrome separates these into distinct categories:
        │   │
        │   ├── COOKIES
        │   │   │   Location: Cookies (SQLite database)
        │   │   │
        │   │   │   Traditional HTTP cookies including:
        │   │   │   • Session cookies (expire on browser close)
        │   │   │   • Persistent cookies (expire on set date)
        │   │   │   • HttpOnly cookies (not accessible via JS)
        │   │   │   • Secure cookies (HTTPS only)
        │   │   │   • SameSite cookies (CSRF protection)
        │   │   │
        │   │   └── Transmitted with every HTTP request to matching domains
        │   │
        │   └── SITE DATA
        │       │   Location: Various (Local Storage/, IndexedDB/, etc.)
        │       │
        │       │   HTML5 Web Storage mechanisms:
        │       │
        │       ├── localStorage
        │       │   • Per-origin key-value storage
        │       │   • Persists until explicitly cleared
        │       │   • ~5-10MB per origin
        │       │   • Synchronous API
        │       │
        │       ├── sessionStorage
        │       │   • Per-origin, per-tab storage
        │       │   • Cleared when tab closes
        │       │   • ~5-10MB per origin
        │       │   • Synchronous API
        │       │
        │       ├── IndexedDB
        │       │   • Per-origin structured database
        │       │   • Supports indexes, transactions, cursors
        │       │   • Large storage capacity (quota-based)
        │       │   • Asynchronous API
        │       │
        │       ├── Cache Storage (Service Workers)
        │       │   • Request/response pair caching
        │       │   • Used by Progressive Web Apps
        │       │
        │       └── Web SQL (Deprecated)
        │           • SQLite-based, being phased out
        │
        ├── EXTENSIONS
        │   │   Location: Extensions/
        │   │
        │   ├── Installed extensions and their versions
        │   ├── Extension-specific storage (chrome.storage API)
        │   ├── Extension preferences and state
        │   └── Content scripts and background pages
        │
        ├── SETTINGS
        │   │   Location: Preferences (JSON file)
        │   │
        │   ├── Browser preferences (homepage, search engine, etc.)
        │   ├── Site-specific permissions (camera, mic, notifications)
        │   ├── Content settings (JavaScript, cookies, popups)
        │   └── Sync settings
        │
        ├── BOOKMARKS
        │   │   Location: Bookmarks (JSON file)
        │   │
        │   └── URL bookmarks organized in folders
        │
        └── CREDENTIALS
            │   Location: Login Data, Web Data (SQLite databases)
            │
            ├── Saved passwords (encrypted)
            ├── Payment methods (encrypted)
            └── Addresses and autofill data
```

### Key Insight: Cookies vs Site Data

Chrome deliberately separates **Cookies** and **Site Data** in its UI and documentation:

| Aspect             | Cookies                         | Site Data                         |
|--------------------|---------------------------------|-----------------------------------|
| **Transmission**   | Sent with HTTP requests         | Browser-only, never transmitted   |
| **Security Model** | Domain + path scoping           | Origin-only (scheme://host:port)  |
| **Size Limits**    | ~4KB per cookie                 | 5MB+ per origin                   |
| **Expiration**     | Set by server or script         | Manual or programmatic clearing   |
| **API**            | `document.cookie`, HTTP headers | `localStorage`, `indexedDB`, etc. |

When Chrome displays "Cookies and site data" together, it's combining these for user convenience, but they remain technically distinct systems.

---

## Browser Vendor Terminology

Different browsers use similar but subtly different terminology:

### Chrome (Chromium)

| Term                    | Meaning                                        |
|-------------------------|------------------------------------------------|
| **User Data Directory** | Top-level folder containing all profiles       |
| **Profile**             | Complete user identity container               |
| **Browsing Data**       | Clearable data (history, cookies, cache, etc.) |
| **Site Data**           | HTML5 storage: localStorage, IndexedDB, etc.   |
| **Cookies**             | HTTP cookies (separate from Site Data)         |
| **Preferences**         | Browser and site-specific settings             |

### Firefox

| Term        | Meaning                                                   |
|-------------|-----------------------------------------------------------|
| **Profile** | Atomic user identity unit (less hierarchical than Chrome) |
| **Places**  | Bookmarks and history database                            |
| **Storage** | Site data including localStorage, IndexedDB               |
| **Cookies** | HTTP cookies                                              |

Firefox treats profiles more atomically - the profile IS the container, with less internal hierarchy exposed to users.

### Microsoft Edge

Edge is Chromium-based and mirrors Chrome's terminology exactly:
- User Data Directory
- Profiles
- Browsing Data
- Site Data

### Common Ground

**"Profile" is the universal concept across all major browsers.** It represents a user identity container that holds all browser state for that identity. This is why `profile_state` is the most accurate and generalizable term for what we capture.

---

## Automation Framework Approaches

Different automation tools have taken different approaches to browser state management:

### Playwright: `storageState`

**Terminology**: Storage-centric (emphasizes mechanism)

```javascript
// Capture
await context.storageState({ path: 'auth.json' });

// Restore
const context = await browser.newContext({ storageState: 'auth.json' });
```

**What it captures**:
- Cookies (all types)
- localStorage (per-origin)
- IndexedDB (as of recent versions)

**Characteristics**:
- Unified abstraction for auth persistence
- JSON format with `cookies` and `origins` arrays
- Context-level API (not page-level)
- No support for extensions, permissions, or settings

**Naming Analysis**: "Storage state" emphasizes the technical mechanism (browser storage APIs) rather than the conceptual purpose (preserving user identity state).

---

### Cypress: `cy.session()`

**Terminology**: Session-centric (emphasizes lifecycle)

```javascript
cy.session('user', () => {
  cy.visit('/login');
  cy.get('#username').type('user');
  cy.get('#password').type('pass');
  cy.get('form').submit();
}, {
  validate: () => {
    cy.getCookie('auth_token').should('exist');
  }
});
```

**What it captures**:
- Cookies
- localStorage
- sessionStorage

**Characteristics**:
- Session lifecycle management (create, validate, cache)
- Automatic cache invalidation
- Named sessions for test organization
- No IndexedDB support

**Naming Analysis**: "Session" is familiar to web developers but carries ambiguity - could mean HTTP session, browser session, or user session. Cypress mitigates this through documentation, but the term itself doesn't clearly indicate scope.

---

### Puppeteer: Granular Approach

**Terminology**: None unified - each storage type handled separately

```javascript
// Cookies
const cookies = await page.cookies();
await page.setCookie(...cookies);

// localStorage (via evaluate)
const storage = await page.evaluate(() => JSON.stringify(localStorage));
await page.evaluate(data => {
  const parsed = JSON.parse(data);
  Object.keys(parsed).forEach(key => localStorage.setItem(key, parsed[key]));
}, storage);
```

**What it captures**: Depends on what you implement manually

**Characteristics**:
- Maximum flexibility
- No unified abstraction
- Developer assembles state capture/restore manually
- Page-level cookie API (vs context-level in Playwright)

**Naming Analysis**: No terminology choice - Puppeteer delegates the abstraction decision to the developer.

---

### Selenium (Native)

**Terminology**: Cookie-focused only

```python
# Cookies only
cookies = driver.get_cookies()
driver.add_cookie(cookie)

# No built-in localStorage/sessionStorage capture
# Must use execute_script() manually
```

**What it captures**: Only cookies natively

**Characteristics**:
- Cookie management only in WebDriver spec
- No localStorage/sessionStorage in standard API
- Must use JavaScript execution for other storage
- No unified state abstraction

**Naming Analysis**: Selenium's limited scope means no terminology decision was needed - it's just "cookies."

---

### Our MCP Server: `profile_state`

**Terminology**: Profile-centric (emphasizes the container concept)

```python
# Capture from running browser
save_profile_state("github.json", include_indexeddb=True)

# Export from Chrome profile on disk
export_chrome_profile_state(chrome_profile="Default", output_file="github.json")

# Restore into browser
navigate_with_profile_state("https://github.com", profile_state_file="github.json")
```

**What it captures**:
- Cookies (all types)
- localStorage (per-origin)
- sessionStorage (per-origin)
- IndexedDB (optional, per-origin)

**Future expansion**:
- Extension state
- Site permissions
- Browser preferences

**Naming Rationale**:
1. **Hierarchically correct**: Profile is the right abstraction level
2. **Browser-vendor aligned**: All browsers use "profile" terminology
3. **Expansion-friendly**: Natural to add extensions, permissions, settings
4. **Semantically clear**: "State" indicates portable, capturable data

---

## Ecosystem Comparison Matrix

| Feature                 | Playwright     | Cypress   | Puppeteer | Selenium       | **This MCP**    |
|-------------------------|----------------|-----------|-----------|----------------|-----------------|
| **Term Used**           | `storageState` | `session` | (none)    | (cookies only) | `profile_state` |
| **Cookies**             | Yes            | Yes       | Yes       | Yes            | Yes             |
| **localStorage**        | Yes            | Yes       | Manual    | Manual         | Yes             |
| **sessionStorage**      | No             | Yes       | Manual    | Manual         | Yes             |
| **IndexedDB**           | Yes            | No        | Manual    | Manual         | Yes (optional)  |
| **Unified API**         | Yes            | Yes       | No        | No             | Yes             |
| **Chrome Export**       | No             | No        | No        | No             | **Yes**         |
| **Future: Extensions**  | No             | No        | No        | No             | Planned         |
| **Future: Permissions** | No             | No        | No        | No             | Planned         |

---

## Our Terminology Choice: Why `profile_state`

### Candidates Considered

| Term                | Pros                            | Cons                                      | Verdict    |
|---------------------|---------------------------------|-------------------------------------------|------------|
| `session`           | Short, familiar                 | Implies temporary/expiring                | Rejected   |
| `storage_state`     | Playwright standard             | Mechanism over purpose                    | Rejected   |
| `auth_state`        | Clear purpose                   | Too narrow (we save non-auth data)        | Rejected   |
| `site_data`         | Chrome's term                   | Excludes cookies in Chrome's taxonomy     | Rejected   |
| `browser_state`     | Clear scope                     | Redundant with "chrome" in export         | Rejected   |
| `web_state`         | Clean compound                  | Less common term                          | Considered |
| **`profile_state`** | Correct abstraction, expandable | Potential confusion with runtime profiles | **Chosen** |

### Why `profile_state` Won

1. **Correct Abstraction Level**: Profile is the container for all the data we capture (and will capture in the future)

2. **Browser Vendor Alignment**: Chrome, Firefox, and Edge all use "profile" as the core concept

3. **Future-Proof**: Adding extensions, permissions, or settings naturally fits under "profile"

4. **Semantic Clarity**: "State" indicates a snapshot that can be captured, serialized, and restored

5. **Developer Intuition**: Testing showed developers correctly interpret `save_profile_state()` as "save the browser profile's state"

### Distinguishing from Runtime Profile Loading

We previously had (and removed) a feature to run Selenium with a Chrome profile directory at runtime. This was unreliable because Chrome locks active profiles.

**Clear distinction**:
- `profile_directory` / `profile_path` = Filesystem location (removed feature)
- `profile_state` = Serialized, portable data (current feature)

The word "state" is key - it clarifies we're dealing with extracted, portable data, not the live profile directory.

---

## Profile State Format

Our format is designed from scratch for the `profile_state` concept, optimized for:
- Semantic clarity
- Future expansion (extensions, permissions, settings)
- Origin-centric organization

```json
{
  "schema_version": "1.0",
  "captured_at": "2024-01-15T10:30:00Z",

  "cookies": [
    {
      "name": "auth_token",
      "value": "abc123",
      "domain": ".example.com",
      "path": "/",
      "expires": 1735689600,
      "http_only": true,
      "secure": true,
      "same_site": "Lax"
    }
  ],

  "origins": {
    "https://example.com": {
      "local_storage": {
        "user_id": "42",
        "preferences": "{\"theme\":\"dark\"}"
      },
      "session_storage": {
        "nav_state": "dashboard"
      },
      "indexed_db": [
        {
          "name": "MyAppDB",
          "version": 1,
          "object_stores": [...]
        }
      ]
    }
  },

  "extensions": {},    // Future: extension state
  "permissions": {},   // Future: site permissions
  "preferences": {}    // Future: browser preferences
}
```

### Design Principles

1. **Origin-centric storage**: All storage types grouped under their origin (not separate arrays)
2. **Snake_case keys**: Consistent with Python conventions
3. **Expansion slots**: Empty objects for future data types
4. **Metadata**: Schema version and capture timestamp for debugging

### Format Adapters (Future)

If interoperability with other tools is needed, **adapters** can convert between formats:

```python
# Hypothetical future adapters
export_playwright_format(profile_state) -> playwright_storage_state
import_playwright_format(playwright_file) -> ProfileState
```

This keeps our core format clean while enabling interop as needed.

---

## Glossary

| Term                    | Definition                                                                         |
|-------------------------|------------------------------------------------------------------------------------|
| **Profile**             | A user identity container in the browser holding all user-specific data            |
| **Profile State**       | Serialized, portable representation of profile data (cookies, storage, etc.)       |
| **User Data Directory** | Chrome's top-level folder containing all profiles                                  |
| **Browsing Data**       | Chrome's umbrella term for clearable user data                                     |
| **Site Data**           | Chrome's term for HTML5 storage (localStorage, IndexedDB, etc.) - excludes cookies |
| **Storage State**       | Playwright's term for cookies + localStorage + IndexedDB                           |
| **Session**             | Cypress's term for cached browser state (cookies, localStorage, sessionStorage)    |
| **Origin**              | A unique combination of scheme, host, and port (e.g., `https://example.com:443`)   |

---

## References

- [Chromium User Data Directory Documentation](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/user_data_dir.md)
- [Chromium Preferences System](https://www.chromium.org/developers/design-documents/preferences/)
- [Playwright Authentication Docs](https://playwright.dev/docs/auth)
- [Cypress Session API](https://docs.cypress.io/api/commands/session)
- [MDN Web Storage API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API)
- [MDN IndexedDB API](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)