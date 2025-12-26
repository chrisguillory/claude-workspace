# Selenium Browser Automation MCP Server

Browser automation with CDP stealth injection to bypass Cloudflare bot detection. Runs locally via `uv --script` for visible browser monitoring.

## Setup

```bash
# From within the claude-workspace repo:
claude mcp add --scope user selenium-browser-automation -- \
  uv run --project "$(git rev-parse --show-toplevel)/mcp/selenium-browser-automation" \
  --script "$(git rev-parse --show-toplevel)/mcp/selenium-browser-automation/server.py"
```

## Comparison with Claude in Chrome

This server is designed to match or exceed the capabilities of Anthropic's official
[Claude in Chrome extension](docs/claude-in-chrome.md). We iterate faster than official
releases, so gaps should be addressed proactively.

> **Tested against:** Claude in Chrome v1.0.36 (2025-12-23)

> **Legend**: CiC = Claude in Chrome (Anthropic's extension) · SMCP = Selenium MCP (this server)
>
> **Type codes**: 1:1 = direct equivalent · 1:N = one CiC maps to multiple SMCP · 1:2 = one CiC requires two SMCP calls · Gap = CiC has it, SMCP doesn't · SMCP-only = SMCP has it, CiC doesn't · UI/Arch = not applicable (UI feature or architectural difference)

### Feature Summary

| Capability                       |      CiC      |  SMCP   | Notes                                                    |
|----------------------------------|:-------------:|:-------:|----------------------------------------------------------|
| Text extraction (hidden content) |       ✓       |    ✓    | Both extract display:none, visibility:hidden, etc        |
| Text extraction (Shadow DOM)     |       -       |    ✓    | **Advantage**: Traverses web component shadows           |
| Text extraction (filtering)      |       -       |    ✓    | **Advantage**: Filters SCRIPT/STYLE noise                |
| JavaScript execution             |       ✓       |    ✓    | **Parity**: Both execute JS; SMCP adds statement support |
| Console log access               |       ✓       | **Gap** | P1: Chrome reads `console.log/error/warn`                |
| Core Web Vitals                  |       -       |    ✓    | **Advantage**: LCP, CLS, INP, FCP, TTFB                  |
| HAR export                       |       -       |    ✓    | **Advantage**: Full network traffic logging              |
| Proxy/IP rotation                |       -       |    ✓    | **Advantage**: Rate limit bypass                         |
| Network timing details           |     Basic     |    ✓    | **Advantage**: DNS, connect, TLS breakdown               |
| Chrome profiles                  | Via extension |    ✓    | **Advantage**: Direct profile access                     |
| GIF recording                    |       ✓       | **Gap** | P3: Low priority                                         |
| Natural language find            |       ✓       |    -    | Chrome's `find` tool uses semantic matching              |

### CiC Advantages (Gaps to Close)

Capabilities where Claude in Chrome exceeds us:

| Capability                   | What CiC Provides                                            | Priority |
|------------------------------|--------------------------------------------------------------|:--------:|
| ~~**Text Extraction**~~      | ~~Captures all text including hidden content~~ ✅ CLOSED      |  ~~P0~~  |
| **ARIA Whitespace**          | Properly normalized accessible names                         |    P0    |
| ~~**JavaScript Execution**~~ | ~~Run ad-hoc JS for debugging (`javascript_tool`)~~ ✅ CLOSED |  ~~P1~~  |
| **Console Log Access**       | Read browser console (`read_console_messages`)               |    P1    |
| **Form Input**               | Set form values by element reference                         |    P2    |
| **Window Resize**            | Set browser dimensions for responsive testing                |    P2    |
| **GIF Recording**            | Record interactions as animated GIF                          |    P3    |
| **Natural Language Find**    | Find elements by semantic description                        |    -     |

### SMCP Advantages

Capabilities where we exceed Claude in Chrome:

| Capability                  | What It Provides                                   | Use Case                                         |
|-----------------------------|----------------------------------------------------|--------------------------------------------------|
| **Shadow DOM Traversal**    | Extracts text from web component shadow roots      | Modern sites using custom elements               |
| **SCRIPT/STYLE Filtering**  | Removes non-content elements from text extraction  | Clean AI-readable output without code noise      |
| **PRE/CODE Preservation**   | Exact whitespace in preformatted blocks            | Code extraction, config files, ASCII art         |
| **Smart Extraction**        | main > article > body with 500-char threshold      | Better element selection than article-only       |
| **Transparency Metadata**   | `source_element`, `fallback_used`, coverage ratio  | Detect silent partial extraction failures        |
| **Core Web Vitals**         | LCP, CLS, INP, FCP, TTFB with ratings              | Performance auditing, identifying slow pages     |
| **HAR Export**              | Full network traffic in DevTools-compatible format | Deep network debugging, API analysis             |
| **Detailed Network Timing** | DNS, connect, TLS, TTFB breakdown per request      | Identifying slow requests, bottleneck analysis   |
| **Proxy/IP Rotation**       | Authenticated proxies via mitmproxy                | Bypass rate limiting, geographic testing         |
| **Chrome Profiles**         | Direct access to saved browser profiles            | Authenticated sessions without re-login          |
| **CDP Stealth**             | Bypass Cloudflare bot detection                    | Scrape protected sites where official tools fail |

### Test Results (2025-12-23)

Complete comparison testing of all example files with both SMCP and CiC:

| Test File              |  SMCP   |    CiC     |  Winner  | Evidence                                    |
|------------------------|:-------:|:----------:|:--------:|---------------------------------------------|
| accordion-pattern.html |  ✅ 5/5  |   ✅ 5/5    |   Tie    | Both extract hidden accordion content       |
| hidden-content.html    | ✅ 12/12 |  ✅ 12/12   |   Tie    | Both extract all CSS-hidden content         |
| kitchen-sink.html      | ✅ 16/16 | ⚠️ 16/16*  | **SMCP** | CiC: SCRIPT leaked, no Shadow DOM traversal |
| marriott-modals.html   |  ✅ 4/4  |   ✅ 4/4    |   Tie    | Both extract modal hidden content           |
| preformatted.html      |  ✅ 5/5  |  ⚠️ 5/5*   | **SMCP** | CiC normalizes `const    x` → `const x`     |
| semantic-blocks.html   |  ✅ 6/6  |   ❌ 0/6    | **SMCP** | CiC extracted only `<article>` (2 words)    |
| shadow-dom.html        |  ✅ 7/7  |  ⚠️ 7/7*   | **SMCP** | CiC markers from SCRIPT, not shadow DOM     |
| threshold-test.html    | ✅ main  | ⚠️ article | **SMCP** | CiC: 11% coverage vs SMCP: 74% coverage     |
| whitespace.html        |  ✅ 9/9  |   ✅ 9/9    |   Tie    | Both normalize whitespace correctly         |

\* Markers appear but with quality issues (SCRIPT leak, wrong source, normalized whitespace)

**Key Failures Observed:**

1. **semantic-blocks.html**: CiC found `<article>` in Test 4 and extracted ONLY "Article content" (2 words), missing 1500+ chars of actual test content. This is silent partial extraction.

2. **threshold-test.html**: Page has `<main>` containing `<article>`. CiC extracted nested article (~200 chars, 11% coverage). SMCP extracted main (1363 chars, 74% coverage).

3. **shadow-dom.html**: CiC test markers appear from SCRIPT source code (the JavaScript defining web components), NOT from actual shadow DOM traversal.

### Real-World Site Testing (2025-12-23)

| Site            | SMCP                            | CiC                        |  Winner  | Notes                                     |
|-----------------|---------------------------------|----------------------------|:--------:|-------------------------------------------|
| nytimes.com     | ✅ `main`, 13,050 chars, 50%     | ⚠️ `main`, HTML/JS leak    | **SMCP** | CiC leaks `<img>` tags and inline JS      |
| bbc.com         | ✅ `main`, 11,915 chars, 54%     | ✅ `article`, ~11,500 chars |   Tie    | Both clean, different source element      |
| amazon.com      | ✅ `body` fallback, 10,057 chars | ❌ **FAILED**               | **SMCP** | CiC error: "body too large"               |
| docs.python.org | ✅ `[role="main"]`, 1,155 chars  | ✅ `div`, ~1,200 chars      |   Tie    | Both find focused content (SMCP improved) |
| usa.gov         | ✅ `main`, 3,010 chars, 58%      | ⚠️ `main`, CSS leak        | **SMCP** | CiC leaks ~1,500 chars of CSS             |

**Key Findings:**

1. **CiC fails on e-commerce sites**: Amazon triggers "No semantic content element found and page body is too large" error. SMCP's body fallback handles this gracefully.

2. **CiC leaks CSS**: On usa.gov, CiC output starts with ~1,500 characters of inline CSS media queries before actual content.

3. **SMCP transparency enables debugging**: Coverage ratios (50-58%) immediately show when extraction is partial.

### ARIA Snapshot Comparison (`get_aria_snapshot` vs `read_page`)

Comprehensive testing (2025-12-23) of accessibility tree extraction:

| Feature                        | SMCP `get_aria_snapshot` |    CiC `read_page`    | Notes                                   |
|--------------------------------|:------------------------:|:---------------------:|-----------------------------------------|
| **Reference IDs**              |            ❌             | ✅ (ref_1, ref_2, ...) | CiC advantage - enables click(ref)      |
| **aria-hidden exclusion**      |         ❌ (BUG)          |     ❌ (SAME BUG)      | **P0**: Both show aria-hidden content   |
| **Explicit ARIA roles**        |            ✅             |           ✅           | dialog, tablist, menu, etc.             |
| **Named landmarks**            |            ✅             |           ✅           | `navigation "Main nav"`                 |
| **Interactive filter**         |            ❌             |           ✅           | CiC: `filter="interactive"`             |
| **expanded/selected/pressed**  |            ❌             |           ❌           | Neither tracks ARIA states              |
| **valuenow/valuemin/valuemax** |            ❌             |           ❌           | Neither shows slider values             |
| **Native `<details>`**         |    → generic (wrong)     |   → generic (wrong)   | Should be `group`                       |
| **Native `<progress>`**        |    → generic (wrong)     |   → generic (wrong)   | Should be `progressbar`                 |
| **Native `<meter>`**           |    → generic (wrong)     |   → generic (wrong)   | Should be `meter`                       |
| **Native `<dl>/<dt>/<dd>`**    |    → generic (wrong)     |   → generic (wrong)   | Should be term/definition               |
| **Header inside article**      |     → banner (wrong)     |   → banner (wrong)    | Should be `generic` (context-sensitive) |
| **Section without name**       |     → region (wrong)     |   → region (wrong)    | Should be `generic` (context-sensitive) |
| **aria-live attributes**       |            ❌             |           ❌           | Neither shows polite/assertive          |

**Key Finding**: Both implementations have identical native role mapping and context-sensitive role gaps. The P0 aria-hidden bug exists in both.

### `get_aria_snapshot` Roadmap

Prioritized implementation plan based on Phase 6-7 testing:

| Priority | Issue                         | Impact                                    | Test File                  |
|:--------:|-------------------------------|-------------------------------------------|----------------------------|
|  **P0**  | aria-hidden content appearing | Security/privacy - exposes hidden content | `aria-states.html` Test 11 |
|  **P1**  | No reference IDs              | Can't target elements for click/input     | Chrome comparison          |
|  **P1**  | Missing ARIA states           | Can't detect expanded/collapsed/selected  | `aria-states.html`         |
|  **P2**  | Native role mapping gaps      | details/progress/meter show as generic    | `role-mapping.html`        |
|  **P2**  | Context-sensitive roles       | header inside article still shows banner  | `context-roles.html`       |
|  **P2**  | Value attributes              | No valuenow/valuemin/valuemax for sliders | `aria-values.html`         |
|  **P2**  | Live region attributes        | No aria-live/atomic/relevant              | `live-regions.html`        |
|  **P3**  | Interactive filter            | CiC has filter="interactive" option       | Feature gap                |

### Philosophy Comparison

| Aspect                  | SMCP                       | CiC                           |
|-------------------------|----------------------------|-------------------------------|
| **Extraction priority** | main > article > body      | article-first (any article)   |
| **Size threshold**      | 500 chars minimum          | None (extracts tiny articles) |
| **SCRIPT/STYLE/CSS**    | Filtered out               | Leaks into output             |
| **Shadow DOM**          | Traversed (open roots)     | Not traversed                 |
| **PRE/CODE whitespace** | Preserved exactly          | Normalized                    |
| **Transparency**        | `smart_info` with coverage | None (silent extraction)      |
| **Failure mode**        | Graceful fallback to body  | Silent partial extraction     |

**Design Rationale:**

SMCP prioritizes **transparency over magic**. When smart extraction selects a subset of the page:
- `source_element` tells you what was extracted
- `fallback_used` indicates if main/article weren't suitable
- `body_character_count` enables coverage calculation
- Caller can detect if extraction was unexpectedly small

CiC uses **implicit smart extraction** without metadata. If it finds any `<article>`, it extracts just that content regardless of size. The caller has no way to know content was silently truncated.

### Tool Mapping

How CiC tools map to SMCP tools:

| CiC Tool(s)             | SMCP Tool(s)                                    |   Type    | Notes                                                                                                     |
|-------------------------|-------------------------------------------------|:---------:|-----------------------------------------------------------------------------------------------------------|
| `navigate`              | `navigate`                                      |    1:1    | SMCP adds `fresh_browser`, `profile`, `enable_har_capture` params                                         |
| `get_page_text`         | `get_page_text`                                 |  Exceeds  | SMCP adds: Shadow DOM traversal, SCRIPT/STYLE filtering, structured output                                |
| `read_page`             | `get_aria_snapshot`                             |    1:1    | CiC returns ref-based tree; SMCP returns YAML tree                                                        |
| `find`                  | -                                               |    Gap    | CiC uses natural language; SMCP alternative: `get_interactive_elements`                                   |
| `computer`              | `click`, `press_key`, `type_text`, `screenshot` |    1:N    | CiC unified tool; SMCP gaps: scroll, hover, zoom, drag, wait                                              |
| `form_input`            | -                                               |    Gap    | P2: Set form values by element reference ID from `read_page`                                              |
| `javascript_tool`       | `execute_javascript`                            |  Exceeds  | See [JavaScript Execution Comparison](#javascript-execution-comparison)                                   |
| `read_console_messages` | -                                               |    Gap    | P1: Read console.log/error/warn messages with pattern filtering                                           |
| `read_network_requests` | `get_resource_timings`                          |  Exceeds  | Both single call; SMCP adds detailed DNS/connect/TLS/TTFB breakdown via Performance API |
| `resize_window`         | -                                               |    Gap    | P2: Set browser window dimensions for responsive testing                                                  |
| `gif_creator`           | -                                               |    Gap    | P3: Record browser interactions as animated GIF                                                           |
| `upload_image`          | -                                               |    Gap    | Upload screenshot to file input or drag-drop target                                                       |
| `update_plan`           | N/A                                             |    UI     | CiC user approval flow; SMCP doesn't need (different permission model)                                    |
| `shortcuts_*`           | N/A                                             |    UI     | CiC extension shortcuts; not applicable to SMCP                                                           |
| `tabs_*`                | N/A                                             |   Arch    | CiC manages tab groups; SMCP uses single browser with shared session                                      |
| -                       | `get_interactive_elements`                      | SMCP-only | Find clickable elements by CSS selector or text content                                                   |
| -                       | `get_focusable_elements`                        | SMCP-only | Get keyboard-navigable elements sorted by tab order                                                       |
| -                       | `wait_for_network_idle`                         | SMCP-only | Wait for network activity to settle after navigation or clicks                                            |
| -                       | `capture_web_vitals`                            | SMCP-only | LCP, CLS, INP, FCP, TTFB with good/needs-improvement/poor ratings                                         |
| -                       | `export_har`                                    | SMCP-only | Export network traffic to HAR 1.2 format for DevTools analysis                                            |
| -                       | `configure_proxy`, `clear_proxy`                | SMCP-only | Configure/clear authenticated proxy via mitmproxy for IP rotation                                         |
| -                       | `download_resource`                             | SMCP-only | Download files using browser session cookies (bypasses bot detection)                                     |
| -                       | `list_chrome_profiles`                          | SMCP-only | List available Chrome profiles with name, email, directory metadata                                       |

See [docs/claude-in-chrome.md](docs/claude-in-chrome.md) for complete Claude in Chrome tool reference.

### JavaScript Execution Comparison

Detailed comparison of `execute_javascript` (SMCP) vs `javascript_tool` (CiC):

| Test Case             | SMCP Result                          | CiC Result                             |  Winner  |
|-----------------------|--------------------------------------|----------------------------------------|:--------:|
| `NaN`                 | `"NaN"` + note                       | `NaN` (string)                         |   Tie    |
| `Infinity`            | `"Infinity"` + note                  | `Infinity` (string)                    |   Tie    |
| `-0`                  | `"-0"` + note                        | `-0` (string)                          |   Tie    |
| `Symbol('test')`      | `"Symbol(test)"` string              | **ERROR**: Object couldn't be returned | **SMCP** |
| Circular reference    | `{a:1, self:"[Circular Reference]"}` | **ERROR**: Reference chain too long    | **SMCP** |
| `new Error('msg')`    | `{name, message, stack}`             | `{}` (loses all info)                  | **SMCP** |
| `BigInt(42)`          | `"42"`                               | `42n`                                  |   Tie    |
| `{x: NaN}`            | `{x: "[NaN]"}`                       | `{x: null}` (loses info)               | **SMCP** |
| `[1, Infinity, -0]`   | `[1, "[Infinity]", "[-0]"]`          | `[1, null, 0]` (loses info)            | **SMCP** |
| `new WeakMap()`       | `null` + note explaining why         | `{}` (no explanation)                  | **SMCP** |
| `new ArrayBuffer(16)` | `null` + note with workaround        | `{}` (no explanation)                  | **SMCP** |
| `new Blob([...])`     | `null` + note with workaround        | `{}` (no explanation)                  | **SMCP** |
| Generator             | `null` + note with workaround        | `{}` (no explanation)                  | **SMCP** |
| `function test()...`  | Works (undefined)                    | Works (undefined)                      |   Tie    |
| `throw new Error()`   | Works (statement support)            | Works                                  |   Tie    |
| `about:blank`         | ✓ Supported                          | ✗ Not supported                        | **SMCP** |
| `data:` URLs          | ✓ Supported                          | ✗ Not supported                        | **SMCP** |

**Summary:**
- **Tie on special numbers**: Both return string representations (NaN, Infinity, -0); SMCP adds explanatory notes
- **SMCP advantages**: Symbol handling, circular reference handling, Error object extraction, **nested special values** (`{x: NaN}` → `{x: "[NaN]"}`), explanatory notes for unserializable types, `about:blank`/`data:` URL support

**Design Philosophy:**
- SMCP uses structured `{success, result, result_type, note}` responses with explanatory notes
- CiC uses simpler return values but fails silently on edge cases or throws errors
- SMCP prioritizes **debuggability** - notes explain *why* values can't be serialized and suggest workarounds

## Navigation Best Practices

**Prefer ARIA snapshots over screenshots for understanding page structure:**

| Task | Use This | Not This |
|------|----------|----------|
| Find interactive elements | `get_aria_snapshot(selector="body")` | `screenshot()` |
| Locate buttons/links | `get_interactive_elements(selector_scope="body", ...)` | `screenshot()` |
| Understand page layout | `get_aria_snapshot()` | `get_page_html()` |
| Debug visual issues | `screenshot()` | - |
| Verify visual state | `screenshot()` | - |

**Why ARIA over screenshots:**
- ARIA snapshots are text-based, faster to process, and more precise
- Screenshots require visual interpretation and are larger payloads
- ARIA provides semantic structure (roles, names, states)
- Use screenshots only when visual verification is needed

**Navigation workflow:**
```
1. navigate(url)                    # Load page
2. get_aria_snapshot("body")        # Understand structure
3. get_interactive_elements(...)    # Find clickable elements
4. click(selector)                  # Interact
5. wait_for_network_idle()          # Wait for dynamic content
```

## Performance Analysis

### What We Capture

**Core Web Vitals** (`capture_web_vitals`):
- LCP (Largest Contentful Paint) - main content load time
- CLS (Cumulative Layout Shift) - visual stability
- INP (Interaction to Next Paint) - responsiveness (requires user interaction)
- FCP (First Contentful Paint) - initial render
- TTFB (Time to First Byte) - server response time

**Resource Timings** (`get_resource_timings`):
- All resource requests with timing breakdown (no setup required)
- Identifies slow API calls automatically
- Breakdown: DNS, connect, SSL, wait (TTFB), receive

**HAR Export** (`export_har`) - opt-in for full HTTP details:
- Request/response headers, status codes
- Optional response bodies
- Requires `enable_har_capture=True` at browser init (adds overhead)

**Workflow for performance investigation:**
```
1. navigate(url)                    # Load page
2. click(selector)                  # Trigger actions
3. wait_for_network_idle()          # Wait for API calls
4. get_resource_timings(min_duration_ms=500)  # Show slow requests
5. capture_web_vitals()             # Get Core Web Vitals
```

For detailed HTTP data (headers, bodies), use HAR capture:
```
1. navigate(url, fresh_browser=True, enable_har_capture=True)
2. [interact with page]
3. export_har("capture.har", include_response_bodies=True)
```

### Planned Capabilities

Features with complete implementation designs, ready for development.

#### CPU Profiler

**Status:** Implementation design complete

Captures JavaScript execution using CDP Profiler, exporting `.cpuprofile` format compatible with DevTools.

**Key capabilities:**
- Hot function detection (functions consuming >1% execution time)
- React pattern recognition (render methods, hooks, reconciliation)
- Flame graph export for DevTools import

**When to use:** Network timings look fine but UI is sluggish during interaction.

**Technical approach:** CDP `Profiler.enable` → `setSamplingInterval` → `start`/`stop` → analyze and export `.cpuprofile`

#### Tracing & Timeline

**Status:** Implementation design complete

Full DevTools-style performance traces with long task detection and timeline analysis.

**Key capabilities:**
- Long task detection (>50ms main thread blocking)
- Total Blocking Time (TBT) calculation
- GC pause and layout thrashing detection
- FCP/LCP extraction from trace events
- Export to `chrome://tracing` or DevTools Performance panel

**When to use:** Deep performance investigations requiring precise timing relationships between rendering, scripting, and network.

**Technical approach:** CDP `Tracing.start` with `transferMode: "ReturnAsStream"` (works around Selenium's event listener limitations) → category presets (minimal/standard/comprehensive) → trace JSON export

---

### What DevTools Has That We Don't (Yet)

Ideas without implementation plans:

1. **Console errors/warnings** - JS exceptions, React errors, deprecation warnings. Useful for debugging but potentially noisy. (Also a [Claude in Chrome parity item](#comparison-with-claude-in-chrome) at P1 priority.)

2. **HTTP status codes** - Resource Timing API doesn't expose 4xx/5xx errors. Would require always-on CDP logging (has overhead).

3. **Request/response bodies** - Actual API payloads. Heavy, specialized use case.

4. **Memory profiling** - Heap snapshots, memory leak detection. Specialized debugging.

## Tool Reference

### Navigation & Content
- `navigate(url, fresh_browser?, profile?, enable_har_capture?)` - Load URL with optional Chrome profile
- `get_page_text(selector?, include_images?)` - Smart content extraction with semantic element priority (see below)
- `get_page_html(selector?, limit?)` - Extract raw HTML source or specific elements
- `get_aria_snapshot(selector, include_urls?)` - Semantic page structure
- `screenshot(filename, full_page?)` - Capture viewport or full page

### Smart Extraction (`get_page_text`)

**Default behavior (`selector='auto'`):**

Smart extraction focuses on main content by trying semantic elements in priority order:
1. `<main>` or `[role="main"]` (if >500 characters)
2. `<article>` element (if >500 characters)
3. Falls back to `<body>`

This typically captures 80-90% of meaningful content while excluding navigation, sidebars, and footers.

**Transparency metadata (auto mode only):**

```python
result = get_page_text()  # Uses smart extraction

result.source_element      # "main", "article", or "body"
result.smart_info.fallback_used       # True if no main/article found
result.smart_info.body_character_count  # Total body chars for coverage calc
# Coverage ratio: result.character_count / result.smart_info.body_character_count
```

**Explicit extraction:**

```python
get_page_text(selector="body")     # Full page (no smart_info)
get_page_text(selector=".content") # Specific element (no smart_info)
```

**Image alt text (`include_images=True`):**

```python
get_page_text(include_images=True)
# Output includes: [Image: A chart showing Q3 revenue growth]
# Images without alt: [Image: (no alt)]
```

**Extraction features:**
- Traverses Shadow DOM components
- Filters SCRIPT/STYLE/TEMPLATE noise
- Preserves whitespace in PRE/CODE/TEXTAREA
- Normalizes whitespace elsewhere

### Interaction
- `get_interactive_elements(selector_scope, text_contains?, tag_filter?, limit?)` - Find clickable elements
- `get_focusable_elements(only_tabbable?)` - Keyboard-navigable elements
- `click(selector, wait_for_network?, network_timeout?)` - Click element
- `press_key(key)` - Keyboard input (ENTER, ESCAPE, CONTROL+A, etc.)
- `type_text(text, delay_ms?)` - Type text character by character
- `wait_for_network_idle(timeout?)` - Wait for network activity to settle

### Performance
- `capture_web_vitals(timeout_ms?)` - Core Web Vitals metrics
- `get_resource_timings(clear_resource_timing_buffer?, min_duration_ms?)` - Resource timing via Performance API (no setup required)
- `export_har(filename, include_response_bodies?, max_body_size_mb?)` - Export to HAR file (requires `enable_har_capture=True` on navigate)

### Utilities
- `download_resource(url, output_filename)` - Download with session cookies
- `list_chrome_profiles(verbose?)` - Available Chrome profiles

### Proxy (Rate Limit Bypass)
- `configure_proxy(host, port, username, password)` - Configure authenticated proxy via mitmproxy
- `clear_proxy()` - Stop proxy and return to direct connection

## Authenticated Proxy Support

Bypass IP-based rate limiting using residential proxies (e.g., Bright Data).

### Quick Start

```python
# 1. Configure proxy (starts mitmproxy subprocess)
configure_proxy(
    host="brd.superproxy.io",
    port=33335,
    username="brd-customer-XXXXX-zone-residential-country-us",
    password="YOUR_PASSWORD"
)

# 2. Navigate - requests now go through proxy
navigate("https://api.ipify.org")  # Shows proxy IP, not your real IP

# 3. For IP rotation, use fresh_browser=True
navigate(url, fresh_browser=True)  # Forces new IP from proxy pool

# 4. Clean up when done
clear_proxy()
```

### Why mitmproxy? (Chrome Extensions Don't Work)

We extensively tested Chrome extension-based proxy authentication and discovered it **fundamentally cannot work reliably**:

| Approach | Result | Why It Fails |
|----------|--------|--------------|
| Manifest V3 extension | ❌ Auth dialog appears | Service worker not ready when auth challenge occurs |
| Manifest V2 extension | ❌ Auth dialog appears | Race condition: auth fires before extension loads |
| Selenium Wire | ❌ Broken | Deprecated, incompatible with Python 3.13 |
| **mitmproxy** | ✅ Works | Handles auth at proxy layer, not browser |

**The solution**: mitmproxy runs locally on `localhost:8080`. Chrome connects to it without authentication. mitmproxy forwards requests to the upstream proxy (Bright Data) with credentials.

```
Chrome → localhost:8080 (mitmproxy) → brd.superproxy.io:33335 (Bright Data)
         [no auth needed]              [mitmproxy handles auth]
```

### IP Rotation Behavior

| Scenario | IP Behavior |
|----------|-------------|
| Same browser session | Same IP (connection reuse) |
| `fresh_browser=True` | **New IP** from proxy pool |
| New `configure_proxy()` call | New IP (restarts mitmproxy) |

### Rate Limit Bypass Pattern

For sites with aggressive rate limiting (e.g., government record searches):

```python
for item in items_to_search:
    # Each iteration gets fresh browser = new IP
    navigate(search_url, fresh_browser=True)

    # Perform search
    click(input_selector)
    type_text(search_term)
    click(submit_selector)
    wait_for_network_idle()

    # Extract results
    results = get_page_text()

    # Human-like delay between searches
    time.sleep(random.uniform(2, 5))
```

### Requirements

- **mitmproxy**: Install with `brew install mitmproxy` (macOS) or `pip install mitmproxy`
- **Proxy credentials**: Bright Data or similar residential proxy service

### Bright Data Specifics

- Default port: `33335` (HTTP) or `22228` (SOCKS5)
- Username format: `brd-customer-{ID}-zone-{ZONE}-country-{CC}`
- Country codes: `us`, `gb`, `de`, etc.
- Session stickiness: Add `-session-{ID}` to username for sticky sessions

## Architecture

### Browser Model

Unlike Claude in Chrome (which manages multiple tabs per conversation), this server uses a **single browser instance** with shared state:

```
Selenium MCP Server
├── Browser Instance (single, lazy-initialized)
│   ├── CDP Connection
│   └── Session State (cookies, localStorage)
├── Proxy Layer (optional)
│   └── mitmproxy → upstream proxy
└── Temp Files (auto-cleanup on shutdown)
```

**Why single browser?** Simplicity and session sharing. All tool calls share the same cookies and authentication state, which is ideal for workflows that require login persistence.

### Process Architecture

```
Selenium MCP Server (Python)
├── chromedriver (1 process)
└── Chrome (5-6 processes: main, GPU, renderer, network, utility)
```

Compare with Claude in Chrome, which is a browser extension controlling existing tabs via DevTools Protocol - no additional processes.

**Cleanup on reconnect:** Signal handlers (SIGTERM/SIGINT) ensure browser processes close cleanly when `claude mcp reconnect` terminates the server.

### CDP Stealth Injection

**Problem**: Modern bot detection (Cloudflare, DataDome) fingerprints automation by detecting WebDriver properties, CDP artifacts, and missing browser features.

**Solution**: We inject JavaScript via CDP *before* page load to mask automation signals. This works where Playwright's stealth plugins fail because:
- Injection happens at the CDP level, not via browser extensions
- Scripts execute before any page JavaScript runs
- We patch `navigator.webdriver`, `window.chrome`, and other fingerprinting targets

### Session Management

| Aspect             | Behavior                                                     |
|--------------------|--------------------------------------------------------------|
| Cookie persistence | Maintained across all tool calls                             |
| Fresh session      | `navigate(url, fresh_browser=True)` starts clean             |
| Profile loading    | `navigate(url, profile="Default")` uses saved Chrome profile |

### Network Capture Design

**Two complementary systems:**

1. **Performance API** (`get_resource_timings`) - JavaScript API running in page context. Always available, lightweight, provides timing breakdown. Limited to what the Performance API exposes (no headers, no bodies).

2. **Chrome Performance Logging** (`export_har`) - Chrome's internal logging of CDP Network events. Opt-in due to overhead. Provides full HTTP details in HAR format.

**Why not CDP Network interception?**

We intentionally don't expose direct CDP Network event streaming for request interception. Reasons:
- WebDriver BiDi is replacing CDP as the standard protocol
- Two-system approach covers 90% of use cases
- Adding CDP would create Chrome-only, deprecated functionality

When BiDi matures with cross-browser network interception support, we may add it then.

### Why Not Playwright?

We chose Selenium over Playwright for this MCP server because:
1. **CDP stealth works better** - Our injection approach bypasses detection that blocks Playwright
2. **Chrome profile support** - Native access to saved browser profiles for authenticated sessions
3. **mitmproxy integration** - Cleaner proxy authentication than Playwright's approach
