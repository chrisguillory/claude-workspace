# Selenium Browser Automation MCP Server

Browser automation with CDP stealth injection to bypass Cloudflare bot detection. Runs locally via `uv --script` for visible browser monitoring.

## Setup

```bash
# From within the claude-workspace repo:
claude mcp add --scope user selenium-browser-automation -- \
  uv run --project "$(git rev-parse --show-toplevel)/mcp/selenium-browser-automation" \
  --script "$(git rev-parse --show-toplevel)/mcp/selenium-browser-automation/server.py"
```

## Navigation Best Practices

**Prefer ARIA snapshots over screenshots for understanding page structure:**

| Task | Use This | Not This |
|------|----------|----------|
| Find interactive elements | `get_aria_snapshot(selector="body")` | `screenshot()` |
| Locate buttons/links | `get_interactive_elements(selector_scope="body", ...)` | `screenshot()` |
| Understand page layout | `get_aria_snapshot()` | `get_page_content(format="html")` |
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

**Network Timings** (`start_network_capture` + `get_network_timings`):
- All resource requests with timing breakdown
- Identifies slow API calls automatically
- Breakdown: DNS, connect, SSL, wait (TTFB), receive

**Workflow for performance investigation:**
```
1. start_network_capture()          # Enable capture
2. navigate(url)                    # Load page
3. click(selector)                  # Trigger actions
4. wait_for_network_idle()          # Wait for API calls
5. get_network_timings(min_duration_ms=500)  # Show slow requests
6. capture_web_vitals()             # Get Core Web Vitals
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

1. **Console errors/warnings** - JS exceptions, React errors, deprecation warnings. Useful for debugging but potentially noisy.

2. **HTTP status codes** - Resource Timing API doesn't expose 4xx/5xx errors. Would require always-on CDP logging (has overhead).

3. **Request/response bodies** - Actual API payloads. Heavy, specialized use case.

4. **Memory profiling** - Heap snapshots, memory leak detection. Specialized debugging.

## Tool Reference

### Navigation & Content
- `navigate(url, fresh_browser?, profile?, enable_har_capture?)` - Load URL with optional Chrome profile
- `get_page_content(format, selector?, limit?)` - Extract text/HTML
- `get_aria_snapshot(selector, include_urls?)` - Semantic page structure
- `screenshot(filename, full_page?)` - Capture viewport or full page

### Interaction
- `get_interactive_elements(selector_scope, text_contains?, tag_filter?, limit?)` - Find clickable elements
- `get_focusable_elements(only_tabbable?)` - Keyboard-navigable elements
- `click(selector, wait_for_network?, network_timeout?)` - Click element
- `press_key(key)` - Keyboard input (ENTER, ESCAPE, CONTROL+A, etc.)
- `type_text(text, delay_ms?)` - Type text character by character
- `wait_for_network_idle(timeout?)` - Wait for network activity to settle

### Performance
- `capture_web_vitals(timeout_ms?)` - Core Web Vitals metrics
- `start_network_capture(resource_types?)` - Enable network timing capture
- `get_network_timings(clear?, min_duration_ms?)` - Retrieve captured timings
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
    results = get_page_content(format="text")

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

- **CDP Stealth Injection**: Bypasses Cloudflare where Playwright fails
- **Shared Session**: All requests share cookies/auth state
- **Lazy Browser Init**: Browser created on first navigation
- **Temp File Cleanup**: Auto-cleanup on shutdown
