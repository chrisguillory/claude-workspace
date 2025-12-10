# Selenium Browser Automation MCP Server

Browser automation with CDP stealth injection to bypass Cloudflare bot detection. Runs locally via `uv --script` for visible browser monitoring.

## Setup

```bash
claude mcp add --transport stdio selenium-browser-automation -- \
  uv run --script "$(git rev-parse --show-toplevel)/mcp/selenium-browser-automation/server.py"
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

### What DevTools Has That We Don't (Yet)

Future capabilities we could add:

1. **Console errors/warnings** - JS exceptions, React errors, deprecation warnings. Useful for debugging but potentially noisy.

2. **HTTP status codes** - Our Resource Timing API doesn't expose 4xx/5xx errors. Would require always-on CDP logging (has overhead).

3. **Request/response bodies** - Actual API payloads. Heavy, specialized use case.

4. **Memory profiling** - Heap snapshots, memory leak detection. Specialized debugging.

5. **CPU profiling** - Flame graphs, long task detection. Performance deep-dives.

For most performance investigations, network timings + Core Web Vitals are sufficient to identify bottlenecks.

## Tool Reference

### Navigation & Content
- `navigate(url, fresh_browser?, profile?)` - Load URL with optional Chrome profile
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

### Utilities
- `download_resource(url, output_filename)` - Download with session cookies
- `list_chrome_profiles(verbose?)` - Available Chrome profiles

## Architecture

- **CDP Stealth Injection**: Bypasses Cloudflare where Playwright fails
- **Shared Session**: All requests share cookies/auth state
- **Lazy Browser Init**: Browser created on first navigation
- **Temp File Cleanup**: Auto-cleanup on shutdown