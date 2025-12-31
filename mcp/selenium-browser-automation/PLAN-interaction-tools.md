# Selenium MCP: Interaction Tools Enhancement Plan

## Status Summary

**Completed (2025-12-29):**
- `hover` - Implemented at line 1686 using ActionChains
- `sleep` - Implemented at line 1882 using asyncio.sleep (renamed from `wait` 2025-12-30)
- `resize_window` - Implemented at line 1849 using set_window_size
- `get_console_logs` - Implemented at line 2371 using driver.get_log("browser")
- Browser logging enabled in get_browser() at line 491
- ActionChains import added at line 42
- Hover testing vs CiC documented in README (lines 564-587)

**Wait Tools Completed (2025-12-30):**
- `sleep(duration_ms)` - Renamed from `wait()`, added validation + graduated warnings
- `wait_for_selector(selector, state, timeout)` - 4 states: visible, hidden, attached, detached

**Hover Enhancements Completed (2025-12-30):**
- ✅ Duration validation (0-30000ms) - lines 1710-1714
- ✅ Scroll into view - lines 1724-1729
- ✅ Visibility check - lines 1731-1734
- ✅ Stability check (multi-signal: RAF timing, getAnimations, 5px threshold) - lines 1736-1800
- ✅ Pointer events check (elementFromPoint) - lines 1802-1823

**Hover Test Artifacts Created (2025-12-30):**
- `examples/hover-stability.html` - 10 stability detection test scenarios
- `examples/hover-visibility.html` - 8 visibility check test scenarios
- `examples/hover-occlusion.html` - 8 occlusion detection test scenarios
- `tests/hover_tests.yaml` - YAML test definitions (30+ test cases)
- `tests/HOVER_TEST.md` - Manual test documentation (14 procedures)

**This document now tracks: Remaining enhancements, edge cases, and future work.**

---

## Wait Strategy Architecture (Perplexity Research 2025-12-30)

### The Foundational Wait Taxonomy

Modern browser automation frameworks have converged on a hierarchical approach to waiting that distinguishes between **explicit timing requirements** and **implicit readiness conditions**. This is not arbitrary - it reflects the genuine diversity of synchronization scenarios in web automation.

| Wait Type | Purpose | When to Use | Framework Equivalent |
|-----------|---------|-------------|---------------------|
| **Deterministic** | Fixed delay | Known timing (animations, debounce) | Playwright `waitForTimeout`, Selenium `sleep` |
| **Network-based** | Wait for idle | AJAX, dynamic content loading | Playwright `waitForLoadState('networkidle')` |
| **Selector-based** | Wait for element | Specific UI state required | Playwright `waitForSelector`, Selenium `WebDriverWait` |
| **Function-based** | Custom condition | Complex application state | Playwright `waitForFunction`, Puppeteer `waitForFunction` |

**Key Insight:** Playwright's documentation **explicitly discourages** `waitForTimeout()` as an anti-pattern that causes flakiness. However, they still provide it because certain scenarios genuinely require fixed delays (CSS transitions, debounce implementations). The key is that deterministic waits should be a **minority** of wait operations.

### Our Current Implementation

| Tool | Type | Status | Gap Analysis |
|------|------|--------|--------------|
| `sleep(duration_ms)` | Deterministic | ✅ Implemented 2025-12-30 | Renamed from `wait()`, added validation + graduated warnings |
| `wait_for_network_idle(timeout)` | Network-based | ✅ Implemented | Only tracks Fetch/XHR (see limitations below) |
| `wait_for_selector(selector, state)` | Selector-based | ✅ **IMPLEMENTED 2025-12-30** | 4 states: visible, hidden, attached, detached |
| `wait_for_function(js_expression)` | Function-based | ❌ Missing | Medium priority (P4.1) |

### Network Idle Limitations (Critical)

Our `wait_for_network_idle()` implementation has fundamental limitations that must be documented:

**What we track:**
- ✅ Fetch API requests
- ✅ XMLHttpRequest (XHR)

**What we DON'T track (premature idle detection risk):**
- ❌ Image loading (`<img>` tags, `background-image` CSS)
- ❌ Stylesheet loading (`<link rel="stylesheet">`)
- ❌ Font loading (`@font-face`)
- ❌ Script loading (`<script>` tags)
- ❌ WebSocket connections (persistent, never "idle")
- ❌ Service Worker fetch events (transparent interception)

**Sites that will NEVER reach idle:**
- Real-time dashboards (polling)
- Chat applications (WebSockets)
- Trading platforms (live data feeds)
- Collaborative editors (sync heartbeats)

**Recommendation:** For these sites, use `wait_for_selector()` instead of `wait_for_network_idle()`.

### 500ms Idle Threshold Validation

Our 500ms idle threshold is **industry-standard**:
- Playwright: 500ms (networkidle)
- Puppeteer networkidle0: 500ms with 0 connections
- Puppeteer networkidle2: 500ms with ≤2 connections

**Do not change this value** - it balances sensitivity with timing jitter tolerance.

### Graduated Wait Duration Guidance

For AI agents using `sleep()`, long durations signal problems:

| Duration | Log Level | Interpretation |
|----------|-----------|----------------|
| < 100ms | None | Routine transient delay |
| 100ms - 1000ms | INFO | Expected animation/debounce |
| 1000ms - 10000ms | WARNING | Suspicious - consider condition-based wait |
| > 10000ms | ERROR | Almost certainly wrong - missing sync logic |
| > 300000ms | REJECT | Hard limit (5 minutes) |

---

## Critical Finding: Hover Implementation Superiority

### Why Our Hover Works Better Than CiC

Our ActionChains-based hover **genuinely triggers CSS `:hover` states**, while Claude-in-Chrome's CDP-based hover does not. This was validated through testing:

| Test | SMCP | CiC |
|------|:----:|:---:|
| `element.matches(':hover')` | `true` | `false` |
| Hover state persists across tool calls | Yes | No |
| CSS background-color change observed | Yes | No |

**Root Cause:** ActionChains maintains internal mouse state within the WebDriver-controlled browser. The cursor position persists until another action moves it. CiC likely uses CDP `Input.dispatchMouseEvent` which fires a one-time event without establishing persistent cursor position.

**Practical Impact:** SMCP can test dropdown menus, tooltips, and hover-triggered UI. CiC cannot reliably do this.

---

## Priority 1: Critical Enhancements (Do Now)

### ~~1.1 Add Scroll-Into-View Before Hover~~ ✅ DONE

Implemented at lines 1724-1729.

### ~~1.2 Add Visibility Check Before Hover~~ ✅ DONE

Implemented at lines 1731-1734.

### ~~1.3 Add Duration Validation to Sleep~~ ✅ DONE (2025-12-30)

Implemented in `sleep()` function with:
- Negative duration validation
- 5-minute max (300000ms) validation
- Graduated warnings for durations > 10000ms

### ~~1.4 Add Dimension Validation to Resize Window~~ ✅ DONE (2025-12-30)

Implemented with KISS principle:
- Positive integer validation only (`width > 0`, `height > 0`)
- No arbitrary max limits - browser/OS handles clamping naturally
- Returns actual achieved size (already documents OS constraints)

---

## Priority 2: Robustness Enhancements

### 2.1 Fix Resize Window to Use Viewport Size (Not Window Size)

**Problem:** `set_window_size()` sets the ENTIRE browser window including chrome (address bar, tabs, borders). Viewport is smaller by 50-200px. For responsive testing, viewport size is what matters.

**File:** `server.py`
**Location:** Rewrite `resize_window()` function

**Implementation:**
```python
async def resize_window(width: int, height: int, ctx: Context) -> dict:
    """Resize the browser viewport to specified dimensions."""
    logger = PrintLogger(ctx)
    driver = await service.get_browser()

    # Validation: positive integers only (KISS - no arbitrary max limits)
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive integers. Got: {width}x{height}")

    await logger.info(f"Resizing viewport to {width}x{height}")

    # Get current window and viewport sizes
    current_window = await asyncio.to_thread(driver.get_window_size)
    current_viewport = await asyncio.to_thread(
        driver.execute_script,
        "return {width: window.innerWidth, height: window.innerHeight};"
    )

    # Calculate chrome overhead
    chrome_width = current_window['width'] - current_viewport['width']
    chrome_height = current_window['height'] - current_viewport['height']

    # Set window size to achieve target viewport
    target_window_width = width + chrome_width
    target_window_height = height + chrome_height

    await asyncio.to_thread(driver.set_window_size, target_window_width, target_window_height)

    # Verify actual viewport
    actual_viewport = await asyncio.to_thread(
        driver.execute_script,
        "return {width: window.innerWidth, height: window.innerHeight};"
    )

    if abs(actual_viewport['width'] - width) > 5 or abs(actual_viewport['height'] - height) > 5:
        await logger.info(
            f"Warning: Requested {width}x{height}, got {actual_viewport['width']}x{actual_viewport['height']}"
        )

    return {"width": actual_viewport['width'], "height": actual_viewport['height']}
```

**Note:** This changes the semantics from window-size to viewport-size. Update docstring accordingly.

### 2.2 Add Timestamp ISO Format to Console Logs

**Problem:** Console log timestamps are epoch milliseconds. Not human-readable. AI agents correlating logs with actions would benefit from ISO strings.

**File:** `src/models.py`
**Location:** `ConsoleLogEntry` class (line ~638)

**Add field:**
```python
class ConsoleLogEntry(BaseModel):
    level: str
    message: str
    source: str
    timestamp: int
    timestamp_iso: str  # NEW: ISO formatted timestamp
```

**File:** `server.py`
**Location:** Inside `get_console_logs()` where ConsoleLogEntry is created (line ~2414)

**Implementation:**
```python
from datetime import datetime

# When creating ConsoleLogEntry:
entries.append(ConsoleLogEntry(
    level=level,
    message=message,
    source=source,
    timestamp=timestamp,
    timestamp_iso=datetime.fromtimestamp(timestamp / 1000).isoformat(),
))
```

### 2.3 Add Regex Validation to Console Logs Pattern

**Problem:** Invalid regex pattern will raise cryptic exception. Should catch and return user-friendly error.

**File:** `server.py`
**Location:** Inside `get_console_logs()` where pattern is compiled (line ~2390)

**Implementation:**
```python
# Replace current pattern compilation:
if pattern:
    try:
        pattern_regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
else:
    pattern_regex = None
```

### 2.4 Add Viewport Presets Parameter to Resize Window

**Problem:** AI agents must remember common device dimensions. Presets improve discoverability.

**File:** `server.py`
**Location:** Add constant before `resize_window()` function, modify signature

**Implementation:**
```python
VIEWPORT_PRESETS: dict[str, tuple[int, int]] = {
    "mobile-iphone-se": (375, 667),
    "mobile-iphone-11": (414, 896),
    "mobile-iphone-14-pro": (393, 852),
    "tablet-ipad": (768, 1024),
    "tablet-ipad-pro": (1024, 1366),
    "desktop-laptop": (1366, 768),
    "desktop-1080p": (1920, 1080),
    "desktop-1440p": (2560, 1440),
}

async def resize_window(
    width: int | None = None,
    height: int | None = None,
    preset: str | None = None,
    ctx: Context,
) -> dict:
    """Resize the browser viewport to specified dimensions.

    Provide either width+height OR preset, not both.

    Presets: mobile-iphone-se, mobile-iphone-11, mobile-iphone-14-pro,
             tablet-ipad, tablet-ipad-pro, desktop-laptop, desktop-1080p, desktop-1440p
    """
    if preset and (width is not None or height is not None):
        raise ValueError("Cannot specify both preset and explicit dimensions")

    if preset:
        if preset not in VIEWPORT_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Valid: {', '.join(VIEWPORT_PRESETS.keys())}")
        width, height = VIEWPORT_PRESETS[preset]

    if width is None or height is None:
        raise ValueError("Must specify either preset or both width and height")

    # ... rest of implementation
```

---

## Priority 3: Future Enhancements

### 3.1 Shadow DOM Hover Support

**Problem:** CSS selectors don't traverse shadow DOM. Elements inside web components are invisible to our hover.

**Complexity:** High - requires detecting shadow boundaries, executing JS to traverse shadow roots.

**Workaround:** User can use `execute_javascript` to find shadow DOM elements and return their coordinates, then use a hypothetical coordinate-based hover.

**Implementation Approach (when needed):**
1. Detect if selector contains shadow DOM path indicators (e.g., `::shadow`, custom separator)
2. If so, use JavaScript to traverse shadow roots and find element
3. Get element's bounding rect via JS
4. Use ActionChains to move to those coordinates

### 3.2 iframe Console Logs Support

**Problem:** `driver.get_log("browser")` only captures main frame logs. JavaScript errors in iframes are invisible.

**Workaround:** Use `execute_javascript` inside each iframe to capture logs.

**Implementation Approach (when needed):**
1. Get all iframes on page
2. For each iframe (same-origin only), switch context and inject console interceptor
3. Aggregate logs from all frames

### 3.3 Mobile/Touch Device Detection for Hover

**Problem:** Hover is meaningless on touch devices. If browser is in mobile emulation mode, hover should either warn or use tap instead.

**Implementation Approach:**
```python
is_touch = await asyncio.to_thread(
    driver.execute_script,
    "return 'ontouchstart' in window || navigator.maxTouchPoints > 0"
)
if is_touch:
    await logger.info("Warning: Hover on touch device may not work as expected")
```

### 3.4 Headless Mode Resize Warning

**Problem:** `set_window_size()` may not work correctly in headless Chrome. Should detect and warn.

**Implementation Approach:**
```python
# In resize_window():
is_headless = await asyncio.to_thread(
    driver.execute_script,
    "return navigator.userAgent.includes('HeadlessChrome')"
)
if is_headless:
    await logger.info("Warning: Resize in headless mode may be unreliable")
```

### 3.5 Add `since_timestamp` Filter to Console Logs

**Problem:** After multiple interactions, AI agent may only want logs since their last check. Currently must manually filter returned logs.

**Implementation:**
```python
async def get_console_logs(
    ctx: Context,
    level_filter: Literal["ALL", "SEVERE", "WARNING", "INFO"] | None = None,
    pattern: str | None = None,
    since_timestamp: int | None = None,  # NEW
) -> ConsoleLogsResult:
    # ...
    # In the filtering loop:
    if since_timestamp and timestamp < since_timestamp:
        continue
```

---

## Priority 4: Wait Tool Enhancements

### 4.1 Add `wait_for_function()` Tool

**Problem:** Complex application state (Redux store, React render, custom conditions) cannot be checked with selector-based or network-based waits.

**Use Case:** Wait for `window.APP_READY === true` or `document.querySelector('.loading').classList.contains('hidden')`.

**Implementation:**
```python
@mcp.tool(
    annotations=ToolAnnotations(
        title="Wait for Function",
        readOnlyHint=True,
    )
)
async def wait_for_function(
    expression: str,
    timeout: int = 30000,
    polling_interval: int = 100,
    ctx: Context | None = None,
) -> dict:
    """Wait for a JavaScript expression to return a truthy value.

    Args:
        expression: JavaScript expression to evaluate (must return truthy when ready)
        timeout: Maximum wait duration in milliseconds (default 30000)
        polling_interval: How often to check the expression in ms (default 100)

    Returns:
        Dict with success status, elapsed_ms, and final expression value

    Examples:
        - wait_for_function("window.APP_READY === true")
        - wait_for_function("document.querySelectorAll('.item').length >= 10")
        - wait_for_function("!document.querySelector('.loading')")

    Raises:
        TimeoutError: if expression doesn't become truthy within timeout
    """
    logger = PrintLogger(ctx) if ctx else None
    driver = await service.get_browser()

    if timeout < 0 or timeout > 300000:
        raise ValueError("timeout must be between 0 and 300000ms")
    if polling_interval < 10 or polling_interval > 5000:
        raise ValueError("polling_interval must be between 10 and 5000ms")

    start_time = time.time()
    timeout_s = timeout / 1000

    while time.time() - start_time < timeout_s:
        try:
            result = await asyncio.to_thread(driver.execute_script, f"return !!({expression})")
            if result:
                elapsed_ms = int((time.time() - start_time) * 1000)
                if logger:
                    await logger.info(f"Function condition met after {elapsed_ms}ms")
                return {"success": True, "elapsed_ms": elapsed_ms}
        except Exception as e:
            # Expression threw - keep polling
            pass

        await asyncio.sleep(polling_interval / 1000)

    elapsed_ms = int((time.time() - start_time) * 1000)
    raise TimeoutError(
        f"wait_for_function timed out after {elapsed_ms}ms. "
        f"Expression '{expression}' never returned truthy. "
        "Consider increasing timeout or verifying the expression is correct."
    )
```

### 4.2 Expand Network Monitor to Track Resource Loading

**Problem:** Current network monitor only tracks Fetch/XHR. Images, CSS, fonts, and scripts load independently.

**Complexity:** Medium - requires `PerformanceObserver` API integration.

**Implementation Approach:**
```javascript
// Enhanced network_monitor_setup.js
window.__networkMonitor = {
    fetchRequests: 0,
    xhrRequests: 0,
    resourceLoading: 0,
    lastActivityTime: null,

    recordActivity() {
        this.lastActivityTime = Date.now();
    },

    isIdle() {
        return this.fetchRequests === 0 &&
               this.xhrRequests === 0 &&
               this.resourceLoading === 0;
    }
};

// Track resource loading via PerformanceObserver
const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        if (entry.initiatorType !== 'fetch' && entry.initiatorType !== 'xmlhttprequest') {
            window.__networkMonitor.recordActivity();
        }
    }
});
observer.observe({ entryTypes: ['resource'] });
```

### 4.3 Add Load State Options to Navigation

**Problem:** `wait_for_network_idle()` is not always the right choice. Sometimes `domcontentloaded` or `load` is sufficient and faster.

**Use Case:** For static pages, waiting for `domcontentloaded` is faster than `networkidle`. For pages with heavy analytics, `load` is faster than waiting for tracking pixels.

**Implementation Approach:**
Add `wait_until` parameter to `navigate()`:
```python
async def navigate(
    url: str,
    wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = "load",
    ...
)
```

**Playwright load states:**
| State | Description | Use When |
|-------|-------------|----------|
| `commit` | Response received, document loading | Fastest, for immediate JS execution |
| `domcontentloaded` | DOM parsed, not all resources loaded | Static content interaction |
| `load` | Full page with images/scripts | Default, most reliable |
| `networkidle` | No network for 500ms | SPAs, AJAX-heavy pages |

---

## AI Agent Guidance (Perplexity Research 2025-12-30)

### When to Use Each Wait Tool

**Use `sleep(duration_ms)` when:**
- You know the exact duration required (CSS animation is 300ms)
- Testing timeout-dependent behavior
- Waiting for debounce timers to fire
- Creating intentional delays between actions

**Avoid `sleep()` when:**
- Content loading time is unknown
- Waiting for network responses
- Waiting for dynamic content to appear
- You don't know why you're waiting (red flag!)

**Use `wait_for_network_idle()` when:**
- You've triggered an action that loads network content
- The page has completed all data fetching
- Testing traditional multi-page applications
- Network activity is a reliable indicator of readiness

**Avoid `wait_for_network_idle()` when:**
- The application uses WebSockets or persistent connections
- The application implements polling (real-time dashboards)
- Lazy-loading is used extensively
- You need to interact with specific elements (use `wait_for_selector` instead)

**Use `wait_for_selector()` when:**
- You can specify the exact element you need
- UI state is the reliable indicator of readiness
- Testing modern SPAs with dynamic content
- You need to verify that content is visible

**Recover from wait timeout by:**
1. Verify that your wait criteria are correct
2. Check if the application uses unsupported transport (WebSockets, polling)
3. Switch to a different wait strategy if appropriate
4. If all waits fail, page state may be unrecoverable; consider reloading

### Error Message Design for AI Agents

Error messages should enable self-correction:

```python
# ❌ Bad: Generic message
raise TimeoutError("Timeout after 30000ms")

# ✅ Good: Actionable guidance
raise TimeoutError(
    "wait_for_selector('#modal') timed out after 30000ms. "
    "Element never became visible. Possible causes: "
    "(1) Selector is incorrect, (2) Element is in iframe, "
    "(3) Element requires user action to appear. "
    "Try: verify selector with get_page_html(), check for iframes."
)
```

### Return Objects Should Enable Reasoning

```python
# ❌ Bad: Boolean only
return {"success": True}

# ✅ Good: Context for reasoning
return {
    "success": True,
    "elapsed_ms": 1250,
    "strategy": "network_idle",
    "final_state": {
        "active_requests": 0,
        "idle_duration_ms": 500
    }
}

# ✅ Good: Timeout with diagnostic info
return {
    "success": False,
    "reason": "timeout",
    "elapsed_ms": 30000,
    "final_state": {
        "active_requests": 3,
        "pending_urls": ["api.example.com/poll", "analytics.example.com/beacon"]
    },
    "suggestion": "Site has active polling. Use wait_for_selector() instead."
}
```

---

## Implementation Roadmap

### Phase 1: MVP (Current) ✅
- `sleep()` - deterministic timing (renamed from `wait()` 2025-12-30)
- `wait_for_network_idle()` - condition-based network monitoring

### Phase 2: Near-term ✅ COMPLETE
- ~~`wait_for_selector(selector, state, timeout)`~~ **IMPLEMENTED 2025-12-30**
- ~~Add duration validation to `sleep()`~~ **IMPLEMENTED 2025-12-30**
- ~~Add dimension validation to `resize_window()`~~ **IMPLEMENTED 2025-12-30** (positive integers only, KISS)
- ~~Add graduated timeout warnings for `sleep()`~~ **IMPLEMENTED 2025-12-30**

### Phase 3: Medium-term
- `wait_for_function(js_expression, timeout)` (P4.1)
- Expand network monitoring to track images/CSS/scripts (P4.2)
- Add load state options to `navigate()` (P4.3)
- Enhanced return objects with detailed status

### Phase 4: Long-term
- Migrate to DevTools Protocol for native network monitoring
- WebSocket/Service Worker awareness
- URL pattern exclusion for polling sites
- Adaptive polling intervals

---

## Naming Convention Decision

**Current State (2025-12-30):**
- Function: `hover` / Title: "Hover Over Element"
- Function: `sleep` / Title: "Sleep" (renamed from `wait` - Perplexity research recommended)
- Function: `resize_window` / Title: "Resize Browser Window"
- Function: `get_console_logs` / Title: "Get Console Logs"
- Function: `wait_for_selector` / Title: "Wait for Selector"
- Function: `wait_for_network_idle` / Title: "Wait for Network Idle"

**Perplexity Research (2025-12-30):** Renamed `wait` to `sleep` because:
1. Industry standard (Python, JavaScript, Go all use `sleep`)
2. Clearly signals NOT a condition-based wait (unlike `wait_for_*` pattern)
3. Subtly hints this is not the preferred approach for modern automation
4. `wait_for_*` pattern reserved for condition-based tools

**Pattern established:**
- `sleep(duration_ms)` - deterministic, time-based
- `wait_for_*(...)` - condition-based (network_idle, selector, function)

If we ever add more action verbs (tap, swipe, pinch), consider consistent naming then.

---

## Edge Cases to Document

These are NOT bugs to fix, but limitations to document clearly:

### Hover Edge Cases
1. **Hover on elements that move:** If CSS `:hover` triggers a transform that moves the element, hover state may deactivate. Workaround: disable animations before hover.

2. **Cross-origin iframes:** Cannot hover elements in cross-origin iframes (browser security).

### Console Log Edge Cases
3. **Console log buffer overflow:** Long sessions may lose older logs. Workaround: retrieve logs frequently.

4. **iframe console logs invisible:** `driver.get_log("browser")` only captures main frame logs.

### Resize Edge Cases
5. **Headless resize:** Viewport sizing may be unreliable in headless mode.

### Wait Tool Edge Cases (Perplexity Research 2025-12-30)

6. **Network idle on polling sites:** Sites with constant polling (analytics heartbeats, real-time dashboards) will **never** reach network idle. The `wait_for_network_idle()` tool will always timeout. **Workaround:** Use `wait_for_selector()` to wait for specific UI state instead.

7. **Network idle on WebSocket sites:** Chat applications, trading platforms, and collaborative tools maintain persistent WebSocket connections. These connections don't count as "network activity" in our Fetch/XHR instrumentation, but they indicate the application is active. **Workaround:** Use `wait_for_selector()` or `wait_for_function()`.

8. **Lazy-loaded content:** Infinite scroll interfaces and lazy-loaded images will trigger new network requests as the user scrolls. Network idle state is transient. **Workaround:** Wait for specific content elements, not network state.

9. **Race conditions in network idle:** If requests are queued rapidly, there's a narrow window where the monitor might report idle between batches. Our 50ms polling interval mitigates this but doesn't eliminate it. **Mitigation:** Use 500ms threshold (our current value) to reduce false positives.

10. **Service Worker interception:** Service Workers can intercept and handle requests transparently. Network timing becomes opaque to client-side monitors. **Impact:** Network idle may report early if Service Worker handles requests from cache.

11. **Image/CSS/font loading not tracked:** Our network monitor only instruments Fetch and XHR. Resource loading via `<img>`, `<link>`, `@font-face` is not tracked. **Impact:** Network idle may trigger before visual resources finish loading. **Future fix:** P4.2 adds PerformanceObserver tracking.

---

## Testing Checklist

### Hover Actionability Tests (2025-12-30) ✅ ALL PASSED

**Stability Check (8/8 tests):**
- [x] Static element - passes quickly (2 frames)
- [x] CSS transform animation - detects via getAnimations()
- [x] CSS margin animation - detects via position polling
- [x] Infinite animation - warns "Element has infinite animation" and proceeds
- [x] JS setInterval animation - position polling catches it (getAnimations=0)
- [x] JS RAF animation - position polling catches it (getAnimations=0)
- [x] Paused animation - passes immediately (playState='paused')
- [x] Distance threshold - 3px movement considered stable (<5px threshold)

**Visibility Check (2/2 tests):**
- [x] `display:none` - rejected with "Element is not visible - cannot hover"
- [x] `visibility:hidden` - rejected with "Element is not visible - cannot hover"

**Occlusion Check (2/2 tests):**
- [x] Modal overlay - rejected with "Element is obscured by another element"
- [x] pointer-events:none overlay - passes through to button underneath

**Duration Validation (tested earlier):**
- [x] Negative duration (-100) - rejected with "cannot be negative"
- [x] Excessive duration (50000) - rejected with "exceeds maximum"
- [x] Valid duration (0-30000ms) - works correctly

**Test Fixtures:** `examples/hover-stability.html`, `hover-visibility.html`, `hover-occlusion.html`

---

**Sleep (2025-12-30) ✅ VERIFIED:**
- [x] `sleep(-100)` → ValueError: "duration_ms cannot be negative"
- [x] `sleep(400000)` → ValueError with guidance to use wait_for_selector()
- [x] `sleep(15000)` completes, returns `{slept_ms: 15000}` (warning logged server-side)
- [x] Renamed from `wait()` to `sleep()` per Perplexity research

**Console Logs (2025-12-30) ✅ VERIFIED:**
- [x] Basic: Returns logs with level breakdown (severe_count, warning_count, info_count)
- [x] Level filter: `level_filter="SEVERE"` returns only error-level logs
- [x] Pattern filter: `pattern="BETA"` returns only matching messages
- [x] Logs cleared after retrieval (Chrome logging behavior)

**Resize Window (2025-12-30) ✅ VERIFIED:**
- [x] `resize_window(800, 600)` → returns `{width: 800, height: 600}`
- [x] `resize_window(375, 667)` → clamped to OS minimum (500x667)
- [x] `resize_window(0, 600)` → ValueError: "must be positive integers"
- [x] `resize_window(-100, 600)` → ValueError: "must be positive integers"
- [x] No max limits - browser/OS handles clamping (KISS principle)

**Wait for Network Idle (2025-12-30) ✅ VERIFIED:**
- [x] Static page → returns immediately (no pending requests)
- [x] After fetch → waits for completion then returns

---

## Files to Modify

| File | Changes |
|------|---------|
| `server.py` | P1 validation, P2 viewport logic |
| `src/models.py` | Add `timestamp_iso` field to ConsoleLogEntry |
| `README.md` | Document viewport vs window, presets, limitations |

---

## Reference: Perplexity Research Citations

### Hover Research (2025-12-29)
- Playwright actionability checks: playwright.dev/docs/actionability
- ActionChains persistent state: browserstack.com/guide/mouse-hover-in-selenium
- Viewport vs window size: browserstack.com/guide/selenium-window-size
- Chrome console logging: developer.chrome.com/docs/chromedriver/logging

### Wait Strategy Research (2025-12-30)

**Framework Documentation:**
- [Playwright waitForLoadState](https://www.browserstack.com/guide/playwright-waitforloadstate)
- [Puppeteer waitUntil options](https://www.browserless.io/blog/waituntil-option-for-puppeteer-and-playwright)
- [Selenium wait commands](https://www.browserstack.com/guide/wait-commands-in-selenium-webdriver)
- [Cypress waiting best practices](https://filiphric.com/waiting-in-cypress-and-how-to-avoid-it)
- [Playwright waitForSelector](https://autify.com/blog/playwright-waitforselector)
- [Puppeteer waitForFunction](https://latenode.com/blog/web-automation-scraping/puppeteer-fundamentals-setup/custom-wait-conditions-with-waitforfunction-in-puppeteer)

**Network Idle Implementation:**
- Playwright/Puppeteer networkidle: 500ms with 0 connections (industry standard)
- Puppeteer networkidle2: 500ms with ≤2 connections (for sites with background activity)
- [CloudLayer waitUntil analysis](https://cloudlayer.io/blog/puppeteer-waituntil-options/)

**AI Agent Design:**
- [API error messages for AI agents](https://nordicapis.com/designing-api-error-messages-for-ai-agents/)
- [MCP timeout and retry patterns](https://octopus.com/blog/mcp-timeout-retry)
- [Agent builder best practices](https://www.uipath.com/blog/ai/agent-builder-best-practices)

**Testing Anti-Patterns:**
- [Test automation anti-patterns](https://www.testdevlab.com/blog/5-test-automation-anti-patterns-and-how-to-avoid-them)
- [Playwright waits and timeouts guide](https://checklyhq.com/docs/learn/playwright/waits-and-timeouts)
- [Playwright waitForTimeout discouraged](https://www.browserstack.com/guide/playwright-waitfortimeout)

**Key Findings:**
1. Our dual-wait architecture (deterministic + network-based) mirrors Playwright/Puppeteer patterns
2. 500ms idle threshold is industry-standard (validated)
3. Critical gap identified: `wait_for_selector()` essential for SPAs
4. Network idle limitations documented (Fetch/XHR only, no WebSocket/Service Worker)
5. AI agent error messages should enable self-correction with actionable guidance