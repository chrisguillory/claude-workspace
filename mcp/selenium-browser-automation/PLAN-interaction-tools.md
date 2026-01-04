# Selenium MCP: Roadmap & Research

## Purpose

This document tracks **future enhancements**, **research findings**, and **architectural decisions**.

- **Current tools**: See [README.md](README.md)
- **Implementation**: See [server.py](server.py)
- **Test fixtures**: See [examples/](examples/) and [tests/](tests/)

---

## Outstanding Gaps

### From Original Plan (Low Priority)

| Gap | Effort | Workaround | Notes |
|-----|--------|------------|-------|
| `scroll` | 10 min | `execute_javascript("window.scrollBy(0, 500)")` | Common operation, clean JS workaround exists |
| `upload_file` | 10 min | `element.send_keys(file_path)` | Specific use case, trivial to implement |
| `drag_and_drop` | 15 min | ActionChains already imported | Rare use case (Kanban boards, sortable lists) |

**Assessment**: The high-value items from our original plan are complete. These remaining gaps have workarounds and are low priority.

### CiC Parity Gaps (Not Currently Planned)

- `zoom` - Browser zoom control
- `gif_creator` - Record actions as GIF

---

## Future Enhancements

### Priority 2: Robustness

- [ ] **Viewport vs window size**: `set_window_size()` sets outer window; viewport is smaller by 50-200px due to browser chrome. For accurate responsive testing, calculate and compensate for chrome overhead.

- [ ] **ISO timestamps in console logs**: Currently epoch milliseconds. Adding `timestamp_iso` field would help AI agents correlate logs with actions.

- [ ] **Regex validation for console log pattern**: Invalid regex raises cryptic exception. Should catch `re.error` and return user-friendly message.

- [ ] **Viewport presets**: Named presets like `mobile-iphone-se`, `tablet-ipad`, `desktop-1080p` improve discoverability.

### Priority 3: Advanced Features

- [ ] **Shadow DOM hover support**: CSS selectors don't traverse shadow boundaries. Would require JS traversal + coordinate-based hover.

- [ ] **iframe console logs**: `driver.get_log("browser")` only captures main frame. Would need to switch context per iframe.

- [ ] **Mobile/touch detection**: Warn when hovering in mobile emulation mode where hover is meaningless.

- [ ] **Headless mode resize warning**: Viewport sizing may be unreliable in headless Chrome.

- [ ] **since_timestamp filter for console logs**: Allow filtering logs by timestamp to get only new entries.

### Priority 4: Wait Tool Expansion

- [ ] **wait_for_function(js_expression)**: For complex application state (Redux, React render, custom conditions). Completes the wait taxonomy.

- [ ] **Network monitor expansion**: Add PerformanceObserver to track images, CSS, fonts (currently only Fetch/XHR).

- [ ] **Load state options for navigate()**: Add `wait_until` parameter with options: `commit`, `domcontentloaded`, `load`, `networkidle`.

---

## Research Archive

### Wait Strategy Taxonomy (Perplexity Research 2025-12-30)

Modern browser automation frameworks use four wait types:

| Type | Purpose | When to Use | Our Tool |
|------|---------|-------------|----------|
| **Deterministic** | Fixed delay | Known timing (CSS animation, debounce) | `sleep()` |
| **Network-based** | Wait for idle | AJAX, dynamic content | `wait_for_network_idle()` |
| **Selector-based** | Wait for element | Specific UI state | `wait_for_selector()` |
| **Function-based** | Custom condition | Complex app state | Not implemented (P4) |

**Key insight**: Playwright explicitly discourages `waitForTimeout()` as an anti-pattern. Deterministic waits should be a minority of wait operations.

### Network Idle Limitations (Critical)

**What we track:**
- Fetch API requests
- XMLHttpRequest (XHR)

**What we DON'T track (premature idle risk):**
- Image/CSS/font loading
- WebSocket connections
- Service Worker fetch events
- Script tags

**Sites that NEVER reach idle:**
- Real-time dashboards (polling)
- Chat apps (WebSockets)
- Trading platforms (live feeds)
- Collaborative editors (sync heartbeats)

**Recommendation**: For these sites, use `wait_for_selector()` instead.

### 500ms Idle Threshold

Industry standard (Playwright, Puppeteer). Do not change - it balances sensitivity with timing jitter tolerance.

### Hover Implementation Design

Our ActionChains-based hover **genuinely triggers CSS `:hover` states**, unlike CDP-based approaches (Claude-in-Chrome). This enables testing dropdown menus, tooltips, and hover-triggered UI.

**Why ActionChains works and CDP doesn't:**

| Approach               | Mechanism                         | CSS :hover |    State Persistence    |
|------------------------|-----------------------------------|:----------:|:-----------------------:|
| ActionChains           | Maintains internal mouse position |   ✅ Yes    | ✅ Persists across calls |
| CDP dispatchMouseEvent | One-time event dispatch           |    ❌ No    |   ❌ Lost immediately    |

ActionChains maintains a virtual cursor position within the WebDriver-controlled browser. The cursor stays where you put it until another action moves it. CDP's `Input.dispatchMouseEvent` fires a single event without establishing persistent cursor state.

**Practical impact:** If you need to hover → wait → click a dropdown item, only ActionChains works. CDP hover would lose the dropdown before you can click.

Five actionability checks (Playwright-style):
1. Duration validation (0-30000ms)
2. Scroll into view
3. Visibility check
4. Stability check (animation-aware)
5. Pointer events check (occlusion detection)

### AI Agent Error Message Design

Error messages should enable self-correction:

```
# Bad: Generic
"Timeout after 30000ms"

# Good: Actionable
"wait_for_selector('#modal') timed out after 30000ms. Element never became visible.
Possible causes: (1) Selector incorrect, (2) Element in iframe, (3) Requires user action.
Try: verify selector with get_page_html(), check for iframes."
```

### Chrome Session Export (Perplexity Research 2025-01)

Captures session state from standalone Chrome for Selenium automation.

**Use case:** Log in manually (CAPTCHA, MFA) → export session → automate with authenticated state.

**Libraries evaluated:**

| Library | Purpose | Notes |
|---------|---------|-------|
| `browser-cookie3` | Cookie decryption | Handles macOS Keychain automatically |
| `ccl_chromium_reader` | LevelDB parser | Pure Python, no C compilation required |
| `plyvel` | LevelDB access | ❌ C compilation issues on macOS ARM64 |

**Chrome profile storage locations:**

| Storage Type | Path | Format |
|--------------|------|--------|
| Cookies | `{profile}/Cookies` | SQLite |
| localStorage | `{profile}/Local Storage/leveldb/` | LevelDB |
| sessionStorage | `{profile}/Session Storage/` | LevelDB |
| IndexedDB | `{profile}/IndexedDB/{origin}.indexeddb.leveldb/` | LevelDB |

**Critical path discovery:** localStorage is at `Local Storage/leveldb/`, NOT just `Local Storage/`. Tests found 0 origins without the subdirectory, 1,341 with correct path.

**Chrome epoch conversion:** `expires // 1_000_000 - 11644473600` (microseconds since 1601 → Unix seconds)

**sameSite SQLite values:** -1=unspecified, 0=None, 1=Lax, 2=Strict

**sessionStorage persistence:** Chrome persists sessionStorage to disk for crash recovery, contradicting browser specs but enabling useful automation workflows.

**IndexedDB schema limitation:** `ccl_chromium_reader` raises `NotImplementedError` for:
- `DatabaseMetadataType.IdbVersion`
- `ObjectStoreMetadataType.KeyPath`
- `ObjectStoreMetadataType.AutoIncrementFlag`

Records can be exported but schema cannot be recreated. For full IndexedDB support, use `save_profile_state()` from a Selenium session.

---

## Architectural Decisions

### `sleep()` instead of `wait()`

Renamed per Perplexity research:
- Industry standard (Python, JavaScript, Go)
- Distinguishes from `wait_for_*` condition-based tools
- Subtly signals this is not the preferred approach

### KISS Validation

Prefer simple validation over arbitrary limits:
- `resize_window`: Validates positive integers only, no magic max limits
- Browser/OS handles clamping naturally
- Returns actual achieved size

### Selector States

Four states matching Playwright semantics:
- `visible`: In DOM AND displayed
- `hidden`: Not visible OR not in DOM
- `attached`: Present in DOM (regardless of visibility)
- `detached`: Removed from DOM

---

## Known Edge Cases

These are **limitations to document**, not bugs to fix.

### Hover
1. **Moving elements**: If `:hover` triggers transform that moves element, hover may deactivate
2. **Cross-origin iframes**: Cannot hover elements in cross-origin iframes (browser security)

### Console Logs
3. **Buffer overflow**: Long sessions may lose older logs
4. **iframe logs invisible**: `driver.get_log("browser")` only captures main frame

### Resize
5. **Headless mode**: Viewport sizing may be unreliable

### Wait Tools
6. **Polling sites**: Never reach network idle (use `wait_for_selector()`)
7. **WebSocket sites**: Persistent connections don't trigger idle
8. **Lazy-loaded content**: Network idle is transient
9. **Race conditions**: Narrow window between request batches (mitigated by 500ms threshold)
10. **Service Workers**: Can intercept requests transparently
11. **Resource loading**: Images/CSS/fonts not tracked by network monitor

### Chrome Session Export
12. **macOS only**: Not tested on Windows/Linux
13. **Keychain prompt**: First run requires clicking "Always Allow" for cookie decryption
14. **IndexedDB records only**: Schema (version, keyPath, indexes) cannot be extracted from profile files
15. **Multi-profile cookie limitation**: `browser-cookie3` reads from default Chrome location regardless of profile_name parameter
16. **Binary decode warnings**: Some localStorage values (Slack compression) show decode errors but continue
17. **Race conditions**: Exporting while Chrome is actively writing may capture inconsistent state

---

## Development Workflow

### Testing Protocol

**Always test via MCP tools, not Python scripts:**
```python
# ✅ Correct: Test through MCP interface
mcp__selenium-browser-automation__sleep(duration_ms=-100)
# → Verifies full stack including validation

# ❌ Wrong: Python unit test
# Wouldn't catch integration issues
```

**MCP server reload after changes:**
- Changes to `server.py` require MCP reconnect
- `navigate(fresh_browser=True)` only restarts browser, not MCP server
- Use `/mcp reconnect selenium-browser-automation` after code changes

**Test fixture HTTP server:**
```bash
cd examples && uv run python -m http.server 8888
# Test files at http://localhost:8888/
```
Chrome can't access `file://` URLs in some security contexts.

### Fixture Philosophy

Each test fixture includes JavaScript validation helpers:
```javascript
window.validateHover('button-id')     // Check hover state
window.getAnimationInfo('button-id')  // Check animations
window.checkOcclusion('button-id')    // Check if obscured
```

This enables programmatic verification without visual inspection. The fixture provides both test scenarios AND validation functions.

---

## Implementation Notes (Non-Obvious Details)

### Hover Stability Detection Algorithm

We use THREE signals for stability detection:

| Signal | Catches | Misses |
|--------|---------|--------|
| `getAnimations()` | CSS animations/transitions | JS animations (setInterval, RAF) |
| RAF timing | Frame-to-frame jitter | Nothing (but adds latency) |
| Position polling (5px) | All movement | Nothing (catches everything) |

**Why all three?**
- `getAnimations()` alone misses JS animations
- Position polling alone is slow (needs multiple frames)
- Combining gives fast detection for CSS + fallback for JS

**Edge cases:**
- Paused animations (`playState='paused'`) pass immediately
- Infinite animations warn but don't block (proceed after max checks)
- 5px threshold chosen because smaller movements are likely jitter

### wait_for_selector Polling

- We use 50ms polling interval
- Playwright uses 100ms for `waitForSelector`
- Our lower interval = faster detection, slightly more CPU
- Tradeoff acceptable for automation use case

### sleep() Thresholds

| Duration | Behavior | Rationale |
|----------|----------|-----------|
| 0-10000ms | Silent | Normal for animations, debounce |
| >10000ms | Warning | Suspicious - probably wrong approach |
| >300000ms | Reject | 5 minutes is definitely wrong |

These came from Perplexity research on "when is a fixed wait suspicious vs definitely wrong."

### resize_window KISS Decision

We initially implemented 7680x4320 (8K) max limit. User pushed back:
- "Where are these magic numbers from?"
- No authoritative source
- Browser/OS already handles clamping

**Lesson**: Only validate what actually matters (positive integers). Let platform handle edge cases.

---

## Reference Citations

### Framework Documentation
- [Playwright waitForLoadState](https://browserstack.com/guide/playwright-waitforloadstate)
- [Puppeteer waitUntil options](https://browserless.io/blog/waituntil-option-for-puppeteer-and-playwright)
- [Selenium wait commands](https://browserstack.com/guide/wait-commands-in-selenium-webdriver)
- [Playwright waitForSelector](https://autify.com/blog/playwright-waitforselector)

### Implementation References
- [Playwright actionability checks](https://playwright.dev/docs/actionability)
- [ActionChains persistent state](https://browserstack.com/guide/mouse-hover-in-selenium)
- [Chrome console logging](https://developer.chrome.com/docs/chromedriver/logging)

### AI Agent Design
- [API error messages for AI agents](https://nordicapis.com/designing-api-error-messages-for-ai-agents/)