# Hover Tool Actionability Test Suite

Comprehensive integration tests for the hover tool's actionability checks.

## Overview

The hover tool performs 5 actionability checks before executing a hover operation:

| Check | Purpose | Error Message |
|-------|---------|---------------|
| **Duration validation** | Validate duration_ms parameter (0-30000ms) | "duration_ms cannot be negative" / "exceeds maximum" |
| **Scroll into view** | Bring element into viewport | - |
| **Visibility check** | Verify element.is_displayed() | "Element is not visible - cannot hover" |
| **Stability check** | Multi-signal detection of animations | Warning: "Element has infinite animation" |
| **Pointer events check** | Verify elementFromPoint returns target | "Element is obscured by another element" |

## Setup

### Start HTTP Server

```bash
cd /Users/chris/claude-workspace/mcp/selenium-browser-automation/examples
uv run python -m http.server 8888
```

Test fixtures are then available at `http://localhost:8888/`

### Fixtures

| Fixture | Tests | Location |
|---------|-------|----------|
| hover-stability.html | Stability detection (10 scenarios) | `http://localhost:8888/hover-stability.html` |
| hover-visibility.html | Visibility check (8 scenarios) | `http://localhost:8888/hover-visibility.html` |
| hover-occlusion.html | Occlusion detection (8 scenarios) | `http://localhost:8888/hover-occlusion.html` |

---

## Test 1: Static Element (Baseline)

**Goal:** Verify hover succeeds on a static, visible element with no animations.

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Hover on static button:
```
hover("#static-btn")
```

### Verify

- ✅ Hover succeeds (no exception)
- ✅ Button turns green (CSS :hover state applied)
- ✅ MCP logs show quick stability check (2 frames)

### Validate with JavaScript

```
execute_javascript("window.validateHover('static-btn')")
```

Expected result:
```json
{
  "buttonId": "static-btn",
  "isHovered": true,
  "hoverDetected": true,
  "backgroundColor": "rgb(76, 175, 80)"
}
```

---

## Test 2: CSS Transform Animation Detection

**Goal:** Verify stability check detects CSS transform animations via getAnimations() API.

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Check animation info before hover:
```
execute_javascript("window.getAnimationInfo('transform-btn')")
```

3. Hover on animated button:
```
hover("#transform-btn")
```

### Verify

- ✅ getAnimationInfo shows 1 running animation
- ✅ MCP logs show "Element has infinite animation - hover may be inconsistent"
- ✅ Hover proceeds after max stability checks
- ✅ Button receives hover event (turns green)

---

## Test 3: JavaScript Animation Detection

**Goal:** Verify position-based stability detection catches JS animations (not detected by getAnimations).

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Check animation info (should show 0 animations):
```
execute_javascript("window.getAnimationInfo('setinterval-btn')")
```

3. Hover on JS-animated button:
```
hover("#setinterval-btn")
```

### Verify

- ✅ getAnimationInfo shows 0 animations (JS animations not detected by API)
- ✅ MCP logs show position-based instability detection
- ✅ Hover proceeds after stability checks
- ✅ Button receives hover event

---

## Test 4: Infinite Animation Warning

**Goal:** Verify infinite animations are detected and warned about.

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Hover on infinite animation button:
```
hover("#infinite-btn")
```

### Verify

- ✅ MCP logs contain: "Warning: Element has infinite animation - hover may be inconsistent"
- ✅ Hover still proceeds (doesn't fail)
- ✅ Button receives hover event

---

## Test 5: Paused Animation (Should Pass Immediately)

**Goal:** Verify paused animations don't block hover.

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Hover on paused animation button:
```
hover("#paused-btn")
```

### Verify

- ✅ Hover succeeds quickly (animation exists but playState is 'paused')
- ✅ No stability warnings in logs
- ✅ Button turns green

---

## Test 6: Distance Threshold Test

**Goal:** Verify small movements (<5px) are considered stable.

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Hover on threshold test button (moves 3px):
```
hover("#threshold-btn")
```

### Verify

- ✅ Hover succeeds (3px movement is below 5px threshold)
- ✅ MCP logs show stability confirmed
- ✅ Button turns green

---

## Test 7: Visibility Check - display:none

**Goal:** Verify hidden elements are rejected with clear error.

### Steps

1. Navigate to visibility fixture:
```
navigate("http://localhost:8888/hover-visibility.html")
```

2. Attempt hover on hidden button:
```
hover("#display-none-btn")
```

### Verify

- ✅ Hover fails with exception
- ✅ Error message contains: "Element '#display-none-btn' is not visible - cannot hover"

---

## Test 8: Visibility Check - visibility:hidden

**Goal:** Verify visibility:hidden elements are rejected.

### Steps

1. Navigate to visibility fixture:
```
navigate("http://localhost:8888/hover-visibility.html")
```

2. Attempt hover on invisible button:
```
hover("#visibility-hidden-btn")
```

### Verify

- ✅ Hover fails with exception
- ✅ Error message contains: "not visible"

---

## Test 9: Occlusion Check - Modal Overlay

**Goal:** Verify obscured elements are rejected.

### Steps

1. Navigate to occlusion fixture:
```
navigate("http://localhost:8888/hover-occlusion.html")
```

2. Show modal overlay:
```
execute_javascript("showModal()")
```

3. Check occlusion state:
```
execute_javascript("window.checkOcclusion('modal-covered-btn')")
```

4. Attempt hover on covered button:
```
hover("#modal-covered-btn")
```

### Verify

- ✅ checkOcclusion shows `wouldPass: false`
- ✅ Hover fails with exception
- ✅ Error message contains: "Element '#modal-covered-btn' is obscured by another element at its center"

---

## Test 10: Occlusion Check - pointer-events:none Overlay

**Goal:** Verify overlays with pointer-events:none don't block hover.

### Steps

1. Navigate to occlusion fixture:
```
navigate("http://localhost:8888/hover-occlusion.html")
```

2. Check occlusion state:
```
execute_javascript("window.checkOcclusion('passthrough-overlay-btn')")
```

3. Hover on button with passthrough overlay:
```
hover("#passthrough-overlay-btn")
```

### Verify

- ✅ checkOcclusion shows `wouldPass: true` (pointer-events:none allows passthrough)
- ✅ Hover succeeds
- ✅ Button turns green

---

## Test 11: Nested Child Element

**Goal:** Verify elementFromPoint returning a child of target is acceptable.

### Steps

1. Navigate to occlusion fixture:
```
navigate("http://localhost:8888/hover-occlusion.html")
```

2. Check occlusion state (should return child span):
```
execute_javascript("window.checkOcclusion('nested-child-btn')")
```

3. Hover on button with nested child:
```
hover("#nested-child-btn")
```

### Verify

- ✅ checkOcclusion shows `isChildOfTarget: true`, `wouldPass: true`
- ✅ Hover succeeds (our check allows target.contains(atPoint))
- ✅ Button turns green

---

## Test 12: Duration Validation - Negative

**Goal:** Verify negative duration_ms is rejected.

### Steps

1. Navigate to any page:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Attempt hover with negative duration:
```
hover("#static-btn", duration_ms=-100)
```

### Verify

- ✅ Hover fails with exception
- ✅ Error message contains: "duration_ms cannot be negative"

---

## Test 13: Duration Validation - Excessive

**Goal:** Verify duration_ms > 30000 is rejected.

### Steps

1. Navigate to any page:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Attempt hover with excessive duration:
```
hover("#static-btn", duration_ms=50000)
```

### Verify

- ✅ Hover fails with exception
- ✅ Error message contains: "duration_ms exceeds maximum of 30000ms"

---

## Test 14: Valid Duration with Hold

**Goal:** Verify valid duration_ms causes hover to be held.

### Steps

1. Navigate to stability fixture:
```
navigate("http://localhost:8888/hover-stability.html")
```

2. Hover with 2 second hold:
```
hover("#static-btn", duration_ms=2000)
```

### Verify

- ✅ Hover succeeds
- ✅ Tool call takes ~2 seconds to complete
- ✅ Button remains green during hold

---

## Parity Testing: Selenium MCP vs Claude-in-Chrome

### Key Differences

| Aspect | Selenium MCP | Claude-in-Chrome |
|--------|--------------|------------------|
| **Hover mechanism** | ActionChains.move_to_element() | CDP Input.dispatchMouseEvent |
| **Cursor state** | Maintained across calls | One-time event |
| **CSS :hover persistence** | ✅ Persists | ❌ Lost between calls |
| **Stability check** | Multi-signal (RAF, getAnimations, position) | TBD |

### Testing Approach

For each test, run against both implementations and document:
1. Does hover succeed/fail as expected?
2. Are error messages similar?
3. Does CSS :hover state persist after hover returns?
4. Are stability warnings generated?

---

## Cleanup

Stop the HTTP server with Ctrl+C.

---

## Quick Reference: JavaScript Validation Helpers

```javascript
// Check hover state:
window.validateHover('button-id')

// Check animation info:
window.getAnimationInfo('button-id')

// Check visibility state:
window.checkVisibility('button-id')
window.checkAllVisibility()

// Check occlusion state:
window.checkOcclusion('button-id')
window.checkAllOcclusion()
```