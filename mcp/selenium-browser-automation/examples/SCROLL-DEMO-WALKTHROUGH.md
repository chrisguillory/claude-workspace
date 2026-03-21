# Scroll Demo Walkthrough Script

Interactive headed-browser walkthrough of `scroll-demo.html`, showcasing all scroll
tool capabilities including `behavior: Literal['instant', 'smooth']` and
`position: Literal['top', 'bottom', 'left', 'right']` parameters.

**Prerequisites:** `/mcp reconnect selenium-browser-automation` after any code changes.

---

## Phase 1: Setup + Viewport Basics (Scenarios 1-2)

### Step 1.1: Navigate and screenshot
```
navigate(url='file:///...examples/scroll-demo.html')
screenshot('scroll-demo-start.png')
```
Observe: Header shows `scrollY: 0 / {pageHeight} (0%)`. Scenarios 1 and 2 visible.

### Step 1.2: Viewport scroll down (instant, default)
```
scroll(direction='down')
```
Observe: Page teleports 300px down. Header updates instantly.
Check: `scroll_y: ~300`, `scrolled: true`, `page_height`, `viewport_height`.

### Step 1.3: Larger scroll
```
scroll(direction='down', scroll_amount=5)
```
Observe: Jumps 500px further. Header shows ~800px position.

### Step 1.4: Boundary detection — scroll up at top
```
scroll(direction='up', scroll_amount=20)  # back to top
scroll(direction='up')                    # try to go further
```
Check: Second call returns `scrolled: false`, `scroll_y: 0`. Agent knows to stop.

---

## Phase 2: Instant vs Smooth Comparison (Scenario 3)

### Step 2.1: Scroll to the comparison scenario
```
scroll(css_selector='[data-scenario="3"]')
```

### Step 2.2: Instant viewport scroll
```
scroll(direction='down', scroll_amount=5, behavior='instant')
```
Observe: Page TELEPORTS past the colored bands. No animation visible.

### Step 2.3: Scroll back up
```
scroll(direction='up', scroll_amount=5)
```

### Step 2.4: Smooth viewport scroll
```
scroll(direction='down', scroll_amount=5, behavior='smooth')
```
Observe: Page ANIMATES smoothly past the bands (~300-500ms). The movement is visible.
Check: Return value has same accuracy — `scroll_y` reflects final position after animation.
Explain: Both produce identical return values. The visual experience is the only difference.

### Step 2.5: Screenshot to capture the bands
```
screenshot('scroll-demo-smooth-bands.png')
```

---

## Phase 3: Container Scroll — Chat Log (Scenario 4)

### Step 3.1: Navigate to chat log
```
scroll(css_selector='[data-scenario="4"]')
```

### Step 3.2: Instant container scroll
```
scroll(direction='down', css_selector='#chat-log')
```
Observe: Chat messages jump within the 200px container. Page doesn't move.
Check: `container_scroll_top: ~300`, `scrolled: true`.
Explain: CSS `scroll-behavior: smooth` is set on this container. Default `behavior='instant'` overrides it.

### Step 3.3: Smooth container scroll
```
scroll(direction='down', css_selector='#chat-log', behavior='smooth')
```
Observe: Chat messages animate smoothly within the container. Visually distinct from instant.
Check: Same return shape, `scrolled: true`.
Explain: `behavior='smooth'` respects the CSS. Tool waits for `scrollend` on the container element.

### Step 3.4: Jump to container bottom with position
```
scroll(position='bottom', css_selector='#chat-log')
```
Observe: Chat jumps to the very last message in ONE call (no repeated scrolling needed).
Check: `mode: 'container_scroll_to'`, `scrolled: true`.
Explain: `position='bottom'` uses `element.scrollTo({top: scrollHeight})`. This is what motivated the `position` parameter — the old approach required multiple `scroll(direction='down')` calls.

### Step 3.5: Verify container boundary
```
scroll(position='bottom', css_selector='#chat-log')
```
Check: `scrolled: false` — already at bottom. Agent knows to stop.

### Step 3.6: Smooth position scroll back to top
```
scroll(position='top', css_selector='#chat-log', behavior='smooth')
```
Observe: Chat smoothly animates back to the first message.

---

## Phase 4: Horizontal Scroll — Code Block (Scenario 5)

### Step 4.1: Navigate
```
scroll(css_selector='[data-scenario="5"]')
```

### Step 4.2: Horizontal scroll right
```
scroll(direction='right', css_selector='#code-block')
```
Observe: Code scrolls horizontally, revealing truncated long lines.
Check: `container_scroll_left` increases, `scrolled: true`.

### Step 4.3: Smooth horizontal scroll
```
scroll(direction='right', css_selector='#code-block', behavior='smooth')
```
Observe: Horizontal smooth animation within the code block.

---

## Phase 5: Data Table (Scenario 6)

### Step 5.1: Navigate
```
scroll(css_selector='[data-scenario="6"]')
```

### Step 5.2: Vertical table scroll
```
scroll(direction='down', css_selector='#data-table-container')
```
Observe: Table rows scroll. Sticky header stays pinned.

### Step 5.3: Jump to rightmost column with position
```
scroll(position='right', css_selector='#data-table-container')
```
Observe: Table jumps to the rightmost columns in ONE call.
Check: `mode: 'container_scroll_to'`, `container_scroll_left` at max.

---

## Phase 6: Scroll Into View — Centering (Scenario 7)

### Step 6.1: Go back to top first
```
scroll(direction='up', scroll_amount=20)
scroll(direction='up', scroll_amount=20)  # repeat until scrolled: false
```

### Step 6.2: Instant scroll-into-view
```
scroll(css_selector='#scroll-target-center')
```
Observe: Target box teleports to CENTER of viewport (not bottom edge).
Explain: Uses `scrollIntoView({block: 'center'})` — gives max context above and below.

### Step 6.3: Smooth scroll-into-view
```
scroll(direction='up', scroll_amount=5)  # move away first
scroll(css_selector='#scroll-target-center', behavior='smooth')
```
Observe: Page smoothly glides until target is centered. Most polished visual effect.

---

## Phase 7: Sticky Header (Scenario 8)

### Step 7.1: Navigate
```
scroll(css_selector='#sticky-test-target')
```
Observe: Target is centered, NOT hidden behind the sticky sub-header.
Explain: `block: 'center'` avoids header obstruction. `block: 'start'` would fail here.

---

## Phase 8: Scroll-Snap (Scenario 9)

### Step 8.1: Navigate
```
scroll(css_selector='[data-scenario="9"]')
```

### Step 8.2: Snap scroll right
```
scroll(direction='right', css_selector='#snap-container', scroll_amount=1)
```
Observe: Container SNAPS to the next card. Position may not be exactly 100px.
Check: `container_scroll_left` reflects actual snapped position.
Explain: CSS scroll-snap causes position to differ from requested delta. Return value is truth.

### Step 8.3: Smooth snap scroll
```
scroll(direction='right', css_selector='#snap-container', scroll_amount=1, behavior='smooth')
```
Observe: Smooth animation that "catches" on the snap point. Distinctive visual.

---

## Phase 9: Lazy Loading / IntersectionObserver (Scenario 10)

### Step 9.1: Navigate
```
scroll(css_selector='[data-scenario="10"]')
```

### Step 9.2: Screenshot initial state
```
screenshot('scroll-demo-lazy-before.png')
```
Observe: Gray boxes saying "Loading..."

### Step 9.3: Scroll to trigger loading
```
scroll(direction='down', scroll_amount=3)
```
Observe: Boxes transition to green "Loaded!" as they enter the viewport.
Explain: `element.scrollBy()` fires real scroll events, triggering IntersectionObserver.

### Step 9.4: Screenshot loaded state
```
screenshot('scroll-demo-lazy-after.png')
```

---

## Phase 10: Non-Scrollable Container (Scenario 11)

### Step 10.1: Navigate
```
scroll(css_selector='[data-scenario="11"]')
```

### Step 10.2: Try to scroll
```
scroll(direction='down', css_selector='#non-scrollable')
```
Check: `scrolled: false`. Container has `overflow: hidden`.
Explain: Agent immediately knows this container can't scroll. No wasted retries.

---

## Phase 11: Overscroll Containment (Scenario 12)

### Step 11.1: Navigate
```
scroll(css_selector='[data-scenario="12"]')
```

### Step 11.2: Jump container to bottom with position
```
scroll(position='bottom', css_selector='#overscroll-container')
```
Observe: Container scrolls to bottom. Page does NOT scroll — overscroll contained.

### Step 11.3: Verify boundary
```
scroll(position='bottom', css_selector='#overscroll-container')
```
Check: `scrolled: false`. Container at bottom, page stays put.
Explain: `overscroll-behavior: contain` prevents scroll chaining.

---

## Phase 12: Position Scrolling Showcase

### Step 12.1: Jump to bottom of page
```
scroll(position='bottom')
```
Observe: Page teleports to the absolute bottom in ONE call.
Check: `mode: 'viewport_scroll_to'`, `scrolled: true`.
Explain: No need for `scroll(direction='down', scroll_amount=20)` — `position='bottom'` is the `scrollTo` equivalent.

### Step 12.2: Verify boundary
```
scroll(position='bottom')
```
Check: `scrolled: false`. Already at bottom.

### Step 12.3: Smooth scroll back to top
```
scroll(position='top', behavior='smooth')
```
Observe: Page smoothly animates all the way back to the top. The most dramatic smooth scroll.

### Step 12.4: Final screenshot
```
screenshot('scroll-demo-complete.png')
```

---

## Summary of Capabilities Demonstrated

| Capability | Scenarios | Key Parameter |
|---|---|---|
| Viewport scroll (relative) | 1, 2, 3 | `direction` |
| Container scroll (relative) | 4, 6, 11, 12 | `css_selector` + `direction` |
| Horizontal scroll | 5, 6, 9 | `direction='right'/'left'` |
| Scroll into view (centering) | 7, 8 | `css_selector` only |
| Position scroll (absolute) | 4, 5, 11, 12 | `position='top'/'bottom'/'left'/'right'` |
| Instant behavior (default) | All | `behavior='instant'` |
| Smooth behavior | 3, 4, 5, 7, 9, 12 | `behavior='smooth'` |
| Boundary detection | 2, 4, 10, 11, 12 | `scrolled: false` |
| Position tracking | All | `scroll_y`, `page_height` |
| CSS scroll-behavior override | 4, 5 | Default instant overrides CSS smooth |
| Scroll-snap interaction | 9 | Actual position vs requested |
| Lazy loading trigger | 10 | IntersectionObserver fires |
| Non-scrollable detection | 11 | `scrolled: false` on `overflow:hidden` |
| Overscroll containment | 12 | No scroll chaining |