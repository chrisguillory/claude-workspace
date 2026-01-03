# Text Extraction Test Suite

**Purpose:** Validate content extraction behaviors across multiple tools and implementations. Test files contain patterns; each tool has documented expectations.

**Philosophy:** The test files are tool-agnostic pattern libraries. The same file can validate different tools with different expected behaviors.

---

## Test Files

Each file contains specific patterns with "Test N passed" validation markers.

| File | Patterns | Validates | Marker Format |
|------|----------|-----------|---------------|
| `hidden-content.html` | 12 CSS hiding techniques | display:none, visibility:hidden, opacity:0, offscreen, clip-path, sr-only, aria-hidden, HTML hidden, nested, combined | "Test 1 passed" ... "Test 12 passed" |
| `shadow-dom.html` | 5 web component patterns | Open shadow roots, nested shadows, slotted content, hidden in shadow, multiple hosts | "Shadow DOM Test 1 passed" ... "Test 5c passed" |
| `whitespace.html` | 8 normalization scenarios | Source indentation, multiple spaces, tabs/newlines, nbsp, paragraph breaks, lists, inline elements, CJK text | "Whitespace Test 1 passed" ... "Test 8 passed" |
| `preformatted.html` | 5 preservation cases | PRE blocks, CODE inline, TEXTAREA, nested PRE, mixed formatting | "Preformatted Test 1 passed" ... |
| `semantic-blocks.html` | 6 structure patterns | Headings, paragraphs, DIV vs SPAN, BR, tables | "Semantic Test 1 passed" ... |
| `marriott-modals.html` | 4 modal dialogs | Hidden benefit descriptions (original bug pattern) | Benefit description text |
| `accordion-pattern.html` | 5 FAQ sections | Collapsed/expanded sections with display:none | Section content text |
| `kitchen-sink.html` | All patterns combined | Comprehensive regression test | All markers from above |
| `threshold-test.html` | Smart extraction threshold | 500-char threshold for main/article selection | source_element, smart_info fields |
| `iframes.html` | 6 iframe patterns | EXCLUSION test - iframe content NOT extracted (by design) | "Iframe Test 1-6 passed" (parent doc); NO "SHOULD_NOT_APPEAR" |
| `forms.html` | 12 form element patterns | Labels extracted, input values NOT extracted, aria-describedby | "Forms Test 1-12 passed"; NO "SHOULD_NOT_APPEAR" |
| `tables.html` | 11 table patterns | Caption, headers, cells, hidden rows, colspan/rowspan, nested | "Tables Test 1-11" markers |
| `accessibility.html` | 11 ARIA patterns | aria-hidden extracted, role=presentation, landmarks, live regions | "A11y Test 1-11 passed"; aria-label NOT in output |
| `indexeddb-storage.html` | 7 IndexedDB patterns | `save_storage_state(include_indexeddb=True)` capture and `navigate_with_session(storage_state_file)` restore | `verifyData()` returns `testsPassed` array |

### Hover Tool Actionability Tests

These test files validate the `hover()` tool's 5 actionability checks:

| File                    | Patterns               | Validates                                                                                                           | Marker Format                      |
|-------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------------------|
| `hover-stability.html`  | 10 stability scenarios | Multi-signal stability detection (RAF timing, getAnimations, 5px threshold), CSS/JS animations, infinite animations | "Hover Stability Test 1-10 passed" |
| `hover-visibility.html` | 8 visibility scenarios | display:none, visibility:hidden, opacity:0, pointer-events:none, clip-path, zero-size                               | "Hover Visibility Test 1-8 passed" |
| `hover-occlusion.html`  | 8 occlusion scenarios  | Modal overlays, partial overlays, transparent overlays, dropdown menus, pointer-events:none pass-through            | "Hover Occlusion Test 1-8 passed"  |

**Hover Tool Actionability Checks:**
1. **Duration validation** (0-30000ms)
2. **Scroll into view** (scrollIntoView with behavior:'instant')
3. **Visibility check** (element.is_displayed())
4. **Stability check** (multi-signal: requestAnimationFrame, getAnimations(), position polling)
5. **Pointer events check** (elementFromPoint at element center)

**Validation Approach:** Each fixture includes JavaScript helpers (`validateHover()`, `getAnimationInfo()`, `checkVisibility()`, `checkOcclusion()`) for programmatic verification of test results.

**Test Definitions:** See `tests/hover_tests.yaml` for structured test case definitions.
**Manual Tests:** See `tests/HOVER_TEST.md` for step-by-step test procedures.

### Future Work Test Files (get_aria_snapshot enhancements)

These test files validate features that are **not yet implemented** but document expected behavior for future phases:

| File | Patterns | Validates | Marker Format |
|------|----------|-----------|---------------|
| `aria-states.html` | 11 state patterns | expanded, selected, pressed, haspopup, required, invalid, modal, busy, current, disabled, hidden | "ARIA States Test 1-11 passed" |
| `aria-values.html` | 10 value patterns | valuenow, valuemin, valuemax, valuetext for sliders/progress/meters | "ARIA Values Test 1-10 passed" |
| `role-mapping.html` | 14 role patterns | dialog, details, figure, search, tree, grid, menu, tabpanel, etc. | "Role Mapping Test 1-14 passed" |
| `context-roles.html` | 15 context patterns | header/footer scoping, section naming, anchor href, img alt, form naming | "Context Roles Test 1-15 passed" |
| `live-regions.html` | 14 live region patterns | aria-live, aria-atomic, aria-relevant, alert, status, log, timer | "Live Regions Test 1-14 passed" |

**Note:** These tests are for `get_aria_snapshot()` enhancements. Current implementation may not pass all tests.

---

## Tool-Specific Expectations

### Selenium MCP Tools

| Tool | hidden-content | shadow-dom | whitespace | preformatted | semantic-blocks | Real-world |
|------|----------------|------------|------------|--------------|-----------------|------------|
| **`get_page_text()`** | ✅ All 12 tests | ✅ All 5 tests | ✅ All 8 tests | ✅ All 5 tests | ✅ All 6 tests | ✅ All content |
| **`get_page_html()`** | N/A (returns HTML) | N/A | N/A | N/A | N/A | N/A |
| **`get_aria_snapshot()`** | Different format | Different format | Different format | Different format | Different format | Different format |

**`get_page_text()` Pass Criteria:**
- All validation markers must appear in extracted text
- Hidden content must be included (textContent-based)
- Shadow DOM content must be traversed
- Whitespace must be normalized (except in PRE/CODE)

**`get_aria_snapshot()` Expectations:**
- Returns YAML accessibility tree, not plain text
- Different validation approach needed
- May not include all hidden content (depends on aria-hidden)

### Reference APIs (Native Browser)

| API | hidden-content | Behavior |
|-----|----------------|----------|
| `document.body.textContent` | ✅ Includes all | Raw textContent, includes hidden |
| `document.body.innerText` | ❌ Excludes hidden | Visible text only (rendered) |

**Use Case:** If we ever implement a `get_visible_text()` tool (innerText-based), it should FAIL the hidden-content tests - that would be correct behavior.

---

## Running Tests

### Start HTTP Server

Chrome extensions cannot access `file://` URLs due to security restrictions. Serve the test files via HTTP:

```bash
cd /Users/chris/claude-workspace/mcp/selenium-browser-automation/examples
uv run python -m http.server 8888
```

Test files are then available at `http://localhost:8888/`

### Generic Workflow

```javascript
// 1. Navigate to test file
navigate("http://localhost:8888/hidden-content.html")

// 2. Extract content using your tool
const output = <your_extraction_method>()

// 3. Check for validation markers
const allTestsPassed = [
  "Test 1 passed",
  "Test 2 passed",
  // ... check all expected markers
].every(marker => output.includes(marker))
```

### Selenium MCP Example

```javascript
navigate("http://localhost:8888/hidden-content.html", fresh_browser=true)
const output = get_page_text(selector="body")
// Verify: All 12 "Test N passed" markers present
```

### Claude in Chrome Example

```javascript
navigate("http://localhost:8888/hidden-content.html")
const output = get_page_text()
// Compare with Selenium output
```

### Visual Reference (Direct File Access)

```bash
open examples/hidden-content.html
# DevTools → Elements → Cmd+F → "Test 1 passed"
# Verify all test strings exist in DOM
```

### IndexedDB Storage State Test

Tests `save_storage_state(include_indexeddb=True)` capture and `navigate_with_session(storage_state_file)` restore.

**Key Design:** No auto-initialization on page load. All actions are explicit to prevent false positives.

```javascript
// 1. Navigate to test page (IndexedDB is empty)
navigate("file:///.../examples/indexeddb-storage.html", fresh_browser=true)

// 2. Create test data explicitly
execute_javascript("window.initializeDatabases()")

// 3. Capture with IndexedDB
save_storage_state("test.json", include_indexeddb=true)
// Returns: indexeddb_databases_count=2, indexeddb_records_count=7

// 4. New browser session with restoration
navigate_with_session("file:///.../examples/indexeddb-storage.html",
                      storage_state_file="test.json")

// 5. Verify data exists WITHOUT calling initializeDatabases
execute_javascript(`(async () => {
  const dbs = await indexedDB.databases();
  return { databases: dbs.length }; // Should be 2
})()`)
```

**Pass Criteria:**
- Step 1: `indexedDB.databases().length === 0` (clean load)
- Step 3: Capture returns 2 databases, 7 records
- Step 5: `indexedDB.databases().length === 2` (restored, not recreated)

**Test Cases Validated:**
| Test | What's Validated |
|------|------------------|
| 1 | Object store with keyPath, unique index |
| 2 | Date type serialization (`__type: "Date"`) |
| 3 | Nested objects and arrays |
| 4 | Out-of-line keys (no keyPath) |
| 5 | Auto-increment keys |
| 6 | MultiEntry index |
| 7 | Multiple databases |

---

## Validation Criteria

### Pass/Fail by Tool

**For `get_page_text()` (Selenium MCP):**
| Test File | Pass Criteria |
|-----------|---------------|
| hidden-content.html | All 12 "Test N passed" markers present |
| shadow-dom.html | All 7 validation markers present |
| whitespace.html | All 8 "Whitespace Test N passed" markers present |
| preformatted.html | All 5 tests, exact whitespace in PRE blocks |
| semantic-blocks.html | All 6 tests, correct paragraph breaks |
| marriott-modals.html | All 4 hidden benefit descriptions extracted |
| accordion-pattern.html | All 5 section contents extracted |
| kitchen-sink.html | All markers from all categories |
| threshold-test.html | `source_element="main"`, `smart_info.fallback_used=false`, sidebar excluded |

**For other tools:** Define expected behavior in the matrix above.

---

## Behaviors Tested

### Content Inclusion (should extract)
- ✅ `display: none` content
- ✅ `visibility: hidden` content
- ✅ `opacity: 0` content
- ✅ Shadow DOM content (open roots)
- ✅ `aria-hidden="true"` content
- ✅ Offscreen positioned content
- ✅ Clipped content (clip-path)

### Content Exclusion (should NOT extract)
- ❌ `<script>` element content
- ❌ `<style>` element content
- ❌ `<template>` element content
- ❌ `<noscript>` element content
- ❌ SVG internal structures
- ❌ Iframe content (separate navigation required)
- ❌ Closed shadow roots (browser security)

### Formatting
- Whitespace normalization (collapse spaces, preserve structure)
- PRE/CODE exact preservation
- Block element line breaks
- Paragraph separation

---

## Adding New Tests

When adding tests for new patterns or new tools:

1. **Create test file** with clear validation markers ("Test N passed")
2. **Document in Test Files table** above
3. **Add tool expectations** in Tool-Specific Expectations section
4. **Define pass criteria** for each applicable tool

---

## Future Tools

Space for tools we might add:

| Tool | Purpose | Test Expectations |
|------|---------|-------------------|
| `get_visible_text()` | innerText-based (visible only) | Should FAIL hidden-content (correct behavior) |
| `extract_tables()` | Table-specific extraction | New test file needed |
| ... | ... | ... |

---

## Troubleshooting

**Output differs between tools:**
- Check Tool-Specific Expectations - differences may be expected
- Compare marker presence, not exact text match
- Whitespace differences are often acceptable

**"Selector not found" error:**
- Verify file:// URL is correct
- Try `selector="body"` as default

**Test markers not found:**
- Check if tool extracts hidden content (some don't by design)
- Verify Shadow DOM traversal is implemented
- Check for script/style content leaking through

---

## References

- [textContent vs innerText](https://kellegous.com/j/2013/02/27/innertext-vs-textcontent/)
- [Shadow DOM Specification](https://dom.spec.whatwg.org/#shadow-trees)
- [WAI-ARIA Hidden Content](https://www.w3.org/TR/wai-aria/)