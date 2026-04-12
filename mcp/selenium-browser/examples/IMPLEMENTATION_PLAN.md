# Test Suite Implementation Plan

**Purpose:** Validate `get_page_text()` extraction handles hidden content, Shadow DOM, whitespace, and real-world patterns correctly.

**Architecture Decision:** 7 focused test files + 1 comprehensive file (flat structure, no subdirectories)

**Validation Method:** Open each file in Chrome and Selenium, compare extracted text outputs

---

## File Structure

```
examples/
  README.md                    # Test catalog, usage instructions
  hidden-content.html          # 12 CSS hiding techniques
  shadow-dom.html              # 5 web component patterns
  whitespace.html              # 8 normalization scenarios
  preformatted.html            # 5 whitespace preservation cases
  semantic-blocks.html         # 6 paragraph structure patterns
  marriott-modals.html         # Real-world: Modal dialog pattern
  accordion-pattern.html       # Real-world: Collapsed sections
  kitchen-sink.html            # All tests combined
  IMPLEMENTATION_PLAN.md       # This file
```

---

## Test File Specifications

### 1. hidden-content.html

**Purpose:** Validate extraction of content hidden via CSS techniques

**Original Bug:** Marriott elite status pages had benefit descriptions in `<div class="modal-content d-none">` that Selenium's `innerText` missed but should be extracted.

**Test Cases:**

| # | Technique | CSS | Expected Behavior |
|---|-----------|-----|-------------------|
| 1 | display:none | `display: none` | Text extracted (Marriott modal case) |
| 2 | visibility:hidden | `visibility: hidden` | Text extracted |
| 3 | opacity:0 | `opacity: 0` | Text extracted |
| 4 | height:0 | `height: 0; overflow: hidden` | Text extracted |
| 5 | zero-dimensions | `width: 0; height: 0` | Text extracted |
| 6 | offscreen | `position: absolute; left: -9999px` | Text extracted |
| 7 | clip-path | `clip-path: inset(100%)` | Text extracted |
| 8 | sr-only | `.sr-only` class (screen reader only) | Text extracted |
| 9 | aria-hidden | `aria-hidden="true"` (semantic, not CSS) | Text extracted |
| 10 | HTML hidden | `<div hidden>` attribute | Text extracted |
| 11 | nested-hidden | Parent hidden, child visible | Text extracted |
| 12 | combined | Multiple techniques at once | Text extracted |

**HTML Structure:**

```html
<!DOCTYPE html>
<html>
<head>
  <title>Test: Hidden Content Extraction</title>
  <style>
    .test-case { border: 1px solid #ccc; margin: 20px; padding: 15px; }
    .test-case h2 { color: #333; font-size: 18px; }
    .expected { background: #e8f5e9; padding: 10px; margin: 10px 0; }
    .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; }
  </style>
</head>
<body>
  <!--
  TEST: hidden-content.html
  PURPOSE: Validate CSS-based hiding techniques are extracted
  EXPECTED OUTPUT: All 12 test case descriptions should appear
  EXTRACTION METHOD: textContent (includes hidden elements)
  UPDATED: 2025-12-22
  -->

  <h1>Hidden Content Extraction Tests</h1>
  <p>This page tests 12 CSS hiding techniques. All hidden content should be extracted.</p>

  <section id="test-1" class="test-case">
    <h2>Test 1: display: none (Marriott Modal Pattern)</h2>
    <div class="expected">Expected: "Hertz President's Circle Status" AND "Partner benefit description..."</div>
    <div class="actual">
      <button>Hertz President's Circle® Status</button>
      <div class="modal-content" style="display: none;">
        Partner benefit description: Marriott Bonvoy® has partnered with Hertz to provide
        complimentary President's Circle status enrollment for Platinum, Titanium, and
        Ambassador Elite members.
      </div>
    </div>
  </section>

  <section id="test-2" class="test-case">
    <h2>Test 2: visibility: hidden</h2>
    <div class="expected">Expected: "Visible text" AND "Hidden but should be extracted"</div>
    <div class="actual">
      <p>Visible text</p>
      <p style="visibility: hidden;">Hidden but should be extracted</p>
    </div>
  </section>

  <!-- Tests 3-12 follow similar structure -->

</body>
</html>
```

**Expected Output Pattern:**

```
Title: Test: Hidden Content Extraction
URL: file:///path/to/hidden-content.html
Source: <body>
---
Hidden Content Extraction Tests
This page tests 12 CSS hiding techniques. All hidden content should be extracted.

Test 1: display: none (Marriott Modal Pattern)
Expected: "Hertz President's Circle Status" AND "Partner benefit description..."
Hertz President's Circle® Status
Partner benefit description: Marriott Bonvoy® has partnered with Hertz...
[continues with all 12 tests...]
```

---

### 2. shadow-dom.html

**Purpose:** Validate extraction traverses Shadow DOM boundaries

**Background:** Standard DOM methods don't traverse `node.shadowRoot` automatically. Our implementation explicitly checks for shadow roots and recursively walks them.

**Test Cases:**

| # | Pattern | Description |
|---|---------|-------------|
| 1 | simple-shadow | Basic open shadow root with text content |
| 2 | nested-shadow | Shadow root within shadow root |
| 3 | slotted-content | Shadow DOM with `<slot>` elements |
| 4 | hidden-in-shadow | display:none content inside shadow root |
| 5 | multiple-hosts | Multiple web components on same page |

**HTML Structure:**

```html
<!DOCTYPE html>
<html>
<head>
  <title>Test: Shadow DOM Extraction</title>
</head>
<body>
  <!--
  TEST: shadow-dom.html
  PURPOSE: Validate Shadow DOM traversal
  EXPECTED OUTPUT: All shadow root content should appear
  CAVEAT: Closed shadow roots cannot be traversed (browser limitation)
  UPDATED: 2025-12-22
  -->

  <h1>Shadow DOM Extraction Tests</h1>

  <section id="test-1">
    <h2>Test 1: Simple Open Shadow Root</h2>
    <div class="expected">Expected: "Host element" AND "Shadow content"</div>
    <simple-component></simple-component>
  </section>

  <script>
    // Test 1: Simple shadow root
    class SimpleComponent extends HTMLElement {
      constructor() {
        super();
        const shadow = this.attachShadow({mode: 'open'});
        shadow.innerHTML = '<p>Shadow content inside web component</p>';
      }
    }
    customElements.define('simple-component', SimpleComponent);

    // Test 2: Nested shadow roots
    class NestedComponent extends HTMLElement {
      constructor() {
        super();
        const shadow = this.attachShadow({mode: 'open'});
        shadow.innerHTML = `
          <div>Outer shadow content</div>
          <inner-component></inner-component>
        `;
      }
    }
    class InnerComponent extends HTMLElement {
      constructor() {
        super();
        const shadow = this.attachShadow({mode: 'open'});
        shadow.innerHTML = '<p>Inner shadow content (nested)</p>';
      }
    }
    customElements.define('nested-component', NestedComponent);
    customElements.define('inner-component', InnerComponent);

    // Tests 3-5 follow similar patterns
  </script>

</body>
</html>
```

---

### 3. whitespace.html

**Purpose:** Validate whitespace normalization while preserving semantic structure

**Test Cases:**

| # | Scenario | Input | Expected Output |
|---|----------|-------|-----------------|
| 1 | source-indentation | HTML with 6-space indents | Collapsed to single spaces |
| 2 | multiple-spaces | "Hello     world" | "Hello world" |
| 3 | tabs-newlines | Mixed \t and \n | Normalized to spaces |
| 4 | nbsp | `"Hello&nbsp;world"` | "Hello world" |
| 5 | paragraph-breaks | Two `<p>` elements | Double newline between |
| 6 | list-spacing | `<ul><li>` items | Line break per item |
| 7 | inline-elements | `<span>` and `<a>` | No breaks inserted |
| 8 | cjk-text | Chinese characters | No spaces added |

---

### 4. preformatted.html

**Purpose:** Validate exact whitespace preservation in PRE/CODE/TEXTAREA

**Test Cases:**

| # | Element | Content | Expected |
|---|---------|---------|----------|
| 1 | PRE | Code block with indentation | Exact preservation |
| 2 | CODE inline | `<code>const x = 1;</code>` | Exact preservation |
| 3 | TEXTAREA | User input with newlines | Exact preservation |
| 4 | Nested PRE | PRE inside DIV | Exact preservation in PRE only |
| 5 | Mixed | PRE + regular paragraphs | Preserve PRE, normalize paragraphs |

**Critical Validation:**

```javascript
// This code block should preserve exact formatting:
function example() {
    const    spaces   =   "multiple";
    return   spaces;
}
```

---

### 5. semantic-blocks.html

**Purpose:** Validate block vs inline elements create appropriate line breaks

**Test Cases:**

| # | Element Type | Behavior |
|---|--------------|----------|
| 1 | Headings (H1-H6) | Create line breaks before/after |
| 2 | Paragraphs (P) | Create double newlines between |
| 3 | DIV | Block element, creates breaks |
| 4 | SPAN | Inline element, no breaks |
| 5 | BR | Single line break |
| 6 | Table (TR, TD) | Row breaks, cell spacing |

**Expected Structure:**

```
Heading 1

Paragraph one content here.

Paragraph two content here.

List item 1
List item 2
```

---

### 6. marriott-modals.html

**Purpose:** Real-world pattern replicating the original bug report

**Pattern:** Benefit cards with visible titles and hidden modal descriptions (display:none)

**Structure:**

```html
<div class="benefit-card">
  <h3>Hertz President's Circle® Status</h3>
  <button class="learn-more">Learn More</button>
  <div class="modal-content d-none">
    <h4>Hertz President's Circle® Status</h4>
    <p>Marriott Bonvoy® has partnered with Hertz to provide complimentary
    President's Circle status enrollment for Platinum, Titanium, and
    Ambassador Elite members. President's Circle benefits include...</p>
  </div>
</div>

<!-- Repeat for 4-5 different benefits -->
```

**Validation:** Both card title AND hidden modal content should be extracted.

---

### 7. accordion-pattern.html

**Purpose:** Real-world accordion with collapsed sections

**Pattern:** FAQ or feature list where only one section is expanded, others have `display: none`

**Structure:**

```html
<div class="accordion">
  <div class="section expanded">
    <h3 class="header">Section 1 (Expanded)</h3>
    <div class="content" style="display: block;">
      Visible content in expanded section.
    </div>
  </div>

  <div class="section collapsed">
    <h3 class="header">Section 2 (Collapsed)</h3>
    <div class="content" style="display: none;">
      Hidden content that should still be extracted.
    </div>
  </div>

  <!-- More sections... -->
</div>
```

**Validation:** All section content should be extracted regardless of expanded/collapsed state.

---

### 8. kitchen-sink.html

**Purpose:** Comprehensive test combining all patterns for regression validation

**Contents:**
- 3 hidden content examples (display:none, visibility:hidden, opacity:0)
- 1 shadow DOM component
- 2 whitespace scenarios
- 1 preformatted block
- 2 semantic block patterns
- 1 modal pattern
- 1 accordion pattern

**Usage:** Quick validation that entire extraction pipeline works. If this passes, detailed tests can identify specific failures.

---

## README.md Structure

```markdown
# Text Extraction Test Suite

Validates that `get_page_text()` correctly extracts content from web pages, including hidden elements, Shadow DOM, and complex formatting.

## Quick Start

### Option 1: Chrome Browser
1. Open any `.html` file in Chrome
2. View source to see test structure
3. Expected output documented in HTML comments

### Option 2: Selenium MCP Server
```bash
# In Claude Code with Selenium MCP enabled:
navigate("file:///absolute/path/to/hidden-content.html")
get_page_text(selector="body")
```

## Test File Catalog

| File | Purpose | Test Count | Key Validations |
|------|---------|------------|-----------------|
| `hidden-content.html` | CSS hiding | 12 | display:none, visibility:hidden, opacity:0, offscreen, clip-path, aria-hidden, HTML hidden |
| `shadow-dom.html` | Web components | 5 | Open shadow roots, nested shadows, slotted content |
| `whitespace.html` | Text normalization | 8 | Source indentation, multiple spaces, paragraph breaks, nbsp |
| `preformatted.html` | Whitespace preservation | 5 | PRE, CODE, TEXTAREA exact formatting |
| `semantic-blocks.html` | Paragraph structure | 6 | Headings, paragraphs, DIV vs SPAN, BR, tables |
| `marriott-modals.html` | Real-world pattern | 4 | Hidden benefit descriptions in modal dialogs |
| `accordion-pattern.html` | Real-world pattern | 3 | Collapsed/expanded FAQ sections |
| `kitchen-sink.html` | Comprehensive | All | Regression test combining all patterns |

## Expected Behavior

✅ **Should Extract:**
- Content with `display: none`
- Content with `visibility: hidden`
- Content with `opacity: 0`
- Content inside Shadow DOM (open roots)
- Content with `aria-hidden="true"`
- Content positioned offscreen
- All semantic text structure

❌ **Should NOT Extract:**
- Script/style/template element content
- Iframe content (not traversable without navigation)
- Closed shadow root content (browser security limitation)
- SVG/Canvas internal data structures

## Validation Workflow

1. **Open test file** in Chrome or via Selenium `navigate()`
2. **Extract text** using `get_page_text(selector="body")`
3. **Compare output** to expected patterns in HTML comments
4. **Verify specific test cases** passed by searching for expected strings

### Example Validation

```javascript
// Test: hidden-content.html
const output = get_page_text(selector="body");

// Validate Test 1: display:none modal content
if (output.includes("Partner benefit description") &&
    output.includes("President's Circle")) {
  console.log("✅ Test 1 passed: display:none content extracted");
} else {
  console.log("❌ Test 1 failed: Missing hidden modal content");
}
```

## Adding New Tests

1. Create new `.html` file in `examples/`
2. Add HTML comment header with test metadata
3. Structure each test case as `<section id="test-N" class="test-case">`
4. Include expected output in `.expected` div
5. Update this README table
6. Add validation to `kitchen-sink.html`

## Troubleshooting

**Issue:** "Selector not found" error
- Verify file opened correctly: check URL in error message
- Try `selector="body"` instead of specific selectors

**Issue:** Output differs between Chrome and Selenium
- Check which specific test case fails
- Open focused test file (e.g., `shadow-dom.html`) to isolate
- Verify browser version supports the feature (Shadow DOM requires modern browsers)

**Issue:** Whitespace differences in output
- Check if issue is in preformatted blocks (expected) vs regular text (bug)
- Compare paragraph breaks: should be double newline (`\n\n`)
- Validate spaces normalized: multiple spaces should become one

## Test Coverage Summary

| Category | Coverage | Notes |
|----------|----------|-------|
| CSS Visibility | 12/12 techniques | Comprehensive |
| Shadow DOM | 5/5 patterns | Open roots only (closed roots not traversable) |
| Whitespace | 8/8 scenarios | Normalization + preservation |
| Block Structure | 6/6 element types | Paragraphs, headings, lists, tables |
| Real-World | 2 patterns | Modals, accordions |

## References

- [Original Bug Report](link-to-marriott-issue)
- [Chrome textContent vs innerText](https://kellegous.com/j/2013/02/27/innertext-vs-textcontent/)
- [WAI-ARIA Hidden Content](https://www.w3.org/TR/wai-aria/)
```

---

## Implementation Order

1. ✅ Create `IMPLEMENTATION_PLAN.md` (this file)
2. Create `README.md` with test catalog
3. Create `hidden-content.html` (most critical - original bug)
4. Create `shadow-dom.html`
5. Create `whitespace.html`
6. Create `preformatted.html`
7. Create `semantic-blocks.html`
8. Create `marriott-modals.html`
9. Create `accordion-pattern.html`
10. Create `kitchen-sink.html` (combines all)
11. Test each file in Chrome browser (visual verification)
12. Test each file via Selenium `get_page_text()` (automated verification)
13. Document any discrepancies between Chrome and Selenium
14. Update expected outputs in HTML comments

---

## Success Criteria

✅ All 8 test files created with complete HTML structure
✅ Each file has HTML comment header with test metadata
✅ README.md documents all test files in table format
✅ `kitchen-sink.html` includes all test patterns
✅ Each test file validated in Chrome (manual)
✅ Each test file validated in Selenium (automated)
✅ Expected outputs documented in test files
✅ Any Chrome vs Selenium differences documented

---

## Future Enhancements

- Add automated test runner script
- Generate JSON diff reports (Chrome vs Selenium outputs)
- Add visual regression tests (screenshots)
- Create CI/CD pipeline for test execution
- Add performance benchmarks (extraction time)
- Test additional edge cases (SVG text, MathML)
