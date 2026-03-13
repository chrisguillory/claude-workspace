# Accessible Name Computation Research

Research into how browser automation tools compute accessible names, with focus on
the 4 missing host-language label cases in our ARIA snapshot engine.

**Date:** 2025-03-13
**Scope:** AccName 1.2 spec, Playwright implementation, CDP approach, reference libraries

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Playwright's Implementation](#2-playwrights-implementation)
3. [Chrome CDP Accessibility API](#3-chrome-cdp-accessibility-api)
4. [AccName 1.2 Spec: The 4 Missing Cases](#4-accname-12-spec-the-4-missing-cases)
5. [Other Tools and Libraries](#5-other-tools-and-libraries)
6. [Implementation Recommendations](#6-implementation-recommendations)

---

## 1. Current Implementation Analysis

Our `computeAccessibleName` function in `aria_snapshot.js` and `visual_tree.js` follows
this priority order:

```
1. aria-label
2. aria-labelledby (via getElementById, includes hidden elements)
3. label[for="id"] (explicit label association)
4. closest('label') (implicit/wrapping label)
5. textContent for buttons, links, headings
6. title attribute
7. alt for images
8. placeholder/value for inputs/textareas
```

### Problems with current implementation

**Spec-order violation:** Our implementation checks `aria-label` before `aria-labelledby`.
The W3C AccName 1.2 spec mandates `aria-labelledby` first (step 2B), then embedded
control values (step 2C), then `aria-label` (step 2D), then host language (step 2E).

**Label handling is in the wrong position:** Steps 3-4 (label association) should be
part of step 2E (host language), checked only after aria-label. Currently they fire
before textContent, title, alt, and placeholder -- which is mostly correct for form
controls but applies to ALL elements since there is no tag check.

**Missing host-language cases:**

| Case | Status | Impact |
|------|--------|--------|
| `<fieldset>` name from `<legend>` | MISSING | fieldset nodes appear unnamed |
| `<figure>` name from `<figcaption>` | MISSING | figure nodes appear unnamed |
| `<table>` name from `<caption>` | MISSING | table nodes appear unnamed |
| `<label>` for form controls | PARTIAL | Works for `label[for]` and wrapping label, but uses `textContent` instead of recursive name computation |

**textContent vs recursive computation:** Using `textContent` for label, legend,
figcaption, and caption is a shortcut. The spec requires recursively computing the
text alternative of the labeling element, which handles nested interactive controls,
hidden elements referenced by aria-labelledby, and embedded control values.

---

## 2. Playwright's Implementation

### Architecture: Pure JavaScript, No CDP

Playwright computes accessible names **entirely in injected JavaScript** running in the
page context. They do NOT use Chrome DevTools Protocol `Accessibility.getFullAXTree`.

**Source file:** `packages/injected/src/roleUtils.ts`
(~1235 lines, with ~500 lines dedicated to accessible name computation)

**Key function:** `getElementAccessibleName(element, includeHidden)` which calls
`getTextAlternativeInternal(element, options)` -- a recursive function implementing
the full AccName 1.2 algorithm.

### Algorithm Step Order (Playwright)

Playwright follows the spec order precisely:

```
Step 1:  Check if role prohibits naming (caption, code, generic, etc.)
Step 2A: Hidden Not Referenced - skip if hidden (unless referenced)
Step 2B: aria-labelledby - process IDREFs (only if not already in labelledby traversal)
Step 2C: Embedded Control - return value if control embedded in label/labelledby
Step 2D: aria-label - return if non-empty
Step 2E: Host Language Label (see below)
Step 2F: Name From Content - recurse into children if role allows
Step 2G: Text node - return text content
Step 2I: Tooltip - title attribute fallback
```

### How Playwright Handles the 4 Host-Language Cases (Step 2E)

**All 4 cases are implemented in Playwright.** Here is their exact approach:

#### Fieldset + Legend

```typescript
// https://w3c.github.io/html-aam/#fieldset-and-legend-elements
if (!labelledBy && tagName === 'FIELDSET') {
    for (let child = element.firstElementChild; child; child = child.nextElementSibling) {
        if (elementSafeTagName(child) === 'LEGEND') {
            return getTextAlternativeInternal(child, {
                ...childOptions,
                embeddedInNativeTextAlternative: { element: child, hidden: isElementHiddenForAria(child) },
            });
        }
    }
    return element.getAttribute('title') || '';
}
```

Key behaviors:
- Only checked when `aria-labelledby` is NOT present (aria-labelledby already handled in step 2B)
- Iterates `firstElementChild` siblings -- takes the **first** `<legend>` direct child
- Uses **recursive** `getTextAlternativeInternal`, not `textContent`
- Falls back to `title` attribute
- The `embeddedInNativeTextAlternative` flag allows hidden legend content to contribute

#### Figure + Figcaption

```typescript
// https://w3c.github.io/html-aam/#figure-and-figcaption-elements
if (!labelledBy && tagName === 'FIGURE') {
    for (let child = element.firstElementChild; child; child = child.nextElementSibling) {
        if (elementSafeTagName(child) === 'FIGCAPTION') {
            return getTextAlternativeInternal(child, {
                ...childOptions,
                embeddedInNativeTextAlternative: { element: child, hidden: isElementHiddenForAria(child) },
            });
        }
    }
    return element.getAttribute('title') || '';
}
```

Key behaviors:
- Takes the **first** `<figcaption>` direct child
- Same recursive computation and fallback pattern as fieldset/legend

**Spec evolution note:** W3C HTML-AAM [PR #359](https://github.com/w3c/html-aam/pull/359)
changed figure/figcaption from a "labelled by" relationship to a "details for"
relationship. This means figcaption technically should provide a *description*
(via `aria-details`) rather than an accessible *name*. However, Playwright still
uses it for the name, matching current browser behavior. This is a pragmatic choice
for snapshot accuracy.

#### Table + Caption

```typescript
// https://w3c.github.io/html-aam/#table-element
if (tagName === 'TABLE') {
    for (let child = element.firstElementChild; child; child = child.nextElementSibling) {
        if (elementSafeTagName(child) === 'CAPTION') {
            return getTextAlternativeInternal(child, {
                ...childOptions,
                embeddedInNativeTextAlternative: { element: child, hidden: isElementHiddenForAria(child) },
            });
        }
    }
    // SPEC DIFFERENCE: browsers also support <table summary="...">, spec does not mention it
    const summary = element.getAttribute('summary') || '';
    if (summary) return summary;
    return element.getAttribute('title') || '';
}
```

Key behaviors:
- Takes the **first** `<caption>` direct child
- NOTE: Does NOT check `!labelledBy` guard -- this differs from fieldset/figure
- Falls back to `summary` attribute (non-spec, but browsers support it)
- Then falls back to `title`

#### Form Controls + Label

```typescript
// For textarea, select, and all input types:
if (!labelledBy && (tagName === 'TEXTAREA' || tagName === 'SELECT' || tagName === 'INPUT')) {
    const labels = (element as HTMLInputElement).labels || [];
    if (labels.length)
        return getAccessibleNameFromAssociatedLabels(labels, options);
    // Falls through to placeholder/title handling...
}
```

Where `getAccessibleNameFromAssociatedLabels` is:

```typescript
function getAccessibleNameFromAssociatedLabels(labels, options) {
    return [...labels].map(label => getTextAlternativeInternal(label, {
        ...options,
        embeddedInLabel: { element: label, hidden: isElementHiddenForAria(label) },
        embeddedInNativeTextAlternative: undefined,
        embeddedInLabelledBy: undefined,
        embeddedInDescribedBy: undefined,
        embeddedInTargetElement: undefined,
    })).filter(accessibleName => !!accessibleName).join(' ');
}
```

Key behaviors:
- Uses the **native `element.labels` API** -- handles both `for=` and wrapping labels automatically
- Supports **multiple labels** (joins with space)
- Recursively computes each label's text alternative
- Sets `embeddedInLabel` flag so embedded controls return their values instead of names
- Button, output, and file/image input also check labels before their element-specific rules

### Playwright's Spec Differences (Documented in Source)

Playwright documents several places where they diverge from the spec to match browser behavior:

1. Input type=button/submit/reset: Spec says ignore host-language when aria-labelledby
   is present. Chromium/Firefox ignore this. Playwright follows Chromium/Firefox.
2. `<table summary>`: Not in spec, but all browsers support it.
3. Input type=image: "Submit" instead of spec's "Submit Query".
4. Empty textContent fallback to title: Spec says return empty, browsers fall back to title.

---

## 3. Chrome CDP Accessibility API

### What CDP Provides

The [Chrome DevTools Protocol Accessibility domain](https://chromedevtools.github.io/devtools-protocol/tot/Accessibility/)
offers two relevant methods:

**`Accessibility.getFullAXTree`**
- Returns the complete accessibility tree as `AXNode[]`
- Each node includes `name` (pre-computed by Chrome), `role`, `description`, `value`
- Optional `depth` parameter limits tree depth
- Optional `frameId` for cross-frame access

**`Accessibility.queryAXTree`**
- Queries subtree by accessible name and/or role
- Returns matching `AXNode[]` including ignored nodes
- Useful for targeted lookups

### AXNode Structure

Each node includes:
- `nodeId`: Unique AXNodeId
- `role`: Computed role (AXValue)
- `name`: **Pre-computed accessible name** (AXValue)
- `name.sources`: Array of `AXValueSource` showing WHERE the name came from
- `description`: Computed description
- `value`: Current value
- `properties`: Array of ARIA properties
- `backendDOMNodeId`: Link back to DOM node

### AXValueSource (Name Provenance)

The `sources` array on the `name` property tracks computation provenance:
- `type`: attribute, implicit, style, contents, placeholder, relatedElement
- `nativeSource`: description, figcaption, label, labelfor, labelwrapped, legend, tablecaption, title
- `attribute`: The relevant attribute name
- `superseded`: Whether this source was overridden by a higher-priority one

This means Chrome already computes ALL the host-language label cases natively and
even tells you which mechanism provided the name.

### Selenium 4 CDP Access

Selenium 4 supports CDP via `driver.execute_cdp_cmd()`:

```python
# Get full accessibility tree
result = driver.execute_cdp_cmd("Accessibility.getFullAXTree", {})
nodes = result["nodes"]

# Each node has pre-computed name
for node in nodes:
    name = node.get("name", {}).get("value", "")
    role = node.get("role", {}).get("value", "")
```

### Performance and Trade-offs

| Factor | CDP Approach | JavaScript Approach |
|--------|-------------|-------------------|
| **Accuracy** | Chrome-native, most accurate | Must reimplement spec, risk of bugs |
| **Performance** | Requires enabling accessibility domain, adds overhead | Runs in page context, fast for single elements |
| **Full tree** | One CDP call returns entire tree with names | Must traverse DOM ourselves |
| **Maintenance** | Chrome handles spec updates automatically | Must track spec changes manually |
| **API stability** | Marked "Experimental" | Our code, fully controlled |
| **Cross-browser** | Chrome/Chromium only | Works in any browser |
| **Puppeteer finding** | CDP approach was significantly faster than JS for large pages | N/A |

### Key Finding from Puppeteer/Puppetaria

The [Puppetaria project](https://developer.chrome.com/blog/puppetaria/) found that
CDP-based accessibility tree queries had a "considerable performance gap" over
JavaScript-based approaches, with the difference "increasing dramatically with page
size." Puppeteer ultimately switched to CDP for accessibility queries.

---

## 4. AccName 1.2 Spec: The 4 Missing Cases

### Priority Order (Full Spec Algorithm)

Per the [W3C AccName 1.2 specification](https://w3c.github.io/accname/), step 2E
("host language label") is checked AFTER:

1. `aria-labelledby` (step 2B)
2. Embedded control value (step 2C)
3. `aria-label` (step 2D)

And BEFORE:

5. Name from content / text descendants (step 2F)
6. Tooltip / `title` attribute (step 2I)

This means `aria-label` and `aria-labelledby` ALWAYS win over legend, figcaption,
caption, and label.

### Case 1: `<fieldset>` + `<legend>`

**Spec:** [HTML-AAM: fieldset and legend elements](https://w3c.github.io/html-aam/#fieldset-and-legend-elements)

**Rule:** If the fieldset has no `aria-label` or `aria-labelledby`, compute its
accessible name from the text alternative of its first `<legend>` child element.

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Multiple `<legend>` elements | **First** `<legend>` direct child wins ([Issue #145](https://github.com/w3c/html-aam/issues/145)) |
| `<legend>` with rich content (`<strong>`, `<em>`, `<span>`) | Recursively compute text alternative of legend subtree |
| `<legend>` not a direct child | Does NOT count -- must be direct child of fieldset |
| Fieldset has `aria-label` | `aria-label` wins (handled in step 2D, before 2E) |
| Fieldset has `aria-labelledby` | `aria-labelledby` wins (handled in step 2B) |
| Hidden `<legend>` (`display:none`) | Browser-inconsistent. Firefox uses innerText (ignores hidden). Chrome may not provide name. Spec says hidden legend does not provide name. |
| No `<legend>` | Falls back to `title` attribute, then no name |

**Implementation pattern:**
```javascript
if (tagName === 'fieldset') {
    for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
        if (child.tagName.toLowerCase() === 'legend') {
            return computeTextAlternative(child);  // Recursive, not textContent
        }
    }
    return el.getAttribute('title') || '';
}
```

### Case 2: `<figure>` + `<figcaption>`

**Spec:** [HTML-AAM: figure and figcaption elements](https://w3c.github.io/html-aam/#figure-and-figcaption-elements)

**Rule:** If the figure has no `aria-label` or `aria-labelledby`, compute its
accessible name from the text alternative of its first `<figcaption>` child element.

**Important spec evolution:** [PR #359](https://github.com/w3c/html-aam/pull/359)
changed figure/figcaption from a "labelled by" to a "details for" relationship in
the spec. However, this change is recent (WebKit implemented it June 2025) and
browser support varies. For practical compatibility with what Chrome actually reports
in its accessibility tree, we should still compute figcaption as the name.

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Multiple `<figcaption>` elements | **First** figcaption direct child that is in the accessibility tree |
| Position of figcaption | Can be anywhere among direct children (first or last is common, but spec says first found) |
| Rich content in figcaption | Recursively compute text alternative |
| Figure has `aria-label` | `aria-label` wins |
| No figcaption | Falls back to `title` attribute |

**Implementation pattern:**
```javascript
if (tagName === 'figure') {
    for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
        if (child.tagName.toLowerCase() === 'figcaption') {
            return computeTextAlternative(child);
        }
    }
    return el.getAttribute('title') || '';
}
```

### Case 3: `<table>` + `<caption>`

**Spec:** [HTML-AAM: table element](https://w3c.github.io/html-aam/#table-element)

**Rule:** Compute the table's accessible name from the text alternative of its first
`<caption>` child element.

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Multiple `<caption>` elements | **First** caption direct child wins |
| Rich content in caption | Recursively compute text alternative |
| Table has `aria-label` | `aria-label` wins |
| Table has `aria-labelledby` | `aria-labelledby` wins |
| `<table summary="...">` | Not in spec, but all browsers support it as fallback. Playwright includes it. |
| Hidden caption | Does not provide accessible name |
| No caption | Falls back to `summary` attribute (non-spec), then `title` |

**Implementation pattern:**
```javascript
if (tagName === 'table') {
    for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
        if (child.tagName.toLowerCase() === 'caption') {
            return computeTextAlternative(child);
        }
    }
    // Non-spec but browser-compatible:
    if (el.getAttribute('summary')) return normalize(el.getAttribute('summary'));
    return el.getAttribute('title') || '';
}
```

### Case 4: Form Controls + `<label>`

**Spec:** [HTML-AAM: form element accessible name computation](https://w3c.github.io/html-aam/#input-type-text-input-type-password-input-type-number-input-type-search-input-type-tel-input-type-email-input-type-url-and-textarea-element-accessible-name-computation)

**Rule:** If no `aria-label` or `aria-labelledby`, compute the accessible name from
associated `<label>` elements. This covers both:
- **Explicit:** `<label for="controlId">` with matching `id`
- **Implicit:** `<label>` wrapping the control

**Applies to:** `<input>` (all types), `<select>`, `<textarea>`, `<button>`, `<output>`,
`<meter>`, `<progress>`

**The native `element.labels` API** handles both explicit and implicit association
automatically and is the recommended approach.

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Multiple labels for one control | Join all label texts with space |
| Label contains interactive elements | Recursive computation handles embedded controls |
| `aria-label` on the control | `aria-label` wins (step 2D before 2E) |
| `aria-labelledby` on the control | `aria-labelledby` wins (step 2B before 2E) |
| Label with `for` pointing to wrong ID | Label not associated, no name from it |
| Nested control in label reads own label | Playwright prevents infinite recursion with `isOwnLabel` check |
| No label found | Falls back to `placeholder` (for text inputs) or `title` |

**Implementation pattern:**
```javascript
if (['input', 'select', 'textarea'].includes(tagName)) {
    const labels = el.labels;  // Native API handles for= and wrapping
    if (labels && labels.length) {
        return Array.from(labels)
            .map(label => computeTextAlternative(label))
            .filter(Boolean)
            .join(' ');
    }
    // Fall through to placeholder/title
}
```

**Current implementation gap:** Our code already handles labels (steps 3-4 in our
current function), but uses `textContent` instead of recursive computation. This means
a label like `<label for="x">Name <abbr>*</abbr></label>` would include the abbr
content, which is correct, but a label like
`<label>Amount <input type="text" id="x"> dollars</label>` would incorrectly include
"Amount dollars" text content rather than computing the embedded input's value.

---

## 5. Other Tools and Libraries

### Puppeteer

**Approach:** Uses Chrome CDP `Accessibility.getFullAXTree`, not custom JavaScript.

The `page.accessibility.snapshot()` method returns the full accessibility tree with
pre-computed names from Chrome's native implementation. Puppeteer also supports
`element.computedName` and `element.computedRole` for individual elements.

The [Puppetaria initiative](https://developer.chrome.com/blog/puppetaria/) explicitly
chose CDP over JavaScript because:
1. Computing accessible names "is a non-trivial task"
2. They wanted to "reuse Chromium's existing infrastructure"
3. CDP approach had significantly better performance at scale

### Cypress

**Approach:** Delegates to **axe-core** (by Deque Systems) for accessibility analysis.
Does not compute accessible names directly. Instead, captures page state and runs
axe-core rules which internally compute names using their own implementation.

### dom-accessibility-api (npm)

**Package:** [`dom-accessibility-api`](https://github.com/eps1lon/dom-accessibility-api)
**Approach:** Full JavaScript implementation of AccName 1.2 in TypeScript.

This is the most thorough open-source JavaScript implementation:
- 9.8 million downstream dependents
- 153/159 browser tests passing
- Active maintenance (v0.7.1, November 2025)
- Used by Testing Library (`@testing-library/dom`)

Handles all 4 host-language cases:

```typescript
// Fieldset + Legend
if (isHTMLFieldSetElement(node)) {
    const children = ArrayFrom(node.childNodes);
    for (let i = 0; i < children.length; i++) {
        const child = children[i];
        if (isHTMLLegendElement(child)) {
            return computeTextAlternative(child, { ... });
        }
    }
}

// Table + Caption
if (isHTMLTableElement(node)) {
    const children = ArrayFrom(node.childNodes);
    for (let i = 0; i < children.length; i++) {
        const child = children[i];
        if (isHTMLTableCaptionElement(child)) {
            return computeTextAlternative(child, { ... });
        }
    }
}

// Form controls + Label
const labels = getLabels(node);
if (labels !== null && labels.length !== 0) {
    return ArrayFrom(labels)
        .map(element => computeTextAlternative(element, { isEmbeddedInLabel: true, ... }))
        .filter(label => label.length > 0)
        .join(" ");
}
```

NOTE: `dom-accessibility-api` does NOT handle figure/figcaption (confirmed in source).

### Google accname (npm)

**Package:** [`accname`](https://github.com/google/accname)
**Status:** **Archived** (August 2025). Read-only, no active development.

A simpler TypeScript implementation. Not recommended for new projects.

### aria-api (npm)

**Package:** [`aria-api`](https://github.com/xi/aria-api)
Provides `aria.getName(element)` for accessible name computation. Less widely adopted.

---

## 6. Implementation Recommendations

### Decision: Manual JS vs CDP

**Recommendation: Continue with manual JavaScript, but consider CDP as a future enhancement.**

Rationale:

| Factor | Manual JS (Current) | CDP |
|--------|-------------------|-----|
| **Immediate viability** | Just add the 4 missing cases | Requires architectural refactor |
| **Consistency** | Same approach for ARIA and Visual trees | Would need separate code paths |
| **Shadow DOM** | Our tree walker already handles it | CDP handles it natively |
| **Cross-frame** | Handled by our walker | Would need frame-by-frame CDP calls |
| **Maintenance** | We must track spec changes | Chrome tracks spec automatically |
| **Accuracy risk** | Medium -- we may have edge case bugs | Low -- Chrome's implementation is authoritative |

The 4 missing cases are straightforward to implement following Playwright's pattern.
A CDP refactor would be a much larger change with uncertain benefits for our snapshot
use case.

**Future consideration:** If accuracy issues persist after implementing these cases,
or if performance becomes a concern on large pages, switching to CDP's
`Accessibility.getFullAXTree` would give us Chrome's authoritative accessible names
for free. The `AXValueSource.nativeSource` field would even tell us exactly which
mechanism provided each name.

### Implementation Priority

Ordered by impact (how often users encounter unnamed elements):

| Priority | Case | Difficulty | Reason |
|----------|------|-----------|--------|
| **1** | Form controls + `<label>` | Low | Most common HTML pattern, already partially implemented |
| **2** | `<fieldset>` + `<legend>` | Low | Common in forms, simple first-child iteration |
| **3** | `<table>` + `<caption>` | Low | Common in data display, same pattern as fieldset |
| **4** | `<figure>` + `<figcaption>` | Low | Less common, but same implementation pattern |

### Implementation Approach

For our simplified (non-recursive) `computeAccessibleName`, the pragmatic approach is:

1. **Fix the priority order** to match the spec: aria-labelledby before aria-label
2. **Add the 4 cases** using `textContent` (our current approach) rather than recursive computation
3. **Use `element.labels` API** for form controls instead of manual querySelector

Using `textContent` instead of recursive computation is acceptable because:
- It matches what we do for every other name source already
- The edge cases where it differs (embedded controls in labels) are rare in practice
- It keeps the function simple and fast
- Playwright's recursive approach adds ~400 lines of complexity

### Concrete Implementation

```javascript
function computeAccessibleName(el) {
    // Step 2B: aria-labelledby (MUST be before aria-label per spec)
    if (el.getAttribute('aria-labelledby')) {
        const ids = el.getAttribute('aria-labelledby').split(/\s+/);
        const name = ids
            .map(id => {
                const refEl = document.getElementById(id);
                return refEl ? normalize(refEl.textContent) : '';
            })
            .filter(Boolean)
            .join(' ');
        if (name) return name;
    }

    // Step 2D: aria-label
    if (el.getAttribute('aria-label')) {
        return normalize(el.getAttribute('aria-label'));
    }

    const tagName = el.tagName.toLowerCase();

    // Step 2E: Host language label
    // Form controls: label element (explicit for= and wrapping)
    if (el.labels && el.labels.length) {
        return Array.from(el.labels)
            .map(label => normalize(label.textContent))
            .filter(Boolean)
            .join(' ');
    }

    // Fieldset: first legend child
    if (tagName === 'fieldset') {
        for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
            if (child.tagName.toLowerCase() === 'legend') {
                return normalize(child.textContent);
            }
        }
    }

    // Figure: first figcaption child
    if (tagName === 'figure') {
        for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
            if (child.tagName.toLowerCase() === 'figcaption') {
                return normalize(child.textContent);
            }
        }
    }

    // Table: first caption child
    if (tagName === 'table') {
        for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
            if (child.tagName.toLowerCase() === 'caption') {
                return normalize(child.textContent);
            }
        }
    }

    // Image: alt attribute
    if (tagName === 'img') {
        const alt = el.getAttribute('alt');
        if (alt != null) return normalize(alt);
    }

    // Name from content: buttons, links, headings
    if (['button', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
        return normalize(el.textContent);
    }

    // Input value/placeholder
    if (['input', 'textarea'].includes(tagName)) {
        return normalize(el.placeholder || el.value || '');
    }

    // Step 2I: Tooltip (title attribute) as last resort
    if (el.getAttribute('title')) {
        return normalize(el.getAttribute('title'));
    }

    return '';
}
```

### Additional Missing Name Sources (Beyond the 4 Cases)

Other accessible name sources not yet implemented:

| Source | Elements | Priority |
|--------|----------|----------|
| `<summary>` text content | `<details>` | Low -- details/summary relatively uncommon |
| `<optgroup label>` | `<optgroup>` | Low -- attribute, not child element |
| `<input type="submit">` default "Submit" | submit buttons | Low -- edge case |
| `<input type="image" alt>` | image inputs | Low -- rare element type |
| `<table summary>` | tables | Low -- deprecated attribute but browsers support |
| SVG `<title>` | SVG elements | Medium -- depends on SVG usage |
| `value` for input type=button/submit/reset | button-like inputs | Medium -- already partially covered |
| CSS `::before`/`::after` content | Any element | Low -- only matters for name-from-content |

---

## Sources

### W3C Specifications
- [AccName 1.2 (latest)](https://w3c.github.io/accname/)
- [HTML-AAM 1.0 (latest)](https://w3c.github.io/html-aam/)
- [HTML-AAM Issue #145: fieldset/legend clarification](https://github.com/w3c/html-aam/issues/145)
- [HTML-AAM PR #146: Clarify legend/figcaption/caption](https://github.com/w3c/html-aam/pull/146)
- [HTML-AAM PR #359: Change figure/figcaption to details relationship](https://github.com/w3c/html-aam/pull/359)
- [HTML-AAM Issue #325: Should figcaption participate in name/desc?](https://github.com/w3c/html-aam/issues/325)

### Playwright
- [Playwright roleUtils.ts](https://github.com/microsoft/playwright/blob/main/packages/injected/src/roleUtils.ts) -- Full accessible name implementation (~1235 lines)
- [Playwright ariaSnapshot docs](https://playwright.dev/docs/aria-snapshots)
- [Issue #33644: getByRole + label + invalid aria-labelledby](https://github.com/microsoft/playwright/issues/33644)

### Chrome CDP
- [CDP Accessibility Domain](https://chromedevtools.github.io/devtools-protocol/tot/Accessibility/)
- [Full accessibility tree in Chrome DevTools](https://developer.chrome.com/blog/full-accessibility-tree)

### Puppeteer
- [Puppetaria: accessibility-first Puppeteer scripts](https://developer.chrome.com/blog/puppetaria/)
- [Puppeteer Accessibility.snapshot()](https://pptr.dev/api/puppeteer.accessibility.snapshot)

### JavaScript Libraries
- [dom-accessibility-api](https://github.com/eps1lon/dom-accessibility-api) -- Most complete JS implementation (9.8M dependents)
- [accname (Google, archived)](https://github.com/google/accname) -- TypeScript AccName library
- [aria-api](https://github.com/xi/aria-api) -- Alternative JS implementation

### Selenium
- [Selenium CDP documentation](https://www.selenium.dev/documentation/webdriver/bidi/cdp/)
- [Selenium Chrome DevTools Protocol guide](https://applitools.com/blog/selenium-chrome-devtools-protocol-cdp-how-does-it-work/)
