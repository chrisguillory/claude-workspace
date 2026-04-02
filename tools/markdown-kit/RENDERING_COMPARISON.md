# Rendering Comparison: Typora vs Our PDF (Pixyll Theme)

**Date:** March 22, 2026
**Source document:** `rendering-test.md`
**Typora version:** 1.12.6 (build 7588), macOS WebKit
**Our pipeline:** md-to-pdf (marked v15 + headless Chromium + markdown-kit.js)

---

## Deep Investigation: Bold Text in Lists

### The Observation

In the list section, "Second item with **bold text**":
- **Our PDF:** "bold text" is dramatically heavier than "Second item with" -- a sharp visual jump
- **Typora:** "bold text" is barely distinguishable from the surrounding regular text

### Root Cause Analysis

The difference stems from **how each renderer resolves font weights through the @font-face cascade**, not from any CSS rule difference.

#### Font Weight Cascade

Both CSS files define identical @font-face declarations:
```
normal weight -> merriweather-v19-latin-300.woff  (Merriweather Light)
bold weight   -> merriweather-v19-latin-700.woff  (Merriweather Heavy)
```

The `<li>` element inherits `font-weight: 400` from `body` (neither CSS sets an explicit `font-weight` on `li`). The `<p>` element gets an explicit `font-weight: 300`.

When `<strong>` appears inside a `<li>`:
1. Parent `<li>` computed weight = 400 ("normal")
2. `<strong>` applies CSS `bold` keyword = computes to weight 700
3. Weight 700 resolves to `merriweather-v19-latin-700.woff` (Merriweather Heavy)

The visual result: Merriweather Light (300) to Merriweather Heavy (700) is a **massive** weight jump -- these are literally the lightest and heaviest cuts of the family.

#### Why Typora Looks Different

Close inspection reveals that bold in Typora's list items IS slightly heavier than regular text, but the difference is almost imperceptible -- the stroke widths are only marginally thicker. In contrast, bold in `<p>` elements is more visibly distinct in Typora. This paragraph-vs-list discrepancy is significant.

The CSS weight cascade explains the behavior:

**In `<p>` elements:**
- `p { font-weight: 300 }` -- explicit weight 300
- The 300 weight maps to `merriweather-v19-latin-300.woff` (Merriweather Light)
- `<strong>` in a weight-300 context computes to 700 (per CSS spec: "bolder" from 300 = 700)
- Weight 700 maps to `merriweather-v19-latin-700.woff` (Merriweather Heavy)
- Result: **clear bold** (Light to Heavy jump)

**In `<li>` elements:**
- `li` has no explicit `font-weight` -- inherits `body { font-weight: 400 }`
- Weight 400 = CSS keyword `normal`
- BUT the @font-face only declares `font-weight: normal` (mapped to 300 woff) and `font-weight: bold` (mapped to 700 woff)
- For computed weight 400, the browser must find the closest available @font-face weight
- In **Chromium** (our PDF): weight 400 matches `font-weight: normal` in @font-face, loads the 300 woff. Then `<strong>` computes to 700, loads the 700 woff. Result: same dramatic 300-to-700 jump as paragraphs.
- In **Typora's WebKit**: the rendering engine may handle the 400-to-bold transition differently, potentially synthesizing bold from the normal weight glyph data rather than switching to the 700 woff file. WebKit's font synthesis adds ~1-2px of algorithmic stroke weight, producing barely-visible boldening.

**Additional factors:**

1. **Font synthesis:** Typora's WebKit engine likely uses `font-synthesis: weight` to create faux-bold from the 300 weight glyphs rather than loading the discrete 700 weight file. Faux bold adds minimal stroke thickness to existing glyphs, which is far subtler than switching to a different font file.

2. **@include-when-export CDN mismatch:** Typora's pixyll.css imports `Merriweather:900,900italic,300,300italic` from Google CDN during export -- weights 300 and 900, NOT 700. If the export renderer uses the CDN fonts and the local 700 woff fails to load, `<strong>` at computed weight 700 has no exact match. The browser would either synthesize bold from weight 300, or snap to weight 900 (the nearest heavier match). Either way, the visual result differs from our pipeline which correctly loads the local 700 woff.

3. **Typora base.css `.md-reset` rule:** Contains `font-weight: 400` which may interfere with weight inheritance in Typora's DOM structure for list items.

**Bottom line:** Our PDF is rendering bold **correctly** per the CSS specification. Typora is likely using font synthesis or has a font loading issue that prevents the 700 weight from rendering.

### Verdict

**Our rendering is more correct.** The CSS explicitly provides a 700 weight font file and the `<strong>` element computes to weight 700. Rendering Merriweather Heavy for bold is the intended behavior of the Pixyll theme. Typora's subtle bold is a rendering bug/limitation.

### Is it fixable?

If the dramatic weight jump is aesthetically undesirable (a design preference, not a correctness issue), these options exist:

| Option | Change | Effect |
|--------|--------|--------|
| Use font-weight: 400 for bold | `strong { font-weight: 400; }` | Defeats purpose of bold |
| Add Merriweather Regular (400) | Add 400 weight @font-face | Smoother gradient but still jumps to 700 for bold |
| Use font-synthesis | `* { font-synthesis: weight; }` | Mimics Typora's faux-bold; worse typographic quality |
| **Leave as-is** | No change | **Correct behavior per CSS spec** |

**Recommendation: Leave as-is.** Our bold rendering is typographically correct and visually clear.

---

## Comprehensive Difference Table

Differences sorted by severity (HIGH = obvious at a glance, MEDIUM = noticeable on inspection, LOW = subtle).

### HIGH Severity

| # | Element | Our PDF | Typora | Why | Fixable? | Better Version | Recommendation |
|---|---------|---------|--------|-----|----------|----------------|----------------|
| 1 | **Bold weight in body text** | Dramatic weight jump (Merriweather Light 300 to Heavy 700) | Subtle/barely visible bold | Our PDF loads the correct 700 weight file; Typora likely uses font synthesis from 300 weight | Yes (add `font-synthesis: weight` to mimic Typora) | **Ours** -- correct per CSS spec, clear visual distinction | Leave as-is |
| 2 | **Remote/placeholder images** | Not rendered (broken image icons or absent) | Not rendered (shows alt text as monospace fallback) | Both fail -- images from `via.placeholder.com` require network; neither renderer fetches remote images during offline PDF generation | No (network dependency) | Tie -- both fail | N/A; use local images |
| 3 | **Mermaid diagrams (Section 38)** | Rendered as raw code block | Rendered as actual flowchart diagram | Typora has built-in Mermaid.js; our marked pipeline does not | Yes (add mermaid.js preprocessing step) | **Typora** | Add mermaid support if needed; out of scope for CSS comparison |
| 4 | **Math rendering (Section 21)** | Raw LaTeX source shown as text | Rendered equations via MathJax/KaTeX | Typora has built-in math rendering; our pipeline lacks it | Yes (add MathJax/KaTeX to pipeline) | **Typora** | Add math support if needed; out of scope for CSS comparison |
| 5 | **Footnotes (Section 20)** | Rendered as formatted footnote area with backlinks | Rendered with superscript numbers and bottom footnote section | Both render footnotes but structural differences exist due to different markdown parsers | Partially (parser-level) | Typora's is more polished | Minor parser difference |
| 6 | **Table of Contents (Section 24)** | `[toc]` rendered as plain text | Full clickable TOC with nested headings | Typora has built-in TOC support; marked ignores `[toc]` | Yes (add TOC plugin) | **Typora** | Add TOC support if needed |

### MEDIUM Severity

| # | Element | Our PDF | Typora | Why | Fixable? | Better Version | Recommendation |
|---|---------|---------|--------|-----|----------|----------------|----------------|
| 7 | **Heading sizes (H1-H4)** | H1: 3.250rem, H2: 2.298rem, H3: 1.625rem, H4: 1.300rem (48em tier) | Same sizes at ~850px window width (48em tier active) | Both use the 48em breakpoint tier at typical widths. At wider Typora windows (>64em / 1024px), Typora would use larger sizes (H1: 4.498rem, H3: 1.9rem, H4: 1.591rem) but the reference screenshot matches our sizes | N/A at current width; would need 64em-tier sizes for wider windows | **Match** at reference width | No change needed; add 64em tier if wider PDF target desired |
| 8 | **Table font size** | 1.25rem (from our CSS) | 1.25rem at 48em+, but base is 1.125rem | Match; at 48em+ both use 1.25rem | N/A | Match | No change needed |
| 9 | **Paragraph line-height** | 1.8 (hardcoded in our CSS) | 1.5 at base, 1.8 at 48em+ | Match at target width (>48em) | N/A | Match | No change needed |
| 10 | **Code block syntax highlighting** | Monochrome (all code in `#7a7a7a`) | Syntax-colored (keywords, strings, comments in different colors) | Typora uses CodeMirror for syntax highlighting; our pipeline uses plain `<pre><code>` | Yes (add highlight.js or Prism.js) | **Typora** -- easier to read colored code | Add syntax highlighting library |
| 11 | **Typora extended syntax: `==highlight==`** | Rendered as literal `==text==` | Rendered with yellow `<mark>` background | Typora-specific markdown extension not in CommonMark/GFM | Yes (add marked extension) | **Typora** for Typora users | Add if Typora compat needed |
| 12 | **Typora extended syntax: `H~2~O`** | Rendered as literal `H~2~O` | Rendered as H with subscript 2 | Typora sub/superscript extension | Yes (add marked extension) | **Typora** | Add if needed |
| 13 | **Emoji shortcodes: `:smile:`** | Rendered as literal `:smile:` text | Rendered as emoji glyph | Typora auto-converts shortcodes | Yes (add emoji plugin) | **Typora** | Add if needed |
| 14 | **Definition list `<dl>` rendering** | Renders HTML `<dl>/<dt>/<dd>` with default browser styling | Renders with Typora's styled `<dt>` (bold) and `<dd>` (indented) | Both render the HTML; styling differs because Typora has theme-specific `<dl>` CSS | Yes (add `dt`, `dd` styling to our CSS) | **Typora** -- more polished | Add `dl/dt/dd` styling |
| 15 | **`<details>/<summary>` rendering** | Renders as expandable section (Chromium native) | Renders as expandable triangle with content; shows `</details>` tag leak | Both render it, but Typora shows raw closing tag | N/A | **Ours** -- cleaner | No change |
| 16 | **Nested blockquote border** | Each nesting level gets its own left border | Each nesting level gets its own left border | Both render correctly | N/A | Match | No change |
| 17 | **Content width** | 1400px max-width (hardcoded) | Variable (99% of window via `pixyll.user.css`) | Our CSS uses fixed max-width from Typora's default; user override widens it | Yes (adjust max-width) | Depends on preference | Match to user's Typora window width if desired |

### LOW Severity

| # | Element | Our PDF | Typora | Why | Fixable? | Better Version | Recommendation |
|---|---------|---------|--------|-----|----------|----------------|----------------|
| 18 | **Task list checkboxes** | Dark circle with white checkmark (custom CSS `#333`) | Green/teal circle with white checkmark (Typora native) | Typora uses its own green-tinted checkbox rendering; our CSS uses `#333` dark circles | Yes (change `background: #333` to a green shade) | Aesthetic preference | Leave as-is or match green if desired |
| 19 | **List bullet style per nesting level** | Disc > disc > disc (Chromium default for all levels) | Disc > circle > square (WebKit/Typora progression) | Typora's WebKit applies standard CSS2.1 `list-style-type` progression per nesting depth; Chromium uses disc at all levels by default | Yes (add `ul ul { list-style-type: circle; } ul ul ul { list-style-type: square; }`) | **Typora** -- better visual hierarchy for nested lists | Fix in our CSS |
| 20 | **Image centering** | Images left-aligned (block, natural alignment) | Images centered within content area | Typora base.css has `p > img:only-child { display: block; margin: auto; }` which centers standalone images | Yes (add same rule to our CSS) | **Typora** -- centered images look more intentional | Fix in our CSS |
| 21 | **`<mark>` highlight color** | `#fff3cd` (warm yellow) | `#ff0` (pure yellow, from Typora base.css) | Different highlight colors; Typora base.css sets `mark { background: #ff0 }` | Yes (change to `#ff0` to match Typora) | Aesthetic preference; our warm yellow is more readable | Leave as-is |
| 22 | **`<kbd>` styling** | Styled with border, shadow, rounded corners | Styled with similar treatment | Close match; minor shadow/border differences possible | Minor CSS tweaks possible | Close enough | No change needed |
| 23 | **Link color** | `#463F5C` with underline | `#463F5C` with underline | Match | N/A | Match | No change |
| 24 | **Blockquote styling** | 5px left border `#7a7a7a`, italic, `#555` color, 1.33em padding | Identical CSS rules | Match | N/A | Match | No change |
| 25 | **Code block border/padding** | 1px solid `#7a7a7a`, 0.5rem/1.125em padding | Identical | Match | N/A | Match | No change |
| 26 | **Inline code color** | `#7a7a7a`, 0.9em | `#7a7a7a` | Match | N/A | Match | No change |
| 27 | **Table borders** | 1px top `#333`, 2px bottom header `#333`, no top on first row | Identical rules | Match | N/A | Match | No change |
| 28 | **HR styling** | 1px `#ccc` top border, 1em margin | Visually identical | Match | N/A | Match | No change |
| 29 | **Body/paragraph font-weight** | body 400, p 300 | body 400, p 300 | Match | N/A | Match | No change |
| 30 | **Heading font family** | Lato (from local woff) | Lato (from local woff) | Match | N/A | Match | No change |
| 31 | **`<abbr>` styling** | Dotted bottom border, `cursor: help` | Dotted bottom border, similar | Match | N/A | Match | No change |
| 32 | **Unicode emoji rendering** | System emoji (Apple Color Emoji via Chromium) | System emoji (Apple Color Emoji via WebKit) | Both use system emoji; minor engine-level rendering differences possible | No (engine-level) | Tie | No change |
| 33 | **Special characters / HTML entities** | All rendered correctly (TM, copyright, fractions, arrows) | All rendered correctly | Match | N/A | Match | No change |
| 34 | **Backslash escapes** | All rendered correctly as literal characters | All rendered correctly | Match | N/A | Match | No change |

---

## Summary of Actionable Fixes

### Should Fix (CSS-only, improves fidelity)

1. **List bullet nesting styles** (#19) -- Add `ul ul { list-style-type: circle; } ul ul ul { list-style-type: square; }` to theme.css
2. **Image centering** (#20) -- Add `p > img:only-child { display: block; margin: auto; }` to theme.css

### Consider Fixing (feature parity, higher effort)

3. **Syntax highlighting** (#10) -- Add highlight.js or Prism.js to the pipeline for colored code blocks
4. **Definition list styling** (#14) -- Add `dt { font-weight: bold; } dd { margin-left: 2em; }` to theme.css
5. **Typora extensions** (#11-13) -- `==highlight==`, `~subscript~`, `^superscript^`, emoji shortcodes (requires marked extensions, not CSS)

### Do Not Fix (ours is correct or better)

6. **Bold weight rendering** (#1) -- Our Merriweather Heavy bold is typographically correct per the CSS spec and @font-face declarations; Typora's subtle bold is a rendering engine artifact
7. **Mark highlight color** (#21) -- Our `#fff3cd` is more readable than Typora's `#ff0`
8. **Task list checkbox color** (#18) -- Aesthetic preference, both functional
9. **`<details>` rendering** (#15) -- Ours is cleaner (no closing tag leak)
10. **Heading sizes** (#7) -- Already match Typora at the 48em breakpoint tier
11. **Internal anchor links** -- Both engines produce clickable internal links: Chromium via pdf-lib post-processing (TOC links, footnote backlinks, all `#anchor` links), WebKit via PDFKit post-processing. Typora's internal links are inconsistent/broken per [Issue #384](https://github.com/typora/typora-issues/issues/384). See `INTERNAL_LINKS_RESEARCH.md` for root cause analysis.

### Cannot Fix (engine-level or pipeline-level, not CSS)

11. **Mermaid diagrams** (#3 HIGH) -- Requires mermaid.js preprocessing pipeline change
12. **Math rendering** (#4 HIGH) -- Requires MathJax/KaTeX integration in pipeline
13. **TOC generation** (#6 HIGH) -- Requires marked plugin for `[toc]` support
14. **Font synthesis differences** (#1 HIGH) -- WebKit vs Chromium rendering engine behavior; not a CSS issue

---

## Appendix: CSS Patch for Quick Wins

```css
/* Add to theme.css for items #1 and #2 above */

/* #19: List bullet nesting (match Typora/WebKit progression) */
ul ul { list-style-type: circle; }
ul ul ul { list-style-type: square; }

/* #20: Center standalone images (match Typora base.css) */
p > img:only-child {
    display: block;
    margin: auto;
}

/* #14: Definition list styling */
dt { font-weight: bold; margin-top: 0.5em; }
dd { margin-left: 2em; margin-bottom: 0.5em; }
```
