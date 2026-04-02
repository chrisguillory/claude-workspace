# markdown-kit.js

Markdown to PDF rendering pipeline. Designed for AI-driven (CLI/programmatic) invocation, not human editing. Matches Typora's rendering quality in a headless pipeline with zero interactive dependencies.

**995 lines of Node.js** — single file, self-bootstrapping, no project-level node_modules.

## Quick Start

```bash
# Default: pageless PDF with Pixyll theme
node tools/markdown-kit/markdown-kit.js document.md

# GitHub theme
node tools/markdown-kit/markdown-kit.js document.md --theme github

# Normal A4 paged output
node tools/markdown-kit/markdown-kit.js document.md --no-pageless

# WebKit engine (macOS only, same engine as Typora)
node tools/markdown-kit/markdown-kit.js document.md --engine webkit
```

Output: `document.pdf` alongside the input file. Dependencies auto-install on first run to `~/.claude-workspace/tools/markdown-kit`.

---

## CLI Reference

```
Usage: node markdown-kit.js <input.md> [options]

Options:
  --theme, -t <name>        Theme: pixyll (default), github, gothic, newsprint, night, whitey
  --stylesheet, -s <path>   Custom CSS (overrides --theme)
  --width, -w <pixels>      Page width in pixels (default: 1400)
  --engine, -e <name>       PDF engine: chromium (default) or webkit
  --front-matter <mode>     How to handle YAML front matter: strip (default), render, raw
  --no-pageless             Use normal A4 pages instead of single continuous page
  --help, -h                Show this help
```

### Examples

```bash
# Pageless (single continuous page, auto-fit height)
node markdown-kit.js input.md

# Paged A4 with margins
node markdown-kit.js input.md --no-pageless

# Custom width (narrower)
node markdown-kit.js input.md --width 860

# Custom stylesheet (overrides --theme)
node markdown-kit.js input.md --stylesheet path/to/custom.css --width 860

# WebKit engine for pixel-match with Typora (macOS only)
node markdown-kit.js input.md --engine webkit

# GitHub theme
node markdown-kit.js input.md --theme github
```

---

## Architecture

```
                  markdown-kit.js (Node.js)
┌──────────────────────────────────────────────────────┐
│  1. Parse markdown (marked + extensions)             │
│  2. Generate HTML with embedded CSS theme             │
│  3. Resolve image paths to file:// URLs               │
│  4. Write temp HTML to disk                           │
│                                                      │
│  ┌──────────────────┐  ┌───────────────────────────┐ │
│  │ --engine chromium │  │ --engine webkit            │ │
│  │ Puppeteer/Blink   │  │ webkit-pdf (Swift binary)  │ │
│  │ Cross-platform    │  │ macOS WKWebView            │ │
│  │ 98% Typora match  │  │ ~99.9% Typora match        │ │
│  └──────────────────┘  └───────────────────────────┘ │
│                                                      │
│  5. Two-pass height measurement (pageless mode)       │
│  6. Generate PDF                                     │
│  7. Clean up temp files                              │
└──────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Headless, CLI-driven** — no editor, no GUI, no interactive mode
- **Puppeteer/Chromium default** — full CSS support, cross-platform, JavaScript execution for Mermaid/KaTeX
- **WebKit option** — 120-line Swift binary using WKWebView's `createPDF()` API, same engine as Typora
- **Self-bootstrapping** — dependencies install to persistent cache on first run; no setup steps
- **Pageless by default** — two-pass measurement: render with tall viewport, measure content bottom, re-render at exact height

---

## Markdown Feature Support

Comprehensive feature comparison across rendering targets. Every feature is exercised in `rendering-test.md`.

### Core Markdown (CommonMark / GFM)

| Feature | Syntax | Our Pipeline | Typora | Notes |
|---------|--------|:---:|:---:|-------|
| Headings (ATX) | `# H1` through `###### H6` | Yes | Yes | Match |
| Headings (Setext) | Underline with `===` / `---` | Yes | Yes | Match |
| Bold | `**text**` or `__text__` | Yes | Yes | Match |
| Italic | `*text*` or `_text_` | Yes | Yes | Match |
| Bold-italic | `***text***` | Yes | Yes | Match |
| Strikethrough | `~~text~~` | Yes | Yes | GFM extension |
| Inline code | `` `code` `` | Yes | Yes | Match |
| Fenced code blocks | ` ```lang ` | Yes | Yes | Match |
| Indented code blocks | 4-space indent | Yes | Yes | Match |
| Unordered lists | `- item` | Yes | Yes | Match |
| Ordered lists | `1. item` | Yes | Yes | Match |
| Nested lists | Indented sub-items | Yes | Yes | Match |
| Task lists | `- [x] done` / `- [ ] todo` | Yes | Yes | Checkbox styling differs (ours: dark circle; Typora: green/teal) |
| Tables | Pipe syntax with alignment | Yes | Yes | Match |
| Blockquotes | `> text` | Yes | Yes | Match |
| Nested blockquotes | `> > text` | Yes | Yes | Match |
| Links (inline) | `[text](url)` | Yes | Yes | Match |
| Links (reference) | `[text][ref]` | Yes | Yes | Match |
| Autolinks | `<url>`, bare URLs (GFM) | Yes | Yes | Match |
| Images | `![alt](src)` | Yes | Yes | Local images only (no network fetch in either) |
| Horizontal rules | `---`, `***`, `___` | Yes | Yes | Match |
| Line breaks | Trailing spaces or `\` | Yes | Yes | `breaks:true` matches Typora's `preLinebreakOnExport` |
| HTML inline | `<kbd>`, `<mark>`, `<sub>`, `<sup>`, `<u>`, `<abbr>`, `<del>`, `<ins>` | Yes | Yes | Match |
| HTML blocks | `<div>`, `<dl>`, `<details>` | Yes | Yes | Our `<details>` is cleaner (no closing tag leak) |
| Backslash escapes | `\*`, `\#`, etc. | Yes | Yes | Match |
| Entities | `&copy;`, `&#169;`, `&#x00A9;` | Yes | Yes | Match |

### Extended Features

| Feature | Syntax | Our Pipeline | Typora | Package / Method |
|---------|--------|:---:|:---:|-----------------|
| Syntax highlighting | ` ```python ` | Yes | Yes | highlight.js (GitHub theme) vs Typora's CodeMirror |
| Math (inline) | `$x^2$` | Yes | Configurable | marked-katex-extension + KaTeX |
| Math (block) | `$$..$$` | Yes | Configurable | marked-katex-extension + KaTeX |
| Mermaid diagrams | ` ```mermaid ` | Yes | Yes | mermaid.js (browser-injected) |
| Footnotes | `[^1]` / `[^1]:` | Yes | Yes | marked-footnote |
| Table of Contents | `[toc]` | Yes | Yes | marked-gfm-heading-id + custom replacement |
| Highlight | `==text==` | Yes | Yes | Custom marked inline extension |
| Subscript | `~text~` | Yes | Yes | Custom marked inline extension |
| Superscript | `^text^` | Yes | Yes | Custom marked inline extension |
| Emoji shortcodes | `:smile:` | Yes | Yes | marked-emoji + node-emoji |
| GFM Alerts | `> [!NOTE]`, `> [!TIP]`, etc. | **Yes** | **No** | marked-alert (we do MORE than Typora) |
| Definition lists | `<dl>` HTML | Yes | Yes | HTML pass-through + CSS styling |
| YAML front matter | `---` block at top | Yes | Yes | gray-matter; `--front-matter strip\|render\|raw` |

### Where We Intentionally Do More Than Typora

1. **GFM Alerts** — `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`, `> [!CAUTION]`, `> [!IMPORTANT]` render as styled callout boxes with colored borders and icons. Typora shows these as plain blockquotes.
2. **Math always enabled** — Typora's `enable_inline_math` is `false` on this machine. Our pipeline always renders math because the tool should handle any valid markdown.
3. **`<details>` rendering** — Our output is cleaner; Typora leaks the `</details>` closing tag as visible text.
4. **Bold weight rendering** — Our Merriweather Heavy 700 bold is typographically correct per the CSS @font-face spec. Typora's WebKit uses font synthesis, producing barely-visible bold. (See RENDERING_COMPARISON.md for analysis.)
5. **Internal anchor links** — Both engines produce clickable TOC links, footnote backlinks, and cross-references via post-processing (pdf-lib for Chromium, PDFKit for WebKit). Typora's internal links are inconsistent/broken per [Issue #384](https://github.com/typora/typora-issues/issues/384).

---

## Typora Preference Alignment

Audit of every rendering-affecting Typora preference on this machine against our pipeline behavior. Preferences read from `~/Library/Preferences/abnerworks.Typora.plist` (Typora 1.12.6, build 7588).

### Rendering-Affecting Preferences

| Typora Preference | Value on This Machine | Our Pipeline | Match? | Notes |
|---|---|---|:---:|---|
| `preLinebreakOnExport` | `true` | `marked({ breaks: true })` | Yes | Single newlines become `<br>`. Default for new Typora installs. |
| `strict_mode` | `true` | marked GFM mode (strict by default) | Yes | Disables loose-mode edge cases like lazy continuation. |
| `theme` | `"Github"` | `--theme github` available | Yes | User selects via CLI flag. |
| `darkTheme` | `"Pixyll"` | `--theme pixyll` (default) | Yes | Our default theme. |
| `enable_inline_math` | `false` | Always enabled (KaTeX) | Intentional | We render math regardless. Typora would show raw LaTeX with this setting. |
| `WebAutomaticDashSubstitution` | `false` | No smart dashes | Yes | Both output literal characters from source. |
| `WebAutomaticQuoteSubstitution` | `false` | No smart quotes | Yes | Both output literal characters from source. |
| `copy_markdown_by_default` | `true` | N/A | N/A | Clipboard behavior, editor-only. |
| `use_seamless_window` | `true` | N/A | N/A | Window chrome, editor-only. |
| `useSeparateDarkTheme` | `true` | N/A | N/A | Theme switching, editor-only. |
| `schemeAwareness` | `false` | N/A | N/A | System appearance, editor-only. |
| `send_usage_info` | `true` | N/A | N/A | Telemetry, editor-only. |

### Typora Syntax Extension Settings

Typora exposes these in Preferences > Markdown. The values below are inferred from the plist and Typora's defaults for `strict_mode: true`.

| Extension | Typora Default (Strict) | Our Pipeline | Match? |
|-----------|:---:|:---:|:---:|
| Highlight (`==text==`) | On | On (custom extension) | Yes |
| Subscript (`~text~`) | Off by default; user must enable | On (custom extension) | Intentional |
| Superscript (`^text^`) | Off by default; user must enable | On (custom extension) | Intentional |
| Inline math (`$...$`) | Off on this machine | On (KaTeX) | Intentional |
| Emoji (`:smile:`) | On by default | On (node-emoji) | Yes |
| Footnotes (`[^1]`) | On by default | On (marked-footnote) | Yes |
| TOC (`[toc]`) | On (built-in) | On (gfm-heading-id + custom) | Yes |
| Mermaid (` ```mermaid `) | On (built-in) | On (mermaid.js injected) | Yes |

---

## Marked Extensions

Every extension used in the pipeline, what syntax it handles, and the npm package.

| Extension | npm Package | Syntax Handled | Typora Equivalent |
|-----------|-------------|----------------|-------------------|
| GFM Heading IDs | `marked-gfm-heading-id` ^4 | Auto-generates `id` attributes on headings for anchor links | Built-in |
| Syntax Highlighting | `marked-highlight` ^2 + `highlight.js` ^11 | ` ```lang ` code blocks with language-specific coloring | Built-in (CodeMirror) |
| KaTeX Math | `marked-katex-extension` ^5 + `katex` ^0.16 | `$inline$` and `$$block$$` LaTeX math | Built-in (MathJax) |
| Footnotes | `marked-footnote` ^1 | `[^1]` references and `[^1]:` definitions | Built-in |
| GFM Alerts | `marked-alert` ^2 | `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`, `> [!CAUTION]`, `> [!IMPORTANT]` | **Not supported** |
| Emoji | `marked-emoji` ^2 + `node-emoji` ^2 | `:shortcode:` to Unicode emoji | Built-in |
| Highlight | Custom inline extension | `==text==` to `<mark>text</mark>` | Built-in |
| Subscript | Custom inline extension | `~text~` to `<sub>text</sub>` (single tilde only; `~~text~~` is strikethrough) | Built-in (optional) |
| Superscript | Custom inline extension | `^text^` to `<sup>text</sup>` | Built-in (optional) |
| Mermaid | `mermaid` ^11 (browser bundle) | ` ```mermaid ` code blocks rendered as SVG diagrams in-browser | Built-in |

---

## Parser Differences: marked vs Typora

Known structural differences between marked (our parser) and Typora's parser.

| Area | marked Behavior | Typora Behavior | Impact |
|------|-----------------|-----------------|--------|
| **Footnote HTML** | `<section class="footnotes">` with `<ol>` and backlink arrows | Custom DOM with superscript numbers and styled section | Visual difference in footnote area; both functional |
| **Task list output** | Bare `<li><input type="checkbox">` with no wrapper classes | Custom classes and styled checkboxes | Handled via CSS `:has()` selectors in our themes |
| **Line breaks** | `breaks: true` converts single newlines to `<br>` (must match Typora's `preLinebreakOnExport`) | `preLinebreakOnExport` controls this behavior | Match when both are `true` (default) |
| **GFM Alerts** | Requires `marked-alert` extension; renders as `<div class="markdown-alert">` | Not supported — renders as plain blockquote | We do more |
| **TOC** | `[toc]` rendered as `<p>[toc]</p>` then post-processed via regex replacement | Built-in TOC generation | Equivalent output |
| **Emoji** | Resolved at parse time via `node-emoji` lookup table | Resolved at render time | Same visual output |
| **Intra-word emphasis** | GFM mode: underscores in `file_name_with_underscores` do NOT trigger emphasis | Same behavior in strict mode | Match |
| **Mermaid** | Outputs as `<pre><code class="language-mermaid">`, then rewritten to `<div class="mermaid">` in-browser | Renders directly in WebKit | Same visual output |

---

## Engine Comparison: Chromium vs WebKit

| Aspect | `--engine chromium` (default) | `--engine webkit` (macOS only) |
|--------|------|--------|
| **Rendering engine** | Blink (via Puppeteer) | Apple WebKit (via WKWebView) |
| **Text shaping** | HarfBuzz | CoreText |
| **Rasterizer** | Skia | CoreGraphics |
| **Typora fidelity** | ~98% match | ~99.9% match (same engine) |
| **Cross-platform** | Yes (Linux, macOS, Windows) | macOS only |
| **JavaScript execution** | Full (Mermaid, KaTeX rendered in-browser) | HTML pre-rendered by Node; WebKit does PDF only |
| **PDF generation** | Puppeteer `page.pdf()` | WKWebView `createPDF()` + PDFKit post-processing |
| **Internal anchor links** | Yes (pdf-lib post-processing) | Yes (PDFKit annotations) |
| **Speed** | ~2-5 seconds | ~1-2 seconds |
| **Binary size** | ~300MB (bundled Chromium) | ~200KB (Swift binary) |
| **Known differences** | Sub-pixel font metrics, occasional line-break differences at paragraph boundaries | Near-identical to Typora |

### When to Use Each

- **Chromium (default)**: Cross-platform, full Mermaid/KaTeX rendering, clickable internal links, proven reliable
- **WebKit**: When pixel-matching Typora output is required. macOS only

### Why They Look Different

The engines use fundamentally different text pipelines. No CSS can fix this -- the difference is below the CSS level.

```
Chromium:  HTML → Blink → HarfBuzz (shaping) → Skia (rasterization) → PDF
WebKit:    HTML → WKWebView → CoreText (shaping) → CoreGraphics (raster) → PDF
```

Typical differences: 0.1-0.5px per glyph position, different kerning pairs, different line-break decisions at paragraph boundaries. Only visible in side-by-side comparison.

See `WEBKIT_RESEARCH.md` for the full engine analysis.

---

## Theme Support

### Bundled Themes

| Theme | Font Family | Style | Source |
|-------|-------------|-------|--------|
| **pixyll** (default) | Merriweather (serif) | Warm, readable; 300/700 weights, line-height 1.8 | Typora Pixyll by John Otander |
| **github** | Open Sans (sans-serif) | Clean, technical; regular/bold weights | Typora GitHub theme |
| **gothic** | TeXGyreAdventor (geometric sans) | Modern geometric; clean lines | Typora Gothic theme |
| **newsprint** | PT Serif (serif) | Classic newspaper style; traditional feel | Typora Newsprint theme |
| **night** | Helvetica Neue (sans-serif) | Dark theme; light text on dark background | Typora Night theme |
| **whitey** | Vollkorn / Palatino (serif) | Elegant book style; justified text, centered headings | Typora Whitey theme |

All themes are adapted from Typora's originals with these changes:
- `@include-when-export` removed (fonts loaded from local .woff/.woff2 files)
- `#write` selectors replaced with `body`
- `.md-fences` selectors replaced with `pre, pre > code`
- Editor-only selectors stripped (sidebar, focus mode, etc.)
- Task list selectors use `li:has(> input[type="checkbox"])` for marked v15 output

### Adding a New Theme

1. Copy a Typora theme CSS to `themes/<name>/theme.css`
2. Download the theme's web fonts as .woff/.woff2 into the same directory
3. Replace `@include-when-export` CDN imports with local `@font-face` declarations pointing to `url('./<font-file>')`
4. Replace Typora-specific selectors: `#write` to `body`, `.md-fences` to `pre`
5. Strip editor-only rules (sidebar, focus mode, quick-open, etc.)
6. Add task list CSS for `li:has(> input[type="checkbox"])` (copy from pixyll/theme.css)
7. Use: `node markdown-kit.js input.md --theme <name>`

For one-off custom CSS: `node markdown-kit.js input.md --stylesheet path/to/style.css`

---

## Dependency Management

### The Problem

Node.js has no equivalent of Python's `uvx` for one-shot dependency installation. `npx -p` is broken for this use case -- Node's module resolution (both ESM and CJS) resolves packages relative to the **script's** location, not npx's temp directory. This is fundamental to Node, not version-specific.

### The Solution: Persistent Cache + createRequire()

```
~/.claude-workspace/tools/markdown-kit/
└── node_modules/
    ├── puppeteer/       (+ bundled Chromium)
    ├── marked/
    ├── marked-highlight/
    ├── highlight.js/
    ├── mermaid/
    ├── katex/
    ├── marked-katex-extension/
    ├── marked-gfm-heading-id/
    ├── marked-emoji/
    ├── node-emoji/
    ├── marked-footnote/
    ├── marked-alert/
    ├── gray-matter/
    └── pdf-lib/
```

On first run:
1. Check if all packages exist in `~/.claude-workspace/tools/markdown-kit/node_modules/`
2. If any missing, run `npm install --prefix ~/.claude-workspace/tools/markdown-kit <all packages>`
3. Create a `require()` function anchored to the cache via `createRequire()`
4. Load all packages using the anchored `require()`

Subsequent runs skip installation entirely. Cache persists across sessions, projects, and Node versions.

### Updating Dependencies

```bash
rm -rf ~/.claude-workspace/tools/markdown-kit
node markdown-kit.js input.md   # reinstalls latest versions
```

### Current Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `puppeteer` | ^24 | Headless Chromium for PDF generation |
| `marked` | ^15 | Markdown parser (GFM) |
| `marked-highlight` | ^2 | Syntax highlighting bridge |
| `highlight.js` | ^11 | Language-specific code coloring |
| `mermaid` | ^11 | Diagram rendering (browser bundle injected into page) |
| `katex` | ^0.16 | Math typesetting (CSS + fonts) |
| `marked-katex-extension` | ^5 | KaTeX integration for marked |
| `marked-gfm-heading-id` | ^4 | Auto-generate heading IDs for TOC links |
| `marked-emoji` | ^2 | `:shortcode:` emoji resolution |
| `node-emoji` | ^2 | Emoji database (all GitHub/Typora shortcodes) |
| `marked-footnote` | ^1 | `[^1]` footnote syntax |
| `marked-alert` | ^2 | GFM alert/admonition callout boxes |
| `gray-matter` | ^4 | YAML front matter parsing |
| `pdf-lib` | ^1 | PDF post-processing for internal link annotations (Chromium engine) |

---

## Pageless Mode

The default mode generates a single continuous page with no page breaks. This is implemented via a two-pass approach:

1. **Pass 1**: Render HTML in a 50,000px tall viewport with print media emulation. Measure the `bottom` coordinate of the last visible child element.
2. **Pass 2**: Generate the PDF at the measured height + 10px padding.

In pageless mode, inline `page-break-*` and `break-*` CSS properties are stripped from the HTML before rendering, and `!important` overrides are applied via injected CSS.

For normal A4 paged output, use `--no-pageless`. Margins default to 25mm top/bottom, 20mm left/right.

---

## Internal Anchor Links (Chromium)

Neither Chromium's `Page.printToPDF` ([Chromium Issue #347674894](https://issues.chromium.org/issues/347674894)) nor WebKit's `createPDF()` generate PDF link annotations for internal `#anchor` links. We work around this via pdf-lib post-processing.

### How It Works

After Puppeteer generates the PDF, `markdown-kit.js`:

1. **Collects link data** via `page.evaluate()` (with print media emulation):
   - All `<a href="#...">` elements: source bounding rectangles and target IDs
   - All `[id]` elements: destination positions
   - Document height and viewport width

2. **Post-processes with pdf-lib**:
   - Loads the generated PDF bytes
   - For each internal link, creates a `/Link` annotation with a `/GoTo` destination
   - Coordinate conversion: CSS pixels * 0.75 = PDF points (96 DPI to 72 DPI), Y-axis flipped

3. **Handles both PDF modes**:
   - **Pageless**: Single page, direct coordinate mapping
   - **Paged (A4)**: Multi-page, computes page index from CSS Y position using content area height per page. X coordinates scaled to fit A4 content width.

### What Gets Linked

- TOC entries (`[toc]` generated links to headings)
- Footnote references (`[^1]` inline markers)
- Footnote backlinks (return arrows in the footnote section)
- Any other `<a href="#...">` in the document

### Graceful Degradation

If link collection or post-processing fails for any reason, the un-annotated PDF is still written. A warning is printed but the process does not exit with an error.

---

## WebKit Engine (macOS)

The `--engine webkit` option uses a Swift binary (`webkit-pdf`) that renders HTML via macOS native WKWebView -- the same engine Typora uses.

### Prerequisites

```bash
# Compile the Swift binary (one-time)
cd tools/markdown-kit
swiftc -o webkit-pdf webkit-pdf.swift -framework WebKit -framework AppKit -framework PDFKit
```

Requires Xcode.app (not just Command Line Tools) on macOS 26 beta due to a `SwiftBridging` module redefinition bug in the CLI Tools.

### How It Works

1. `markdown-kit.js` converts markdown to HTML (same as Chromium path)
2. Writes HTML to a temp file
3. Calls `webkit-pdf <temp.html> <output.pdf> --width <width>`
4. The Swift binary:
   a. Loads HTML in WKWebView, waits for fonts
   b. Collects all `<a href="#...">` link positions and `[id]` target positions via JavaScript
   c. Measures content height, calls `createPDF()`
   d. Post-processes the PDF with PDFKit: adds `PDFAnnotation` links for each internal anchor reference
5. Temp file cleaned up

### Internal Link Support (WebKit)

`WKWebView.createPDF()` does not preserve internal anchor links (this is an undocumented API limitation -- see `INTERNAL_LINKS_RESEARCH.md`). The Swift binary works around this by:

1. Injecting JavaScript to collect bounding rectangles of all `<a href="#...">` links and their `[id]` targets
2. After PDF generation, loading the PDF with Apple's PDFKit framework
3. Adding `PDFAnnotation` (`.link` subtype) with `PDFActionGoTo` destinations at the correct coordinates
4. Coordinate conversion: HTML origin is top-left (Y down), PDF origin is bottom-left (Y up). `createPDF()` maps 1 CSS pixel = 1 PDF point.

This makes TOC links, footnote backlinks, and any other internal cross-references clickable in the output PDF. If post-processing fails for any reason, the un-annotated PDF is still written.

### Limitations

- macOS only (requires AppKit + WebKit + PDFKit frameworks)
- No in-browser JavaScript execution for Mermaid diagrams (pre-rendered HTML only)
- KaTeX CSS/fonts are embedded in the HTML, so math renders correctly

---

## Related Documentation

| File | Contents |
|------|----------|
| `RENDERING_COMPARISON.md` | Detailed side-by-side comparison of 34 rendering differences between our pipeline and Typora, with root cause analysis and fix recommendations |
| `WEBKIT_RESEARCH.md` | Research report on WebKit vs Blink for PDF rendering, covering Playwright WebKit (dead end), wkhtmltopdf (deprecated), and the Swift WKWebView approach |
| `AI_NATIVE_DOCUMENT_GENERATION_LANDSCAPE.md` | Market analysis positioning this tool relative to markdown2pdf.ai, Pandoc, WeasyPrint, Typst, and the broader AI-native document generation space |
| `rendering-test.md` | Comprehensive markdown test document exercising every feature (40 sections, ~1200 lines) |
| `INTERNAL_LINKS_RESEARCH.md` | Research report on why `createPDF()` lacks internal links and the PDFKit post-processing solution |
| `webkit-pdf.swift` | Swift source for the macOS WebKit PDF backend with PDFKit internal link annotations |

---

## Rendering Audit Summary

The full rendering comparison is in `RENDERING_COMPARISON.md`. Key findings:

### Correct / Better Than Typora (Do Not Change)

- **Internal anchor links** — Both engines produce clickable TOC links, footnote backlinks, and cross-references (Chromium via pdf-lib, WebKit via PDFKit). Typora's internal links are inconsistent/broken (see `INTERNAL_LINKS_RESEARCH.md`)
- **Bold weight** — Our Merriweather Heavy 700 is correct per CSS spec; Typora's faux-bold is a rendering artifact
- **`<details>` rendering** — No closing tag leak
- **GFM Alerts** — Typora does not support this syntax at all
- **Highlight color** — Our `#fff3cd` is more readable than Typora's `#ff0`

### Fixed Since Initial Audit

- Internal anchor links in both engines (TOC, footnotes, cross-references): Chromium via pdf-lib post-processing, WebKit via PDFKit
- List bullet nesting (`disc > circle > square` progression)
- Image centering (`p > img:only-child`)
- Definition list styling (`dt`/`dd`)
- Syntax highlighting (highlight.js with GitHub theme)
- All Typora extensions (`==highlight==`, `~sub~`, `^sup^`, `:emoji:`)
- Mermaid diagrams
- KaTeX math rendering
- TOC generation
- Footnotes

### Remaining Differences (Engine-Level, Not CSS-Fixable)

- Sub-pixel font metrics (HarfBuzz vs CoreText)
- Line-break decisions at paragraph boundaries
- Font synthesis behavior for intermediate weights

Use `--engine webkit` on macOS to eliminate these.
