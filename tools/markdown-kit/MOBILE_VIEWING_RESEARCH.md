# Mobile / Responsive Viewing Research

Deep research on making markdown-kit's `--serve` mode readable on mobile devices (iPhone, iPad, Android), plus assessment of alternative delivery formats.

**Date**: March 2026
**Context**: markdown-kit serves live HTML preview at a local URL (e.g., `http://192.168.4.30:64556`). On iPhone, the preview is unreadable because: no viewport meta tag, body is fixed at `width: 1400px`, and Mobile Safari zooms out to fit the full width into a ~390px screen.

---

## Table of Contents

- [1. The Viewport Meta Tag Fix](#1-the-viewport-meta-tag-fix)
- [2. Responsive CSS for Serve Mode](#2-responsive-css-for-serve-mode)
- [3. What Documentation Tools Do](#3-what-documentation-tools-do)
- [4. The Dual-Mode Question](#4-the-dual-mode-question)
- [5. Reader Mode](#5-reader-mode)
- [6. PWA / App-Like Experience](#6-pwa--app-like-experience)
- [7. Alternative Delivery Formats](#7-alternative-delivery-formats)
- [8. Recommended Implementation Plan](#8-recommended-implementation-plan)

---

## 1. The Viewport Meta Tag Fix

### The Core Problem

Without a viewport meta tag, mobile browsers assume the page is designed for a ~980px desktop viewport and zoom out to fit it. Our 1400px body width makes this even worse -- the text renders at roughly 28% of readable size on a 390px iPhone screen.

### The Standard Fix

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

This is sufficient for the vast majority of use cases. It tells the browser:
- `width=device-width`: Set the viewport width to the device's screen width (e.g., 390px on iPhone 15)
- `initial-scale=1`: Start at 1:1 zoom (no scaling)

### What Dev Servers Use

| Tool | Viewport Meta Tag |
|------|-------------------|
| **Vite** | None by default (user provides index.html) |
| **Next.js** | `width=device-width, initial-scale=1` (auto-generated) |
| **Create React App** | `width=device-width, initial-scale=1` |
| **Docusaurus** | `width=device-width, initial-scale=1` |
| **VitePress** | `width=device-width, initial-scale=1` |
| **MkDocs Material** | `width=device-width, initial-scale=1` |

Every documentation framework uses the standard tag. None add `viewport-fit=cover` by default.

### viewport-fit=cover and iPhone Notch/Dynamic Island

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
```

**When to use it**: Only when you want content to extend into the safe area insets (behind the notch/Dynamic Island and home indicator bar). This is for apps that want edge-to-edge layouts.

**When NOT to use it**: For document reading. We want the browser's default safe area behavior -- content stays inside the readable area. Adding `viewport-fit=cover` without corresponding `env(safe-area-inset-*)` padding would cause text to be hidden behind the notch.

**Recommendation**: Do NOT add `viewport-fit=cover`. The standard meta tag is correct for document viewing.

### Anti-Patterns to Avoid

```html
<!-- DO NOT USE: prevents user zooming, WCAG 1.4.4 failure -->
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
```

Note: Next.js's default viewport includes `maximum-scale=1, user-scalable=no`, which is an accessibility anti-pattern. We should not copy this. Users reading technical documents absolutely need pinch-to-zoom capability, especially for diagrams and tables.

WCAG SC 1.4.4 (Level AA) requires that text can be zoomed to at least 200%. Setting `user-scalable=no` or `maximum-scale=1` is a compliance failure.

### New CSS Viewport Units (dvh, svh, lvh)

Since June 2025, the new dynamic viewport units are Baseline Widely Available:

- **svh** (Small Viewport Height): viewport height with mobile browser chrome visible
- **lvh** (Large Viewport Height): viewport height with mobile browser chrome hidden
- **dvh** (Dynamic Viewport Height): adjusts as browser chrome appears/disappears

Supported in Chrome 108+, Firefox 101+, Safari 15.4+, Edge 108+. Roughly 95% of global users.

**Relevance to markdown-kit**: These are useful for full-screen layouts (hero sections, modals), not for scrollable documents. Standard `vh` is fine for our use case. The main value is knowing they exist if we ever add a full-screen presentation mode.

### Recommended Viewport Tag for markdown-kit

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

Nothing more. No `viewport-fit`, no `maximum-scale`, no `user-scalable`. Simple and correct.

---

## 2. Responsive CSS for Serve Mode

### The Body Width Problem

Current CSS:
```css
body {
  width: 1400px;
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px;
}
```

This hard-codes a 1400px width that forces mobile browsers to zoom out. The fix:

```css
body {
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px;
  box-sizing: border-box;
}

@media (max-width: 767px) {
  body {
    padding: 16px;
  }
}
```

**Critical distinction**: This responsive CSS should ONLY apply in serve mode. PDF generation still needs the fixed `width: 1400px` for deterministic layout. The `buildHtml({ forServe: true })` path must inject different CSS.

### Tables

Tables are the most common source of mobile overflow in technical documentation. Every documentation platform struggles with this.

**Recommended approach**: Horizontal scroll wrapper.

```css
/* Serve mode responsive overrides */
@media (max-width: 767px) {
  /* Responsive table wrapper — tables get horizontal scroll */
  table {
    display: block;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    white-space: nowrap;
  }
}

/* Alternative: always allow table scroll on any viewport */
.table-wrapper {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  margin: 16px 0;
}
.table-wrapper table {
  min-width: 100%;
}
```

**Why horizontal scroll, not reflowing**: Tables encode two-dimensional relationships. Reflowing table rows vertically (stacking columns) destroys the ability to cross-reference data. Every major documentation tool (GitHub, Docusaurus, MkDocs, MDN) uses horizontal scroll for wide tables. This is the established pattern.

**Scroll indicator**: Mobile browsers hide scrollbars by default, so users may not realize a table is scrollable. A subtle gradient fade at the right edge can hint at scrollable content:

```css
@media (max-width: 767px) {
  .table-scroll-wrapper {
    position: relative;
  }
  .table-scroll-wrapper::after {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 24px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.8));
    pointer-events: none;
  }
  /* Hide indicator when scrolled to the end (requires JS) */
  .table-scroll-wrapper.scrolled-end::after {
    display: none;
  }
}
```

### Code Blocks

Two viable approaches, each with trade-offs:

**Option A: Horizontal scroll (preferred for code)**
```css
pre {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
pre code {
  white-space: pre;  /* preserve formatting */
}
```

**Option B: Word wrap**
```css
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
}
```

**Recommendation**: Horizontal scroll for code blocks. Code is semantically different from prose -- line breaks matter. Wrapping Python or JavaScript at arbitrary character boundaries destroys readability and can make indentation-sensitive code misleading. This matches what GitHub, MDN, Docusaurus, and MkDocs all do.

On small devices the code is harder to read regardless, but horizontal scroll preserves the structural integrity of the code.

### Images

Current CSS already handles this:
```css
img {
  max-width: 100%;
  height: auto;
}
```

This is sufficient. Images scale down to fit the viewport. No additional mobile-specific handling needed.

### Mermaid Diagrams and Graphviz SVGs

SVG diagrams present a special challenge because they can be very wide (especially sequence diagrams and flowcharts with many nodes).

```css
/* Already in place */
.graphviz-diagram svg,
.vega-lite-chart svg {
  max-width: 100%;
  height: auto;
}

/* Mermaid diagrams */
.mermaid svg {
  max-width: 100%;
  height: auto;
}

/* For very complex diagrams, allow horizontal scroll */
@media (max-width: 767px) {
  .mermaid,
  .graphviz-diagram,
  .vega-lite-chart {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  /* Prevent SVG from being squished too small to read */
  .mermaid svg,
  .graphviz-diagram svg {
    min-width: 500px;  /* force scroll for complex diagrams */
  }
}
```

**The tension**: Scaling a complex sequence diagram to 390px wide makes text illegible. But allowing horizontal scroll means the user can't see the full diagram at once. The best compromise:

1. Simple diagrams (< 600px natural width): scale to fit with `max-width: 100%`
2. Complex diagrams (> 600px natural width): set a `min-width` and allow horizontal scroll
3. Let the SVG's natural width determine which path it takes

### KaTeX Math Equations

KaTeX does NOT support responsive line-breaking of equations. This is a known limitation (GitHub issue #327, open since 2016, still unresolved).

**Practical solutions**:

```css
/* Allow horizontal scroll for long equations */
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
  padding: 4px 0;  /* prevent vertical clipping */
}

/* Prevent equation overflow from breaking page layout */
.katex-display > .katex {
  max-width: 100%;
}
```

This matches what Docusaurus does (see issue #9785 -- KaTeX in tables causes overflow and responsiveness issues). There is no good solution for automatically breaking long equations across lines. Horizontal scroll is the only viable approach.

### Font Size Adjustments

Modern CSS fluid typography using `clamp()` can improve readability:

```css
@media (max-width: 767px) {
  body {
    /* Slightly larger base font for touch devices */
    font-size: clamp(15px, 4vw, 18px);
    line-height: 1.7;
  }

  h1 { font-size: clamp(22px, 6vw, 32px); }
  h2 { font-size: clamp(19px, 5vw, 26px); }
  h3 { font-size: clamp(17px, 4.5vw, 22px); }

  /* Code blocks: slightly smaller to fit more per line */
  pre code, code {
    font-size: 13px;
  }
}
```

**Key principle**: Body text should be 16px minimum on mobile for comfortable reading (this is Apple's recommendation and the default in Safari). Slightly larger (17-18px) is even better for serif fonts.

### Complete Responsive CSS Block for Serve Mode

```css
/* ═══════════════════════════════════════════════════════════════
   RESPONSIVE OVERRIDES — serve mode only (not injected into PDF HTML)
   ═══════════════════════════════════════════════════════════════ */

/* Fluid body width instead of fixed */
body {
  width: 100%;
  max-width: 1400px;  /* matches PDF layout width */
  margin: 0 auto;
  padding: 40px;
  box-sizing: border-box;
}

/* ── Tablet breakpoint ──────────────────────────────────────── */
@media (max-width: 1024px) {
  body {
    padding: 24px;
  }
}

/* ── Mobile breakpoint ──────────────────────────────────────── */
@media (max-width: 767px) {
  body {
    padding: 16px;
    font-size: 16px;
    line-height: 1.7;
  }

  /* Tables: horizontal scroll */
  table {
    display: block;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  /* Code blocks: horizontal scroll, not wrap */
  pre {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    font-size: 13px;
  }

  /* Math equations: horizontal scroll */
  .katex-display {
    overflow-x: auto;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
    padding: 4px 0;
  }

  /* Diagrams: allow scroll for complex ones */
  .mermaid,
  .graphviz-diagram,
  .vega-lite-chart {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  /* TOC: tighter padding */
  .md-toc {
    padding: 8px 12px;
  }

  /* Headings: fluid sizing */
  h1 { font-size: 1.6em; }
  h2 { font-size: 1.35em; }
  h3 { font-size: 1.15em; }
}
```

---

## 3. What Documentation Tools Do

### Summary Matrix

| Feature | GitHub | Docusaurus | MkDocs Material | VitePress | GitBook | MDN |
|---------|--------|------------|-----------------|-----------|---------|-----|
| **Viewport tag** | Standard | Standard | Standard | Standard | Standard | Standard |
| **Mobile breakpoint** | 767px | 996px | 960px | 960px | ~768px | 769px |
| **Table strategy** | Horiz scroll | `display:block` + scroll | Scroll wrapper | Scroll wrapper | Horiz scroll | Horiz scroll |
| **Code block strategy** | Horiz scroll | Horiz scroll | Horiz scroll | Horiz scroll | Horiz scroll | Horiz scroll |
| **Math overflow** | N/A | Horiz scroll | Plugin-dependent | Plugin-dependent | N/A | N/A |
| **Max content width** | 980px | 1440px | 1220px | 688px | ~750px | 1440px |
| **Mobile padding** | 15px | 16px | 12px | 24px | 16px | 16px |
| **Font scale on mobile** | Same | Same | Same | Same | Same | Slightly larger |
| **Dark mode** | System pref | Toggle | Toggle | Toggle | Toggle | Toggle + system |

### Key Patterns Observed

1. **Every platform uses horizontal scroll for tables and code**. None reflow table data vertically. None word-wrap code.

2. **Breakpoint consensus is 768px** (or close to it). Some use higher breakpoints (960-996px) for sidebar collapse, but content responsiveness kicks in around 768px.

3. **Max content width ranges from 688-1440px**. VitePress is notably narrow (688px) which makes prose very readable but code blocks tight. GitHub uses 980px as a balanced middle ground.

4. **Nobody scales fonts differently on mobile** (with minor exceptions). The base font is designed to be readable at all sizes, and users can pinch-to-zoom. This is the right approach for documents.

5. **All use `max-width` with `width: 100%`**, never a fixed pixel width. This is the fundamental pattern we need to adopt.

### GitHub's Approach (github-markdown-css)

GitHub's markdown CSS is the most battle-tested responsive markdown stylesheet. The open-source `github-markdown-css` package (sindresorhus) extracts it:

- `max-width: 980px` on desktop
- `padding: 45px` on desktop, `15px` on mobile (767px breakpoint)
- Tables use `display: block; overflow: auto;`
- Code uses `overflow: auto; white-space: pre;`
- Images: `max-width: 100%`
- Single breakpoint at 767px

This is essentially what we need, adapted to our 1400px width and theme system.

### Docusaurus Specifics

- Uses the Infima CSS framework
- 996px breakpoint separates mobile/desktop (sidebar hidden on mobile)
- Tables get `display: block` by default, which enables overflow scroll
- Code blocks have known issues with long lines causing layout breaks (issue #3870)
- KaTeX + tables cause overflow issues (issue #9785) -- they recommend `overflow-x: auto` on `.katex-display`

### MkDocs Material Specifics

- Most polished mobile experience among open-source doc tools
- Content area auto-adapts to available screen estate
- Navigation tabs collapse to menu layer below header on mobile
- Code blocks support annotations that work well on mobile touch
- Tables are scrollable by default
- Grids stretch to full viewport width on narrow screens

---

## 4. The Dual-Mode Question

**Should serve mode have two viewing modes (desktop vs mobile)?**

### Analysis

The question is whether to provide a toggle between "PDF-faithful desktop layout" and "responsive mobile-optimized layout."

**Arguments for dual mode**:
- Some users want to preview exactly what the PDF will look like (pixel-accurate)
- Mobile users want readable content
- These are fundamentally different goals

**Arguments against dual mode**:
- Responsive CSS handles this automatically via media queries
- A toggle adds UI complexity for no user benefit
- No documentation platform offers this toggle
- If the responsive CSS is well-designed, it "just works" on both desktop and mobile

### Recommendation: No Toggle. Use Responsive CSS.

Media queries already give us both modes automatically:
- Desktop browser (>767px): content displays at full width, close to PDF layout
- Mobile browser (<=767px): content reflows for readability

No manual toggle needed. The viewport width IS the mode selector.

If a user on desktop wants to see "exactly what the PDF looks like," they can simply make their browser window 1400px wide. The `max-width: 1400px` CSS ensures the desktop view matches PDF layout at full width.

**URL parameter approach (e.g., `?mobile=true`)**: Not recommended. This is unnecessary complexity. The `@media` query approach is the standard and requires zero user interaction.

---

## 5. Reader Mode

### Should we add a built-in reader mode?

**What Safari Reader Mode does**:
- Strips navigation, sidebars, ads, footers
- Presents just the article content
- Adjustable font size, background color, font family
- Available when Safari detects article-like content (>350 chars in semantic HTML)

**Our situation**: markdown-kit's serve mode already IS a reader mode. There is no navigation, no sidebar, no ads, no footer. It is pure content. Adding a "reader mode" toggle would be redundant.

**What we could add instead**:
- **Dark mode toggle**: A simple button or keyboard shortcut that switches the CSS color scheme. This is genuinely useful for reading on phone screens at night.
- **Font size controls**: `+` / `-` buttons for adjusting text size without pinch-to-zoom. This is a mild convenience but not necessary -- users can zoom.

**Making Safari Reader Mode work with our content**: Safari activates Reader Mode when it detects article-like content. We should ensure our HTML structure cooperates:
- Wrap content in `<article>` element
- Use proper heading hierarchy
- Have >350 characters of text content

```html
<body>
  <article>
    ${resolvedBody}
  </article>
</body>
```

This lets users opt into Safari's native Reader Mode if they want it, without us building our own.

### Recommendation

Do NOT build a custom reader mode. Instead:
1. Wrap body content in `<article>` so Safari Reader Mode activates
2. Optionally add a dark mode toggle (CSS custom properties make this trivial)
3. Let the responsive CSS do its job

---

## 6. PWA / App-Like Experience

### Is PWA worth it for markdown-kit's serve mode?

**What PWA would add**:
- Add to Home Screen with custom icon and name
- Full-screen mode (no Safari chrome)
- Offline caching via service worker
- Native-feeling scrolling

**Assessment: Not worth it.**

markdown-kit's serve mode is a transient dev-time preview tool. It runs while you are editing a markdown file and stops when you Ctrl+C. This is fundamentally incompatible with PWA's value proposition:

1. **Add to Home Screen**: Users don't want to "install" a preview of a markdown file they're actively editing. The URL changes every time (ephemeral port).

2. **Offline support**: The server IS the content source. If the server is down, there is nothing to cache. The file is on the local machine.

3. **Full-screen mode**: Nice but not necessary. Safari's address bar auto-hides on scroll.

4. **Service worker complexity**: Adds code to maintain for zero benefit in our use case.

**The one exception**: If markdown-kit ever generates a standalone HTML file (not served, but saved), PWA features in that HTML file could be valuable. A self-contained HTML document with offline capability is essentially a portable document format -- but that is EPUB territory, not PWA territory.

### Recommendation

Skip PWA entirely. It is overengineered for a local preview server.

---

## 7. Alternative Delivery Formats

### The real question: How to get rendered markdown onto a phone?

Five approaches, ranked by practicality for AI-to-human document delivery:

### 7.1. Responsive HTML via serve mode (recommended)

**What it is**: Our current `--serve` mode with responsive CSS fixes.

**Pros**:
- Already built (just needs responsive CSS)
- Live-reloading for editing workflow
- Full rendering fidelity (math, diagrams, code highlighting)
- Works on any device with a browser

**Cons**:
- Requires computer running the server
- Only works on same WiFi network
- Transient (no persistent document)

**Best for**: Active editing and review workflow.

### 7.2. Standalone HTML file

**What it is**: A self-contained `.html` file with all CSS, fonts, and images inlined/embedded.

**Pros**:
- No server needed
- Can be emailed, AirDrop'd, or shared via any file transfer
- Opens in any browser
- Can include responsive CSS
- All rendering preserved

**Cons**:
- File size can be large (fonts, images)
- No live-reloading
- Need to regenerate on every edit
- Images must be base64-encoded or referenced via absolute URL

**Implementation**: Add a `--html` output flag that generates a complete, self-contained HTML file instead of a PDF. This would use the same `buildHtml()` pipeline but inline all external resources.

**Best for**: Sharing finished documents for mobile reading.

### 7.3. EPUB

**What it is**: A reflowable e-book format natively supported by Apple Books, Google Play Books, and dedicated readers.

**Pros**:
- Purpose-built for reading on small screens
- Native reader apps with font/size controls, night mode, bookmarks
- Reflowable text adapts perfectly to any screen size
- Apple Books provides excellent reading experience on iPhone
- File size is small (compressed XML + CSS)

**Cons**:
- Limited support for advanced features:
  - No KaTeX math rendering (must rasterize or use MathML subset)
  - No Mermaid diagrams (must rasterize to PNG/SVG)
  - Limited code block styling (reader apps override CSS)
  - No Graphviz or Vega-Lite
- Pandoc is the standard tool (`pandoc input.md -o output.epub`) but has its own markdown parser (not marked.js)
- CSS control is limited -- reader apps override fonts and sizing
- Another dependency (pandoc) to manage

**Implementation**: Shell out to pandoc, or use a JS library like `epub-gen` (limited quality). Would need to pre-render all diagrams to images.

**Best for**: Long-form documents that are primarily prose (reports, guides, narratives). Poor fit for technical documentation with code, math, and diagrams.

### 7.4. Mobile-optimized PDF (--width 430)

**What it is**: Generate a PDF with body width set to ~430px, matching phone screen width.

**Pros**:
- No new tooling needed (just `--width 430`)
- PDF is a universal format
- Works offline, shareable
- Preserves exact layout

**Cons**:
- Text must be very small OR content must reflow, which changes layout significantly
- Tables, code blocks, diagrams still overflow or get unreadably small
- Pageless mode produces a very long, narrow document
- Paged mode (A4) makes no sense at 430px width
- Two PDFs (desktop + mobile) means maintaining two outputs
- The fundamental problem: PDFs are fixed-layout, phones need reflowed content

**Assessment**: This is the worst option. PDF's fixed-layout nature is antithetical to responsive design. A 430px-wide PDF is just a bad PDF.

### 7.5. Hosted HTML (deploy to web)

**What it is**: Generate the HTML and deploy to a URL (GitHub Pages, Netlify, S3, etc.)

**Pros**:
- Accessible from any device, any network
- No server to keep running
- Can include analytics, password protection, etc.

**Cons**:
- Deployment step adds complexity
- Overkill for sharing one document
- Requires internet access
- Privacy concerns for sensitive documents

**Best for**: Publishing documentation publicly or within a team.

### Format Comparison Matrix

| Feature | Serve Mode | Standalone HTML | EPUB | Mobile PDF | Hosted |
|---------|-----------|-----------------|------|-----------|--------|
| **Mobile readable** | Yes (with fixes) | Yes | Excellent | Poor | Yes |
| **Offline** | No | Yes | Yes | Yes | No |
| **Code blocks** | Full | Full | Limited | Squished | Full |
| **Math (KaTeX)** | Full | Full | Poor | OK | Full |
| **Diagrams** | Full | Full | Rasterized | Squished | Full |
| **File sharing** | No | Yes | Yes | Yes | URL |
| **Setup effort** | None (exists) | Low | Medium | None | Medium |
| **Ideal use** | Editing | Sharing | Prose reading | None | Publishing |

### Recommendation for AI-to-Human Document Delivery

**Primary**: Responsive serve mode (immediate fix) + standalone HTML export (medium-term).

The standalone HTML file is the sweet spot for AI-generated documents intended for phone reading:
1. AI generates markdown
2. markdown-kit renders to self-contained HTML with responsive CSS
3. HTML file is delivered to user (AirDrop, email, share link)
4. User opens in any browser on any device, fully rendered

This preserves all rendering features (math, diagrams, code) while providing a responsive reading experience. No server needed, no special reader app, no format conversion limitations.

---

## 8. Recommended Implementation Plan

### Phase 1: Immediate Fix (1-2 hours)

**Goal**: Make `--serve` mode readable on iPhone.

Changes to `markdown-kit.js`:

1. **Add viewport meta tag** in the `buildHtml()` function when `forServe` is true:
   ```html
   <meta name="viewport" content="width=device-width, initial-scale=1">
   ```

2. **Replace fixed body width** with responsive CSS in serve mode:
   ```css
   /* Only in serve mode HTML */
   body {
     width: 100%;
     max-width: 1400px;
     /* padding and margin stay the same */
   }
   ```

3. **Add mobile media query** with essential overrides:
   ```css
   @media (max-width: 767px) {
     body { padding: 16px; }
     table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
     pre { overflow-x: auto; -webkit-overflow-scrolling: touch; }
     .katex-display { overflow-x: auto; overflow-y: hidden; }
   }
   ```

4. **Wrap body content in `<article>`** for Safari Reader Mode compatibility.

**Implementation detail**: The responsive CSS is injected conditionally. When `forServe` is true, the body width CSS changes from `width: ${pageWidth}px` to `width: 100%; max-width: ${pageWidth}px`. The mobile media query is appended to the `<style>` block. PDF mode is completely unaffected.

### Phase 2: Polish (1-2 hours)

1. **Scroll indicators for tables**: Add subtle gradient fade to indicate scrollable content on mobile.

2. **Dark mode support**: Add a `<button>` in the serve mode HTML that toggles a `data-theme="dark"` attribute on `<html>`. CSS custom properties handle the color switching. Theme CSS files can define dark-mode colors.

3. **Diagram overflow handling**: Add `overflow-x: auto` on `.mermaid`, `.graphviz-diagram`, `.vega-lite-chart` containers at mobile breakpoint.

4. **Touch-friendly code blocks**: Ensure horizontal scrolling works smoothly on iOS (momentum scrolling via `-webkit-overflow-scrolling: touch`).

### Phase 3: Standalone HTML Export (half day)

Add `--html` flag to generate a self-contained HTML file:

```bash
node markdown-kit.js input.md --html
# Outputs: input.html with all CSS/fonts/images inlined
```

Implementation:
- Base64-encode referenced images
- Inline all CSS (theme, hljs, katex) into `<style>` tags
- Inline KaTeX fonts as base64 `@font-face`
- Include responsive CSS (same as serve mode)
- Include Mermaid JS bundle for client-side rendering
- Set viewport meta tag
- Output is a single `.html` file, zero external dependencies

### Phase 4: Optional Enhancements (lower priority)

1. **EPUB export**: Shell out to pandoc with pre-rendered diagrams. Useful for long prose documents but low priority given HTML's superiority for technical content.

2. **Font size controls in serve mode**: `+` / `-` buttons that adjust `--base-font-size` CSS variable. Mild convenience.

3. **QR code in terminal output**: When `--serve --host 0.0.0.0` starts, print a QR code in the terminal that encodes the network URL. User can scan with iPhone camera to open directly. Uses a small QR library (e.g., `qrcode-terminal`).

---

## Appendix A: CSS Reference for Responsive Technical Content

### Responsive Table Pattern (Complete)

```css
/* Wrapper approach for generated HTML */
.responsive-table-wrapper {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  margin: 16px 0;
  /* Accessibility: make focusable for keyboard scrolling */
}

.responsive-table-wrapper table {
  min-width: 100%;
  border-collapse: collapse;
}

/* Direct table styling (when wrapper isn't possible) */
@media (max-width: 767px) {
  table {
    display: block;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    white-space: nowrap;
  }

  /* Restore wrapping for cells with long text */
  td, th {
    white-space: normal;
    min-width: 120px;
  }
}
```

### Responsive Code Block Pattern

```css
pre {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  max-width: 100%;
}

pre code {
  white-space: pre;
  word-wrap: normal;
  overflow-wrap: normal;
}

@media (max-width: 767px) {
  pre {
    font-size: 13px;
    padding: 12px;
    border-radius: 6px;
  }
}
```

### Responsive Math (KaTeX) Pattern

```css
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
  padding: 4px 0;
  max-width: 100%;
}

/* Inline math: prevent overflow */
.katex {
  max-width: 100%;
  overflow-x: auto;
}
```

### Responsive SVG Diagram Pattern

```css
/* Scale SVGs to fit container */
.mermaid svg,
.graphviz-diagram svg,
.vega-lite-chart svg {
  max-width: 100%;
  height: auto;
}

/* On mobile, allow horizontal scroll for complex diagrams */
@media (max-width: 767px) {
  .mermaid,
  .graphviz-diagram,
  .vega-lite-chart {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
}
```

### Responsive Image Pattern (already correct)

```css
img {
  max-width: 100%;
  height: auto;
}
```

---

## Appendix B: Sources

### Viewport & Responsive Design
- [MDN: Viewport meta tag](https://developer.mozilla.org/en-US/docs/Web/HTML/Viewport_meta_tag)
- [MDN: Using the viewport meta element](https://developer.mozilla.org/en-US/docs/Web/HTML/Guides/Viewport_meta_element)
- [web.dev: Large, small, and dynamic viewport units](https://web.dev/blog/viewport-units)
- [Next.js: generateViewport](https://nextjs.org/docs/app/api-reference/functions/generate-viewport)

### Responsive Tables
- [CSS-Tricks: Under-Engineered Responsive Tables](https://css-tricks.com/under-engineered-responsive-tables/)
- [W3Schools: Responsive Tables](https://www.w3schools.com/howto/howto_css_table_responsive.asp)

### Code Blocks
- [Yihui Xie: CSS Trick for Horizontal Scrollbars in Code Blocks](https://yihui.org/en/2023/08/css-scrollbar/)
- [CSS-Tricks: Make Pre Text Wrap](https://css-tricks.com/snippets/css/make-pre-text-wrap/)
- [Docusaurus: Code block wrapping](https://tw-docs.com/docs/static-site-generators/docusaurus-code-wrap/)

### KaTeX Responsiveness
- [KaTeX Issue #327: Break/wrap formula if too wide](https://github.com/KaTeX/KaTeX/issues/327)
- [KaTeX Issue #455: Is KaTeX responsive?](https://github.com/KaTeX/KaTeX/issues/455)
- [Docusaurus Issue #9785: KaTeX overflow issues](https://github.com/facebook/docusaurus/issues/9785)

### Documentation Platform CSS
- [github-markdown-css](https://github.com/sindresorhus/github-markdown-css)
- [Docusaurus: Styling and Layout](https://docusaurus.io/docs/styling-layout)
- [MkDocs Material: Code Blocks](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/)
- [MkDocs Material: Data Tables](https://squidfunk.github.io/mkdocs-material/reference/data-tables/)

### Typography
- [CSS-Tricks: Linearly Scale font-size with CSS clamp()](https://css-tricks.com/linearly-scale-font-size-with-css-clamp-based-on-the-viewport/)

### Safari Reader Mode
- [Mandy Michael: Building websites for Safari Reader Mode](https://medium.com/@mandy.michael/building-websites-for-safari-reader-mode-and-other-reading-apps-1562913c86c9)
- [Mathias Bynens: How to enable Safari Reader on your site](https://mathiasbynens.be/notes/safari-reader)

### PWA
- [MDN: Making PWAs installable](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Tutorials/js13kGames/Installable_PWAs)
- [isitdev: PWA Guide 2025](https://isitdev.com/progressive-web-apps-pwa-guide-2025/)

### EPUB
- [Pandoc: Creating an ebook](https://pandoc.org/epub.html)
- [Learn By Example: Customizing pandoc for PDF and EPUB](https://learnbyexample.github.io/customizing-pandoc/)

### iPhone Notch / Safe Area
- [CSS-Tricks: The Notch and CSS](https://css-tricks.com/the-notch-and-css/)
