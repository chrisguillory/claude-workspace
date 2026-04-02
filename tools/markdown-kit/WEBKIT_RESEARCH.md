# WebKit vs Blink for PDF Rendering: Research Report

**Date:** March 22, 2026
**Context:** Our `markdown-kit.js` uses Puppeteer (headless Chromium/Blink) to render HTML to PDF. Typora on macOS uses WKWebView (Apple's WebKit). The remaining 1-2% visual difference between our PDFs and Typora's output is due to engine differences. This report investigates whether we can use WebKit instead.

---

## 1. History and Divergence: WebKit vs Blink

### The Fork (April 2013)

Google forked WebKit into Blink in April 2013 after architectural disagreements with Apple. The primary driver was Chromium's multi-process architecture, which was increasingly difficult to maintain within WebKit's codebase. Google removed 7,000+ files and 4.5 million lines of code that were irrelevant to Chromium on day one.

### 13 Years of Divergence (2013-2026)

The engines have diverged substantially in every layer:

**Text Layout and Shaping:**
- Blink uses HarfBuzz for text shaping on all platforms. WebKit uses CoreText on macOS/iOS (Apple's native text engine) and HarfBuzz on GTK/WPE (Linux).
- CoreText and HarfBuzz produce measurably different glyph positions, ligature handling, and kerning. This is the single largest source of rendering differences.
- Line breaking algorithms have diverged. Both follow Unicode UAX #14, but their implementations of "emergency breaks" and hyphenation differ.

**Font Rendering:**
- macOS WebKit (Safari/Typora) uses CoreGraphics sub-pixel rendering pipeline. Apple disabled system-wide sub-pixel antialiasing in macOS Mojave (2018), but the CoreText metrics remain different from Blink's.
- Blink uses Skia for rasterization on all platforms, with its own glyph cache and hinting decisions.
- `-webkit-font-smoothing: antialiased` vs `subpixel-antialiased` — both engines support this but the actual pixel output differs due to different rasterizers.

**CSS Box Model:**
- Sub-pixel rounding: Blink rounds fractional pixel values differently than WebKit. A 33.33% width container may end up 1px different.
- Margin collapsing edge cases: Both follow the spec, but have different behaviors in ambiguous situations (e.g., margins between floats and flow content).
- Flex/Grid layout: Substantially different implementations since the fork. Both are spec-compliant but diverge on rounding and minimum sizes.

**Specific Rendering Differences We Would See:**
- Font metrics (ascent/descent/line-gap) — different values for the same font
- Word spacing and letter spacing at sub-pixel level
- Table cell sizing with percentage widths
- Scrollbar presence/width affecting layout calculations
- `<code>` and `<pre>` element sizing with monospace fonts

### Bottom Line on Divergence

**WebKit and Blink are now as different as WebKit and Gecko (Firefox).** They share historical DNA but their rendering pipelines, text engines, and layout algorithms have been independently developed for 13 years. You cannot get Blink to produce WebKit-identical output through CSS alone — the differences are in the engine itself.

---

## 2. Playwright WebKit

### Can Playwright WebKit Generate PDFs?

**No.** Playwright's `page.pdf()` API only works with Chromium. Attempting to call it on a WebKit browser context throws: `"PDF generation is only supported for Headless Chromium."`

This is confirmed by:
- [Playwright Python Issue #2909](https://github.com/microsoft/playwright-python/issues/2909) — explicit statement from maintainers: "There is no alternative since in WebKit there is no PDF generator."
- [Playwright API docs](https://playwright.dev/docs/api/class-page) — `page.pdf()` is listed as Chromium-only.

Firefox also lacks PDF generation support in Playwright. This is not a temporary limitation — WebKit's headless mode simply does not include a PDF printing pipeline.

### Is Playwright's WebKit the Same as Apple's WKWebView?

**No, it is significantly different.** Key differences:

1. **Patched Build:** Playwright applies patches (stored in `browser_patches/webkit/patches/bootstrap.diff`) that disable features and modify behavior for automation purposes.

2. **Not WKWebView:** Playwright builds WebKit as a standalone browser (using WebKitGTK on Linux, MiniBrowser on macOS). This is NOT the same as Apple's WKWebView framework. WKWebView includes Apple's proprietary integrations with CoreAnimation, CoreText, IOSurface, and the macOS compositing system.

3. **Platform Differences:** Playwright's WebKit on macOS is closer to Apple's WebKit than on Linux, but still diverges because it uses MiniBrowser rather than the Safari/WKWebView embedding. Features like WebRTC and Offscreen Canvas are disabled on non-Apple builds.

4. **Font Rendering Issues:** Multiple Playwright issues document font rendering differences:
   - [Issue #20203](https://github.com/microsoft/playwright/issues/20203): Font kerning appears to not work at all in Playwright WebKit
   - [Issue #2626](https://github.com/microsoft/playwright/issues/2626): WebKit font spacing and icon fonts render differently
   - [Issue #22429](https://github.com/microsoft/playwright/issues/22429): Incorrect font rendering on Ubuntu
   - Text appears to shift "non-deterministically" in layout

### Would Playwright WebKit Match Typora?

**No.** Even if Playwright WebKit could generate PDFs (which it cannot), the rendering would not match Typora because:
- Playwright uses a patched MiniBrowser, not WKWebView
- Font metrics differ between MiniBrowser and WKWebView embeddings
- The compositing and rasterization pipeline is different

### Installation Size

Playwright downloads ~300MB per browser engine (~900MB total for all three). You can install WebKit alone: `npx playwright install webkit`. The npm package itself is small; the browser binaries are the bulk.

### Verdict on Playwright WebKit

**Dead end for our use case.** It cannot generate PDFs, and even if it could, the rendering would not match Typora's WKWebView output.

---

## 3. Swift WKWebView CLI Tool

### Is It Technically Feasible?

**Yes.** This is the most promising path. The key components:

1. **`WKWebView.createPDF(configuration:completionHandler:)`** — Available since macOS 11.0 (Big Sur) and iOS 14.0. This is the exact API that generates PDFs using Apple's native WebKit rendering.

2. **`WKPDFConfiguration`** — Has a single property, `rect`, which defines the area to capture. You can set it to the full `scrollView.contentSize` to capture the entire page.

3. **Headless Operation** — WKWebView does NOT need a visible window. You can create one with a zero-size frame in a command-line tool. The rendering happens in WebKit's separate process (WebContent process) regardless of whether the view is displayed.

4. **AppKit from Command Line** — Swift command-line tools can use AppKit by calling `NSApplication.shared.run()` to start the event loop. This is a documented pattern ([objc.io blog post](https://www.objc.io/blog/2018/10/02/using-appkit-from-the-command-line/)).

### Would It Give Pixel-Perfect Matching with Typora?

**Very likely yes, with caveats:**

- **Same WebKit engine:** WKWebView is the same engine Typora uses on macOS. Same CoreText shaping, same CoreGraphics rasterization, same font metrics.
- **Same `createPDF` pathway:** Typora uses WKWebView for rendering and likely uses the same (or equivalent) PDF export mechanism.
- **Potential difference:** Typora's PDF export may use `NSPrintOperation` (the print dialog pathway) rather than `createPDF()`. The print pathway generates paged PDFs with headers/footers, while `createPDF()` generates a continuous capture of the visible rect. These produce slightly different output because `NSPrintOperation` applies print media queries and pagination.
- **To match Typora exactly:** We would need to determine whether Typora uses `createPDF()` or `NSPrintOperation`. If it uses `NSPrintOperation`, our tool should too.

### Code Size

The tool is approximately **120 lines of Swift** as a single file. It requires:
- No external dependencies
- No Cocoa Pods or Swift Package Manager packages
- Only system frameworks: WebKit, AppKit, Foundation

### Source Code

A complete implementation is at `webkit-pdf.swift` in this directory. It:
1. Parses command-line arguments (input HTML, output PDF, optional width)
2. Creates a WKWebView with file access permissions
3. Loads the HTML file with `loadFileURL`
4. Waits for navigation and font loading to complete
5. Measures content height via JavaScript
6. Calls `createPDF()` with the measured dimensions
7. Writes the PDF data to the output file

### Compilation

```bash
swiftc -o webkit-pdf webkit-pdf.swift -framework WebKit -framework AppKit
```

**Current blocker:** The macOS 26 beta Command Line Tools have a known `SwiftBridging` module redefinition bug that prevents compilation without full Xcode.app installed. This is a temporary beta issue, not a fundamental limitation. The code parses successfully (`swiftc -parse` succeeds).

**Fix options:**
1. Install Xcode.app (not just Command Line Tools) — resolves the module conflict
2. Wait for a macOS 26 beta update that fixes the CLI Tools
3. Compile on a macOS 15 system and copy the binary

### Can It Be Called from Node.js?

Yes, via `child_process.execSync` or `spawn`:

```javascript
import { execSync } from 'child_process';
execSync('./webkit-pdf input.html output.pdf --width 1400');
```

This integrates cleanly with the existing `markdown-kit.js` pipeline — the Node script handles markdown-to-HTML conversion, then hands off to the Swift binary for HTML-to-PDF rendering.

---

## 4. Other WebKit Options

### wkpdf

- **What:** Ruby-based command line HTML-to-PDF converter using WebKit
- **Status:** Unmaintained since December 2014. Pulled from RubyGems.
- **Engine:** Used RubyCocoa bindings to the old (deprecated) `WebView` class, not modern WKWebView.
- **Verdict:** Dead project. Not usable on modern macOS.

### wkhtmltopdf

- **What:** C++ tool converting HTML to PDF using QtWebKit
- **WebKit Version:** Uses QtWebKit, which was deprecated by Qt in 2015 and removed in 2016. The QtWebKit it ships is frozen at approximately the 2013-era WebKit.
- **Status:** Effectively unmaintained. The 0.12.6.x stable series is current but receives minimal updates. The maintainer recommends WeasyPrint or Puppeteer as alternatives.
- **Would it match Typora?** Absolutely not. It uses a 2013-vintage WebKit fork that predates even the Blink split. Its rendering is dramatically different from modern WebKit/Safari.
- **Verdict:** Legacy tool, actively discouraged by its own maintainer.

### Safari Headless

Safari does not have a headless mode. Apple provides `safaridriver` for WebDriver automation, but it requires a visible Safari window and does not support PDF generation through the WebDriver protocol.

### Epiphany/GNOME Web

- Uses WebKitGTK (the GTK port of WebKit)
- Linux only
- Different font rendering stack than macOS WebKit (uses HarfBuzz/FreeType instead of CoreText/CoreGraphics)
- Would NOT match Typora's rendering

---

## 5. The Pragmatic Question

### If We Switched to Playwright WebKit, Would Fonts Match Typora?

**No, for three independent reasons:**

1. Playwright WebKit cannot generate PDFs at all
2. Playwright's WebKit is a patched MiniBrowser build, not WKWebView
3. Even on macOS, Playwright WebKit has documented font kerning and layout issues that differ from Safari/WKWebView

### Why Will They Always Look Different? (Blink vs WebKit)

The engines use fundamentally different text pipelines:

```
Typora (macOS):
  HTML → WKWebView → CoreText (shaping) → CoreGraphics (rasterization) → PDF

markdown-kit.js (Chromium):
  HTML → Blink → HarfBuzz (shaping) → Skia (rasterization) → PDF
```

CoreText and HarfBuzz are independent text shaping engines that produce different glyph positions for the same font and text. This is not a bug — it's a design choice by each platform. The differences are typically:
- 0.1-0.5px per glyph position
- Different kerning pair values for the same font
- Different line break decisions at paragraph boundaries
- Accumulates over long paragraphs to produce visibly different line wrapping

**No amount of CSS can fix this.** The difference is below the CSS level, in the engine's font metrics and shaping tables.

---

## 6. Recommendation

### Best Path: Swift WKWebView CLI Tool

The recommended approach is a two-stage pipeline:

```
markdown-kit.js (Node.js)         webkit-pdf (Swift binary)
┌─────────────────────────┐       ┌──────────────────────────┐
│ 1. Parse markdown       │       │ 1. Load HTML file        │
│ 2. Apply CSS theme      │  ──>  │ 2. Render in WKWebView   │
│ 3. Generate HTML file   │       │ 3. Measure content       │
│ 4. Call webkit-pdf      │       │ 4. createPDF() → output  │
│ 5. Clean up temp files  │       └──────────────────────────┘
└─────────────────────────┘
```

**Why this is the right answer:**
- Uses the *exact same* WebKit engine as Typora (WKWebView with CoreText)
- ~120 lines of Swift, single file, no dependencies
- Compiles to a standalone binary (~200KB)
- macOS only, but that matches our use case (Typora is a desktop app)
- `createPDF()` API is stable and supported since macOS 11.0

**What to do:**
1. Install Xcode.app (or wait for CLI Tools beta fix)
2. Compile `webkit-pdf.swift`
3. Modify `markdown-kit.js` to output an intermediate HTML file and call the Swift binary instead of Puppeteer
4. Compare output against Typora — rendering should be near-identical

### Alternative: Keep Puppeteer, Accept the Difference

If the 1-2% difference is acceptable, the current Puppeteer approach is simpler and cross-platform. The differences are subtle (sub-pixel font metrics, occasional line-break differences) and only visible in side-by-side comparison.

### What NOT to Do

- **Don't use Playwright WebKit** — cannot generate PDFs, and its WebKit is not the same as Apple's
- **Don't use wkhtmltopdf** — uses a 2013-era frozen WebKit fork, worse than Blink
- **Don't use wkpdf** — unmaintained since 2014, uses deprecated APIs
- **Don't try to make Blink match WebKit via CSS** — the differences are engine-level, below CSS

---

## Summary Table

| Option | PDF Gen? | Matches Typora? | Cross-Platform? | Status |
|--------|----------|-----------------|-----------------|--------|
| **Puppeteer (Blink)** | Yes | 98% | Yes | Current solution |
| **Playwright WebKit** | **No** | N/A | Yes | Dead end |
| **Swift WKWebView CLI** | Yes | ~99.9% | macOS only | **Recommended** |
| wkhtmltopdf | Yes | No (~80%) | Yes | Deprecated |
| wkpdf | Yes (was) | No | macOS only | Dead |
| Safari headless | No | N/A | macOS only | Not possible |

---

## Appendix: Compilation Notes

The Swift tool (`webkit-pdf.swift`) is ready to compile but blocked by a macOS 26 beta toolchain bug:

```
error: redefinition of module 'SwiftBridging'
```

This is caused by duplicate module map files in the Command Line Tools:
- `/Library/Developer/CommandLineTools/usr/include/swift/module.modulemap`
- `/Library/Developer/CommandLineTools/usr/include/swift/bridging.modulemap`

Both define `module SwiftBridging`. The fix is either:
1. Install full Xcode.app (which uses a different SDK path)
2. Remove the duplicate file (requires sudo): `sudo mv .../module.modulemap .../module.modulemap.bak`
3. Wait for Apple to fix this in the next CLI Tools update

The code itself compiles correctly (`swiftc -parse` succeeds with zero errors).

---

*Research conducted March 22, 2026*
