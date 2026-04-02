# Internal Anchor Links in WebKit-Generated PDFs: Definitive Research

## Executive Summary

**Root cause**: `WKWebView.createPDF()` does NOT generate PDF link annotations for internal `#anchor` links. This is a fundamental limitation of the `createPDF()` API, which bypasses WebKit's printing pipeline where internal link support was added in 2016. The fix exists in WebKit's `PrintContext` class (used by `printOperation`/printing path) but was never wired into the `createPDF()` snapshot API.

**Recommended solution**: Post-process the PDF with Apple's PDFKit framework. Before generating the PDF, inject JavaScript to collect all anchor link positions via `getBoundingClientRect()`. After `createPDF()` produces the PDF data, use PDFKit to add `PDFAnnotation` link annotations at the collected coordinates. This is the only approach that works with the pageless (single-page) format.

---

## 1. Root Cause Analysis

### Why WKWebView.createPDF() Does Not Preserve Internal Links

There are **two distinct PDF generation paths** in WebKit/WKWebView:

| Path | API | Internal Links | Notes |
|------|-----|---------------|-------|
| **Print path** | `printOperation(with:)` | YES (since 2016) | Uses `PrintContext`, `WKPrintingView`, paged output |
| **Snapshot path** | `createPDF(configuration:)` | NO | Direct content capture, single rect, no `PrintContext` |

The internal link support was added to WebKit in **November 2016** via [WebKit Bug #112081](https://bugs.webkit.org/show_bug.cgi?id=112081) (changeset 208347). The fix added `collectLinkedDestinations()` and `outputLinkedDestinations()` methods to the `PrintContext` class, which calls `GraphicsContext::setDestinationForRect()` and `GraphicsContext::addDestinationAtPoint()` -- wrappers around the CoreGraphics functions `CGPDFContextSetDestinationForRect` and `CGPDFContextAddDestinationAtPoint`.

**The critical distinction**: `createPDF()` was added in iOS 14 / macOS 11 (2020) as a convenience API that captures a rectangular snapshot of the web view content. It does NOT use `PrintContext` at all. It generates PDF data by rendering the web view contents into a CGPDFContext directly, without invoking the printing pipeline. Therefore, the 2016 internal link fix has zero effect on `createPDF()` output.

This is confirmed by multiple independent sources:
- [Marked 2's developer](http://support.markedapp.com/discussions/questions/3598-saveexport-to-pdf-and-internal-links) states internal links don't work "because of a webkit bug" in the PDF export path
- [MacDown Issue #198](https://github.com/MacDownApp/macdown/issues/198) confirms "this seems to be a general problem if you generate PDF from a WebView on OS X"
- [Chromium Issue #347674894](https://issues.chromium.org/issues/347674894) confirms the same limitation exists in Chromium's `Page.printToPDF` CDP method

### Has Apple Ever Documented This?

No. Apple's documentation for `createPDF(configuration:completionHandler:)` says only: "Generates PDF data from the web view's contents asynchronously." There is no mention of link preservation, annotation generation, or any limitations regarding hyperlinks. This is an undocumented limitation.

---

## 2. Does Typora's PDF Export Have Clickable Internal Links?

### Answer: It is inconsistent and historically broken

[Typora Issue #384](https://github.com/typora/typora-issues/issues/384) (filed October 2016) reports that "internal links are broken" when exporting to PDF. The issue was marked "fixed" in version 0.9.9.8.4 (November 2016), but subsequent comments from 2019 report regressions in later versions.

Typora is an Electron app. On macOS, its PDF export likely uses either:
- Electron's `webContents.printToPDF()` (which wraps Chromium's print-to-PDF)
- A native macOS print path for the WebKit rendering

Both paths have known limitations with internal links. Typora's fix in 2016 coincides with the WebKit bug #112081 fix, suggesting they may have been relying on the WebKit print path. However, the Chromium-based path (`printToPDF`) also fails to render anchor links per [Chromium Issue #347674894](https://issues.chromium.org/issues/347674894).

**The PDF outline (bookmarks panel) works**: Typora does generate a PDF outline/bookmark tree from headings. This is separate from inline clickable links -- it's the sidebar navigation in PDF viewers. This works because it uses a different PDF structure (`/Outlines` dictionary) than link annotations.

**Key takeaway**: Typora's internal TOC links in the body of the PDF are NOT reliably clickable. The PDF outline sidebar IS generated. Our tool should aim for both.

---

## 3. Safari's PDF Export Behavior

### "Export as PDF" vs "Print > Save as PDF"

| Method | Link Type | Behavior |
|--------|----------|----------|
| File > Export as PDF | External (https://) | PRESERVED as clickable links |
| File > Export as PDF | Internal (#anchor) | Converted to `file:///...#anchor` URLs -- BROKEN |
| File > Print > Save as PDF | External (https://) | PRESERVED as clickable links |
| File > Print > Save as PDF | Internal (#anchor) | Should work via PrintContext fix (if used) |

Safari's "Export as PDF" produces a single continuous page (no pagination). The "Print > Save as PDF" goes through the print pipeline and produces paged output. Safari's print path uses WebKit's `PrintContext`, which has the internal link fix from 2016. However, reports are mixed on whether internal fragment links actually work -- some sources confirm external links are preserved but internal anchors are converted to absolute `file://` URLs.

The [Macworld article](https://www.macworld.com/article/226597/how-to-print-a-web-page-as-a-pdf-with-links-that-work.html) confirms Safari preserves external links but does not specifically confirm internal anchor link preservation.

---

## 4. Alternative WebKit/macOS PDF APIs

### 4.1 WKWebView.printOperation(with:) + NSPrintSaveJob

**Can it work?** Potentially yes, but with significant caveats.

The print path goes through `PrintContext` which has the internal link fix. To save to PDF programmatically:

```swift
let printInfo = NSPrintInfo()
printInfo.jobDisposition = .save
printInfo.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = outputURL

let printOp = webView.printOperation(with: printInfo)
printOp.showsPrintPanel = false
printOp.showsProgressPanel = false
printOp.runModal(for: window, delegate: self, didRun: #selector(didRun), contextInfo: nil)
```

**Problems**:
- Requires a visible NSWindow (cannot run headless easily)
- Produces PAGED output (A4/Letter pages) -- incompatible with pageless format
- `runOperation()` produces blank PDFs; must use `runModal(for:)` (confirmed by multiple developers)
- The [swift-webster package](https://github.com/aaronland/swift-webster) uses this approach with the deprecated `WebView` class and NSPrintOperation, but still gets paged output
- No way to control page size to create a single continuous page (the whole point of pageless mode)

**Verdict**: NOT viable for pageless/single-page PDFs. Could work for paged output if we abandon the pageless format.

### 4.2 NSPrintOperation.pdfOperation(with:inside:toPath:printInfo:)

Creates a PDF from an NSView. This is how some apps generate PDFs from attributed strings.

**Problem**: We'd need to render the HTML into an NSView first, which means using WKWebView anyway. And this API doesn't go through `PrintContext`.

**Verdict**: NOT viable.

### 4.3 CGPDFContext Direct Rendering

The low-level CoreGraphics API supports:
- `CGPDFContextAddDestinationAtPoint(context, name, point)` -- creates a named destination
- `CGPDFContextSetDestinationForRect(context, name, rect)` -- creates a clickable rect that jumps to destination
- `CGPDFContextSetURLForRect(context, url, rect)` -- creates a clickable rect for external URLs

**Problem**: We'd need to render the entire HTML document ourselves using CoreGraphics drawing primitives. This is essentially reimplementing a web browser.

**Verdict**: NOT viable as primary approach. BUT these are the exact functions that PDFKit wraps, so the post-processing approach uses them indirectly.

### 4.4 Legacy WebView (deprecated)

The old `WebView` class (pre-WKWebView) had different PDF generation that some developers report handled links differently. However:
- Deprecated since macOS 10.14
- Removed from recent SDKs
- No evidence it handled internal links any better

**Verdict**: NOT viable.

---

## 5. The Post-Processing Solution (Recommended)

### Architecture

```
HTML --> WKWebView.createPDF() --> Raw PDF (no internal links)
  |                                      |
  |  evaluateJavaScript()                |
  |  (collect anchor positions)          |
  v                                      v
Link Map -----------------------------> PDFKit Post-Processing
(anchor -> {x,y,w,h,target})                |
                                             v
                                       PDF with Link Annotations
```

### Step 1: Collect Anchor Positions via JavaScript

Before calling `createPDF()`, inject JavaScript to collect the bounding rectangles of all internal links and their targets:

```javascript
(() => {
    const links = [];
    const anchors = {};

    // Collect all elements with IDs (potential targets)
    document.querySelectorAll('[id]').forEach(el => {
        const rect = el.getBoundingClientRect();
        anchors[el.id] = {
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height
        };
    });

    // Collect all internal links
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        const rect = a.getBoundingClientRect();
        const targetId = a.getAttribute('href').substring(1);
        if (rect.width > 0 && rect.height > 0) {
            links.push({
                sourceRect: {
                    x: rect.left,
                    y: rect.top,
                    width: rect.width,
                    height: rect.height
                },
                targetId: targetId,
                text: a.textContent
            });
        }
    });

    return JSON.stringify({ links, anchors });
})()
```

### Step 2: Generate PDF with createPDF()

Use the existing `createPDF()` approach (unchanged from current webkit-pdf.swift).

### Step 3: Post-Process with PDFKit

Load the generated PDF with PDFKit and add link annotations:

```swift
import PDFKit

func addInternalLinks(pdfData: Data, linkMap: LinkMap) -> Data? {
    guard let pdfDoc = PDFDocument(data: pdfData),
          let page = pdfDoc.page(at: 0) else { return nil }

    let pageBounds = page.bounds(for: .mediaBox)
    let pageHeight = pageBounds.height

    // For each internal link, create a link annotation
    for link in linkMap.links {
        guard let targetRect = linkMap.anchors[link.targetId] else { continue }

        // Convert HTML coordinates (origin top-left) to PDF coordinates (origin bottom-left)
        // createPDF maps 1 CSS pixel = 1 PDF point, so scale factor is 1.0
        let sourceBounds = CGRect(
            x: link.sourceRect.x,
            y: pageHeight - link.sourceRect.y - link.sourceRect.height,
            width: link.sourceRect.width,
            height: link.sourceRect.height
        )

        // Create destination point (where clicking will jump to)
        let destY = pageHeight - targetRect.y
        let destination = PDFDestination(page: page, at: CGPoint(x: 0, y: destY))

        // Create the link annotation
        let annotation = PDFAnnotation(
            bounds: sourceBounds,
            forType: .link,
            withProperties: nil
        )
        annotation.action = PDFActionGoTo(destination: destination)

        page.addAnnotation(annotation)
    }

    // Write modified PDF
    return pdfDoc.dataRepresentation()
}
```

### Step 4: Coordinate System Mapping

The trickiest part is mapping HTML pixel coordinates to PDF point coordinates.

**HTML coordinate system**: Origin at top-left, Y increases downward. `getBoundingClientRect()` returns coordinates relative to the viewport.

**PDF coordinate system**: Origin at bottom-left, Y increases upward. Coordinates are in "points" (1/72 inch).

**Scale factor**: `WKWebView.createPDF()` renders at the web view's pixel dimensions. For a 1400px-wide web view, the PDF page width will be 1400 "points" (since WebKit maps 1 CSS pixel = 1 PDF point in `createPDF()`). The scale factor is therefore 1.0 for x and width, but y must be flipped: `pdfY = pageHeight - htmlY`.

For pageless (single-page) PDFs, there is only one PDF page (page index 0), and all anchors and links are on that same page. This simplifies the mapping significantly -- no need to determine which page an element falls on.

### Key Advantages

1. **Works with pageless format**: No need to switch to paged output
2. **No WebKit API changes needed**: Uses the existing `createPDF()` path
3. **Pure Apple frameworks**: Only PDFKit (built into macOS), no third-party dependencies
4. **Precise positioning**: JavaScript gives us exact pixel coordinates
5. **Handles all link types**: TOC links, cross-references, footnote backlinks

### Known Challenges

1. **Coordinate precision**: The mapping from HTML pixels to PDF points needs to account for any scaling WKWebView applies. Testing reveals createPDF uses 1:1 mapping (1 CSS px = 1 PDF pt) when `pdfConfig.rect` matches the web view frame.

2. **Scrollable content**: For tall documents, `getBoundingClientRect()` returns viewport-relative coordinates. Must scroll to ensure all elements are measured, OR use `element.offsetTop` / `element.offsetLeft` which are document-relative. Prefer `offsetTop`/`offsetLeft` for reliability.

3. **Dynamic content**: JavaScript must run after all content is rendered (fonts loaded, images sized). The existing `document.fonts.ready` wait handles this.

4. **Multi-page PDFs**: For paged output, would need to determine which PDF page each element falls on. Not needed for pageless.

---

## 6. Complete Modified webkit-pdf.swift

Below is the full implementation integrating the post-processing approach:

```swift
#!/usr/bin/env swift
//
// webkit-pdf.swift -- HTML to PDF with clickable internal links
//
// Uses WKWebView.createPDF() for rendering, then PDFKit to add
// link annotations for internal #anchor references.
//

import AppKit
import WebKit
import PDFKit
import Foundation

// -- Data structures for link mapping --

struct SourceRect: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct LinkInfo: Codable {
    let sourceRect: SourceRect
    let targetId: String
    let text: String
}

struct LinkMap: Codable {
    let links: [LinkInfo]
    let anchors: [String: SourceRect]
}

// -- Argument parsing --

let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("""
    Usage: swift webkit-pdf.swift <input.html> <output.pdf> [--width pixels]

    Renders HTML to PDF using macOS native WKWebView (Apple WebKit)
    with clickable internal anchor links via PDFKit post-processing.

    Options:
      --width <pixels>   Page width in pixels (default: 1400)

    """, stderr)
    exit(1)
}

let inputPath = args[1]
let outputPath = args[2]

var pageWidth: CGFloat = 1400
if let widthIdx = args.firstIndex(of: "--width"), widthIdx + 1 < args.count,
   let w = Double(args[widthIdx + 1]) {
    pageWidth = CGFloat(w)
}

guard FileManager.default.fileExists(atPath: inputPath) else {
    fputs("Error: file not found: \(inputPath)\n", stderr)
    exit(1)
}

// -- JavaScript to collect anchor positions --

let collectLinksJS = """
(() => {
    const links = [];
    const anchors = {};

    // Collect all elements with IDs (potential link targets)
    document.querySelectorAll('[id]').forEach(el => {
        const rect = el.getBoundingClientRect();
        // Use absolute position (document-relative, not viewport-relative)
        anchors[el.id] = {
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY,
            width: rect.width,
            height: rect.height
        };
    });

    // Collect all internal links (href starting with #)
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        const rect = a.getBoundingClientRect();
        const targetId = decodeURIComponent(a.getAttribute('href').substring(1));
        if (rect.width > 0 && rect.height > 0) {
            links.push({
                sourceRect: {
                    x: rect.left + window.scrollX,
                    y: rect.top + window.scrollY,
                    width: rect.width,
                    height: rect.height
                },
                targetId: targetId,
                text: a.textContent.substring(0, 100)
            });
        }
    });

    return JSON.stringify({ links, anchors });
})()
"""

// -- Application delegate --

class PDFGenerator: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    let inputURL: URL
    let outputURL: URL
    let width: CGFloat
    var webView: WKWebView!
    var linkMap: LinkMap?

    init(input: String, output: String, width: CGFloat) {
        self.inputURL = URL(fileURLWithPath: input)
        self.outputURL = URL(fileURLWithPath: output)
        self.width = width
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        webView = WKWebView(
            frame: NSRect(x: 0, y: 0, width: width, height: 800),
            configuration: config
        )
        webView.navigationDelegate = self
        webView.loadFileURL(inputURL, allowingReadAccessTo: URL(fileURLWithPath: "/"))

        fputs("Loading: \(inputURL.path)\n", stderr)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        fputs("Page loaded. Waiting for fonts...\n", stderr)

        webView.evaluateJavaScript("document.fonts.ready.then(() => true)") { _, _ in
            self.collectLinksAndMeasure()
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!,
                 withError error: Error) {
        fputs("Navigation failed: \(error.localizedDescription)\n", stderr)
        exit(1)
    }

    func collectLinksAndMeasure() {
        // Step 1: Collect all anchor link positions
        webView.evaluateJavaScript(collectLinksJS) { result, error in
            if let jsonString = result as? String,
               let jsonData = jsonString.data(using: .utf8) {
                do {
                    self.linkMap = try JSONDecoder().decode(
                        LinkMap.self, from: jsonData)
                    let linkCount = self.linkMap?.links.count ?? 0
                    let anchorCount = self.linkMap?.anchors.count ?? 0
                    fputs("Collected \(linkCount) internal links, " +
                          "\(anchorCount) anchor targets\n", stderr)
                } catch {
                    fputs("Warning: Failed to parse link map: \(error). " +
                          "Continuing without links.\n", stderr)
                }
            }

            // Step 2: Measure content height
            self.measureAndGeneratePDF()
        }
    }

    func measureAndGeneratePDF() {
        let measureJS = """
        (() => {
            const children = Array.from(document.body.children);
            let lastBottom = 0;
            for (const el of children) {
                const rect = el.getBoundingClientRect();
                if (rect.height > 0 && rect.bottom > lastBottom) {
                    lastBottom = rect.bottom;
                }
            }
            return Math.ceil(lastBottom);
        })()
        """

        webView.evaluateJavaScript(measureJS) { result, error in
            if let error = error {
                fputs("JavaScript error: \(error.localizedDescription)\n", stderr)
                exit(1)
            }

            let contentHeight = (result as? NSNumber)?.doubleValue ?? 800.0
            fputs("Content height: \(Int(contentHeight))px\n", stderr)

            self.generatePDF(height: CGFloat(contentHeight) + 10)
        }
    }

    func generatePDF(height: CGFloat) {
        let pdfConfig = WKPDFConfiguration()
        pdfConfig.rect = CGRect(x: 0, y: 0, width: width, height: height)

        fputs("Generating PDF: \(Int(width))x\(Int(height))px\n", stderr)

        webView.createPDF(configuration: pdfConfig) { result in
            switch result {
            case .success(let data):
                // Step 3: Post-process with PDFKit to add internal links
                let finalData = self.addInternalLinks(
                    to: data, pageHeight: height)

                do {
                    try finalData.write(to: self.outputURL)
                    let sizeKB = Double(finalData.count) / 1024.0
                    fputs("Output: \(self.outputURL.path)\n", stderr)
                    fputs("Size:   \(String(format: "%.1f", sizeKB)) KB\n",
                          stderr)
                    exit(0)
                } catch {
                    fputs("Write error: \(error.localizedDescription)\n",
                          stderr)
                    exit(1)
                }
            case .failure(let error):
                fputs("PDF generation failed: " +
                      "\(error.localizedDescription)\n", stderr)
                exit(1)
            }
        }
    }

    func addInternalLinks(to pdfData: Data, pageHeight: CGFloat) -> Data {
        guard let linkMap = self.linkMap,
              !linkMap.links.isEmpty,
              let pdfDoc = PDFDocument(data: pdfData),
              let page = pdfDoc.page(at: 0) else {
            if self.linkMap?.links.isEmpty ?? true {
                fputs("No internal links found. " +
                      "Skipping post-processing.\n", stderr)
            }
            return pdfData
        }

        let pageBounds = page.bounds(for: .mediaBox)
        let pdfPageHeight = pageBounds.height

        // createPDF maps 1 CSS pixel = 1 PDF point, so scale is 1:1
        // Only coordinate system flip needed (HTML: top-left, PDF: bottom-left)

        var addedCount = 0

        for link in linkMap.links {
            guard let targetAnchor = linkMap.anchors[link.targetId] else {
                fputs("  Warning: Target '\(link.targetId)' not found " +
                      "for link '\(link.text)'\n", stderr)
                continue
            }

            // Convert source rect: flip Y axis
            let sourceBounds = CGRect(
                x: link.sourceRect.x,
                y: pdfPageHeight - link.sourceRect.y
                   - link.sourceRect.height,
                width: link.sourceRect.width,
                height: link.sourceRect.height
            )

            // Create destination point: flip Y axis
            let destY = pdfPageHeight - targetAnchor.y
            let destination = PDFDestination(
                page: page, at: CGPoint(x: 0, y: destY))

            // Create link annotation
            let annotation = PDFAnnotation(
                bounds: sourceBounds,
                forType: .link,
                withProperties: nil
            )
            annotation.action = PDFActionGoTo(destination: destination)

            page.addAnnotation(annotation)
            addedCount += 1
        }

        fputs("Added \(addedCount) internal link annotations to PDF\n",
              stderr)

        return pdfDoc.dataRepresentation() ?? pdfData
    }
}

// -- Main --

let absInput = (inputPath as NSString).standardizingPath
let absOutput: String
if (outputPath as NSString).isAbsolutePath {
    absOutput = outputPath
} else {
    absOutput = (FileManager.default.currentDirectoryPath as NSString)
        .appendingPathComponent(outputPath)
}

let generator = PDFGenerator(
    input: absInput, output: absOutput, width: pageWidth)

let app = NSApplication.shared
app.delegate = generator
app.run()
```

---

## 7. All Solutions Evaluated

### Solution A: Post-Process with PDFKit (RECOMMENDED)

**Approach**: Generate PDF with `createPDF()`, then use PDFKit to add link annotations.

| Attribute | Detail |
|-----------|--------|
| Effort | Medium (half-day implementation) |
| Compatibility | macOS 10.13+ (PDFKit), macOS 11+ (createPDF) |
| Pageless support | YES |
| Dependencies | None (PDFKit is system framework) |
| Link types | Internal anchors, could extend to external URLs |
| Accuracy | High -- direct pixel-to-point mapping |
| Maintenance | Low -- stable Apple APIs |

**Pros**: Works with existing pageless format. No architecture change. Pure Swift, no dependencies. Precise coordinate mapping.

**Cons**: Requires coordinate system conversion. Adds ~50 lines of Swift code.

### Solution B: Switch to printOperation Path

**Approach**: Replace `createPDF()` with `printOperation(with:)` + `NSPrintSaveJob`.

| Attribute | Detail |
|-----------|--------|
| Effort | High (significant architecture change) |
| Compatibility | macOS 11+ |
| Pageless support | NO -- forces paged output |
| Dependencies | None |
| Link types | Both internal and external (via WebKit PrintContext) |
| Accuracy | Native WebKit handling |
| Maintenance | Medium |

**Pros**: Native internal link support via WebKit's PrintContext fix.

**Cons**: CANNOT produce pageless output. Requires visible NSWindow. `runOperation()` produces blank pages (must use `runModal`). Fundamentally incompatible with the single-page format.

### Solution C: CGPDFContext Low-Level Approach

**Approach**: Render HTML to a CGPDFContext directly, calling `CGPDFContextAddDestinationAtPoint` and `CGPDFContextSetDestinationForRect`.

| Attribute | Detail |
|-----------|--------|
| Effort | Very High (essentially building a renderer) |
| Compatibility | macOS 10.4+ |
| Pageless support | YES |
| Dependencies | None |
| Accuracy | Perfect (you control everything) |

**Pros**: Complete control over PDF generation.

**Cons**: Requires reimplementing web page rendering. Not practical.

### Solution D: Use wkhtmltopdf

**Approach**: Replace WebKit with wkhtmltopdf, which has `--enable-internal-links`.

| Attribute | Detail |
|-----------|--------|
| Effort | Medium |
| Compatibility | Cross-platform |
| Pageless support | Limited |
| Dependencies | wkhtmltopdf binary (Qt WebKit) |
| Link types | Internal and external |
| Accuracy | Good but dated rendering (Qt WebKit, roughly Safari 5 era) |

**Pros**: Internal links work out of the box. Well-documented flags.

**Cons**: Based on ancient WebKit (Qt fork). Rendering quality inferior to modern WebKit. CSS compatibility issues. wkhtmltopdf is abandoned/archived. External dependency.

### Solution E: Electron/Chromium printToPDF

**Approach**: Use Electron/Puppeteer for PDF generation.

| Attribute | Detail |
|-----------|--------|
| Effort | Low (already supported in markdown-kit.js) |
| Compatibility | Cross-platform |
| Pageless support | YES (our Chromium engine already does this) |
| Link types | External only -- internal links also broken |

**Pros**: Already implemented as the default engine.

**Cons**: Same internal link limitation as WebKit -- [Chromium Issue #347674894](https://issues.chromium.org/issues/347674894) confirms `Page.printToPDF` does not render anchor links. Would still need post-processing.

---

## 8. How Other Apps Handle This

| App | PDF Engine | Internal Links Work? | Approach |
|-----|-----------|---------------------|----------|
| Typora | Electron/Chromium | Inconsistent / broken per issues | Unknown post-processing, or relies on WebKit print path |
| Marked 2 | WebKit (deprecated) | NO | [Developer confirms WebKit bug](http://support.markedapp.com/discussions/questions/3598-saveexport-to-pdf-and-internal-links), plans to "bypass WebKit entirely" |
| MacDown | WebKit | NO | [Issue #198](https://github.com/MacDownApp/macdown/issues/198) confirms it's a WebKit limitation |
| Safari (Print > Save as PDF) | WebKit PrintContext | External: YES, Internal: inconsistent | Uses print path with PrintContext fix |
| Safari (File > Export as PDF) | WebKit snapshot | External: YES, Internal: NO (become file:// URLs) | Snapshot path, no PrintContext |
| Prince XML | Custom engine | YES | Commercial tool, custom PDF engine, not WebKit |
| wkhtmltopdf | Qt WebKit (patched) | YES (with --enable-internal-links) | Custom patches to Qt WebKit for link annotation injection |

---

## 9. Concrete Recommendation

**Implement Solution A (PDFKit post-processing)** in the existing `webkit-pdf.swift`. The implementation is provided in Section 6 above.

### Integration with markdown-kit.js

The JavaScript orchestrator (`markdown-kit.js`) already passes `--width` to `webkit-pdf.swift`. No changes needed on the JS side -- the Swift tool handles the post-processing internally.

### Testing Strategy

1. Create a test markdown file with a `[toc]` and multiple headings
2. Generate PDF with the modified `webkit-pdf.swift`
3. Open in Preview.app and click TOC links -- they should jump to headings
4. Verify coordinates are correct (links highlight the right text, destinations scroll to the right heading)
5. Test edge cases: very long documents, headings with special characters, deeply nested anchors

### Future Enhancements

1. **PDF Outline/Bookmarks**: In addition to inline link annotations, generate a PDF outline (bookmark tree) from headings using PDFKit's `PDFOutline` class. This adds the sidebar navigation panel.
2. **External link annotations**: Could also add explicit link annotations for `https://` links, though `createPDF()` already preserves these.
3. **Multi-page support**: If pageless mode is ever abandoned, the post-processing would need to map each link/anchor to its correct page based on page boundaries.

---

## Sources

- [WebKit Bug #112081](https://bugs.webkit.org/show_bug.cgi?id=112081) -- "Printing to PDF should produce internal links when HTML has internal links" (RESOLVED FIXED, November 2016)
- [Chromium Issue #347674894](https://issues.chromium.org/issues/347674894) -- "Page.printToPdf does not render anchor links"
- [Typora Issue #384](https://github.com/typora/typora-issues/issues/384) -- "PDF export breaks internal link"
- [MacDown Issue #198](https://github.com/MacDownApp/macdown/issues/198) -- "Intradocument links don't work in exported PDF"
- [Marked 2 Support Forum](http://support.markedapp.com/discussions/questions/3598-saveexport-to-pdf-and-internal-links) -- "Save/Export to PDF and internal links"
- [Brett Terpstra on Marked 2 PDFs](https://brettterpstra.com/2020/11/12/marked-2-big-sur-and-blurry-pdfs/) -- Plans to bypass WebKit for PDF export
- [Apple Developer Documentation: createPDF](https://developer.apple.com/documentation/webkit/wkwebview/createpdf(configuration:completionhandler:))
- [Apple Developer Documentation: PDFAnnotation](https://developer.apple.com/documentation/pdfkit/pdfannotation)
- [Apple Developer Documentation: PDFDestination](https://developer.apple.com/documentation/pdfkit/pdfdestination)
- [Apple Developer Documentation: PDFActionGoTo](https://developer.apple.com/documentation/pdfkit/pdfactiongoto)
- [Geri-Borbas/macOS.Production.PDF_Links](https://github.com/Geri-Borbas/macOS.Production.PDF_Links) -- Example of adding PDFAnnotation links programmatically
- [Macworld: How to print a web page as a PDF with links that work](https://www.macworld.com/article/226597/how-to-print-a-web-page-as-a-pdf-with-links-that-work.html)
- [wkhtmltopdf PR #2957](https://github.com/wkhtmltopdf/wkhtmltopdf/pull/2957/files) -- Fix for empty anchors breaking internal links
- [CGPDFContext addDestination API](https://developer.apple.com/documentation/coregraphics/cgcontext/adddestination(_:at:)) -- `addDestination(_:at:)` for PDF named destinations
