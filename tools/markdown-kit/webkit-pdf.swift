#!/usr/bin/env swift
//
// webkit-pdf.swift — HTML to PDF using macOS native WKWebView (Apple WebKit)
//
// Usage:
//   swift webkit-pdf.swift input.html output.pdf [--width 1400]
//
// This uses the exact same WebKit engine as Typora on macOS (WKWebView),
// producing pixel-identical rendering. No dependencies needed — uses only
// system frameworks (WebKit, AppKit, PDFKit, Foundation).
//
// After generating the PDF via createPDF(), internal anchor links
// (href="#...") are added as clickable PDFAnnotation links using PDFKit
// post-processing. This works around the WKWebView.createPDF() limitation
// where internal links are not preserved (see INTERNAL_LINKS_RESEARCH.md).
//
// Requires: macOS 11.0+ (for WKWebView.createPDF)
//

import AppKit
import WebKit
import PDFKit
import Foundation

// ── Data structures for internal link mapping ─────────────────────────────────

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

// ── Argument parsing ────────────────────────────────────────────────────────────

let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("""
    Usage: swift webkit-pdf.swift <input.html> <output.pdf> [--width pixels]

    Renders HTML to PDF using macOS native WKWebView (Apple WebKit).
    Same rendering engine as Typora on macOS.

    Internal anchor links (href="#...") are added as clickable PDF
    annotations via PDFKit post-processing.

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

// ── JavaScript to collect internal link and anchor positions ──────────────────
//
// Runs in the WKWebView after fonts are loaded, before PDF generation.
// Collects:
//   - All <a href="#..."> elements with their bounding rects (link sources)
//   - All elements with [id] attributes with their bounding rects (link targets)
//
// getBoundingClientRect() returns viewport-relative coordinates. In a
// WKWebView with no scrolling (the frame is sized to content), these
// are equivalent to document-relative coordinates. We add scrollX/scrollY
// as a safety measure for any edge case where the view might scroll.

let collectLinksJS = """
(() => {
    const links = [];
    const anchors = {};

    // Collect all elements with IDs (potential link targets)
    document.querySelectorAll('[id]').forEach(el => {
        const rect = el.getBoundingClientRect();
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
        const rawHref = a.getAttribute('href').substring(1);
        // Decode URI-encoded fragment IDs (e.g., %20 -> space)
        let targetId;
        try {
            targetId = decodeURIComponent(rawHref);
        } catch (e) {
            targetId = rawHref;
        }
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

// ── Application delegate ────────────────────────────────────────────────────────

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
        // Create WKWebView configuration
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        // Create the web view — it does NOT need to be visible.
        // WKWebView renders in its own process; no window required.
        webView = WKWebView(
            frame: NSRect(x: 0, y: 0, width: width, height: 800),
            configuration: config
        )
        webView.navigationDelegate = self

        // Load the HTML file — grant read access to the filesystem root so
        // file:// URLs for fonts (in ~/.cache), theme CSS, and images all resolve.
        webView.loadFileURL(inputURL, allowingReadAccessTo: URL(fileURLWithPath: "/"))

        fputs("Loading: \(inputURL.path)\n", stderr)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        fputs("Page loaded. Waiting for fonts...\n", stderr)

        // Wait for fonts to load, then collect links and generate PDF
        webView.evaluateJavaScript("document.fonts.ready.then(() => true)") { _, _ in
            self.collectLinksAndMeasure()
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        fputs("Navigation failed: \(error.localizedDescription)\n", stderr)
        exit(1)
    }

    // ── Step 1: Collect internal link positions ────────────────────────────────

    func collectLinksAndMeasure() {
        webView.evaluateJavaScript(collectLinksJS) { result, error in
            if let jsonString = result as? String,
               let jsonData = jsonString.data(using: .utf8) {
                do {
                    self.linkMap = try JSONDecoder().decode(LinkMap.self, from: jsonData)
                    let linkCount = self.linkMap?.links.count ?? 0
                    let anchorCount = self.linkMap?.anchors.count ?? 0
                    fputs("Collected \(linkCount) internal links, \(anchorCount) anchor targets\n", stderr)
                } catch {
                    fputs("Warning: Failed to parse link map: \(error). Continuing without internal links.\n", stderr)
                }
            } else if let error = error {
                fputs("Warning: Link collection JS failed: \(error.localizedDescription). Continuing without internal links.\n", stderr)
            }

            // Step 2: Measure content height
            self.measureAndGeneratePDF()
        }
    }

    // ── Step 2: Measure content height ────────────────────────────────────────

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

    // ── Step 3: Generate PDF with createPDF() ─────────────────────────────────

    func generatePDF(height: CGFloat) {
        let pdfConfig = WKPDFConfiguration()
        // Set the rect to capture the full content area
        pdfConfig.rect = CGRect(x: 0, y: 0, width: width, height: height)

        fputs("Generating PDF: \(Int(width))x\(Int(height))px\n", stderr)

        webView.createPDF(configuration: pdfConfig) { result in
            switch result {
            case .success(let data):
                // Step 4: Post-process with PDFKit to add internal link annotations
                let finalData = self.addInternalLinks(to: data)

                do {
                    try finalData.write(to: self.outputURL)
                    let sizeKB = Double(finalData.count) / 1024.0
                    fputs("Output: \(self.outputURL.path)\n", stderr)
                    fputs("Size:   \(String(format: "%.1f", sizeKB)) KB\n", stderr)
                    exit(0)
                } catch {
                    fputs("Write error: \(error.localizedDescription)\n", stderr)
                    exit(1)
                }
            case .failure(let error):
                fputs("PDF generation failed: \(error.localizedDescription)\n", stderr)
                exit(1)
            }
        }
    }

    // ── Step 4: Post-process PDF with PDFKit ──────────────────────────────────
    //
    // Adds clickable link annotations for internal anchor references.
    //
    // Coordinate mapping:
    //   - HTML: origin top-left, Y increases downward
    //   - PDF:  origin bottom-left, Y increases upward
    //   - createPDF() maps 1 CSS pixel = 1 PDF point (scale factor 1.0)
    //   - Conversion: pdfY = pageHeight - htmlY
    //
    // Currently handles the pageless (single-page) case. For multi-page
    // support in the future, each link/anchor would need to be mapped
    // to its containing page based on page boundaries.

    func addInternalLinks(to pdfData: Data) -> Data {
        // If no links were collected, skip post-processing
        guard let linkMap = self.linkMap, !linkMap.links.isEmpty else {
            if self.linkMap == nil || self.linkMap!.links.isEmpty {
                fputs("No internal links found. Skipping post-processing.\n", stderr)
            }
            return pdfData
        }

        // Load the PDF with PDFKit
        guard let pdfDoc = PDFDocument(data: pdfData) else {
            fputs("Warning: Could not load PDF for post-processing. Output will lack internal links.\n", stderr)
            return pdfData
        }

        guard pdfDoc.pageCount > 0, let page = pdfDoc.page(at: 0) else {
            fputs("Warning: PDF has no pages. Skipping link post-processing.\n", stderr)
            return pdfData
        }

        let pageBounds = page.bounds(for: .mediaBox)
        let pdfPageHeight = pageBounds.height

        var addedCount = 0
        var skippedCount = 0

        for link in linkMap.links {
            // Look up the target anchor element
            guard let targetAnchor = linkMap.anchors[link.targetId] else {
                skippedCount += 1
                continue
            }

            // Convert source rect from HTML coordinates to PDF coordinates.
            // HTML: (x, y) is top-left of the rect, y increases downward.
            // PDF: (x, y) is bottom-left of the rect, y increases upward.
            let sourceBounds = CGRect(
                x: link.sourceRect.x,
                y: pdfPageHeight - link.sourceRect.y - link.sourceRect.height,
                width: link.sourceRect.width,
                height: link.sourceRect.height
            )

            // Convert destination point from HTML to PDF coordinates.
            // The destination Y is the top of the target element in HTML.
            // In PDF, this becomes pageHeight - targetY.
            let destY = pdfPageHeight - targetAnchor.y
            let destination = PDFDestination(page: page, at: CGPoint(x: 0, y: destY))

            // Create the link annotation
            let annotation = PDFAnnotation(
                bounds: sourceBounds,
                forType: .link,
                withProperties: nil
            )
            annotation.action = PDFActionGoTo(destination: destination)

            page.addAnnotation(annotation)
            addedCount += 1
        }

        fputs("Added \(addedCount) internal link annotations", stderr)
        if skippedCount > 0 {
            fputs(" (\(skippedCount) targets not found)", stderr)
        }
        fputs("\n", stderr)

        // Serialize the annotated PDF. Fall back to original data on failure.
        guard let annotatedData = pdfDoc.dataRepresentation() else {
            fputs("Warning: Could not serialize annotated PDF. Output will lack internal links.\n", stderr)
            return pdfData
        }

        return annotatedData
    }
}

// ── Main ────────────────────────────────────────────────────────────────────────

let absInput = (inputPath as NSString).standardizingPath
let absOutput: String
if (outputPath as NSString).isAbsolutePath {
    absOutput = outputPath
} else {
    absOutput = (FileManager.default.currentDirectoryPath as NSString)
        .appendingPathComponent(outputPath)
}

let generator = PDFGenerator(input: absInput, output: absOutput, width: pageWidth)

let app = NSApplication.shared
app.delegate = generator
app.run()
