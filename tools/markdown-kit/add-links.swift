#!/usr/bin/env swift
//
// add-links.swift — Add internal link annotations and bookmarks to a PDF using Apple PDFKit
//
// Usage:
//   add-links <pdf-path> <link-data.json> [--pageless] [--remove-existing]
//   add-links <pdf-path> --add-bookmarks <headings.json>
//
// Mode 1 (links): Reads a PDF and a JSON file containing link source rects and
// anchor targets (collected from the browser), then adds clickable link
// annotations using Apple's PDFKit framework. This guarantees compatibility
// with macOS Preview.app and iOS, since PDFKit is the same framework those
// apps use to read PDF annotations.
//
// Mode 2 (bookmarks): Reads a PDF and a JSON file containing heading data
// (title, level, id), searches for heading text in the PDF to find Y positions,
// then builds a hierarchical PDFOutline (bookmark sidebar). Used by the WebKit
// PDF path where no browser-collected position data is available.
//
// Link format:
//   Annotations use PDFAnnotation.destination (which PDFKit serializes as
//   an inline /Dest array on the annotation dictionary). This matches the
//   format that Chromium's own internal link annotations use and that
//   Preview.app handles correctly. Using only PDFActionGoTo (without
//   setting .destination) produces annotations that some PDF viewers
//   (notably macOS Preview) may not navigate correctly.
//
// The JSON format matches the output of COLLECT_LINKS_JS in markdown-kit.js:
// {
//   "links": [{ "sx": x, "sy": y, "sw": w, "sh": h, "targetId": "id" }, ...],
//   "anchors": { "id": { "x": x, "y": y, "w": w, "h": h }, ... },
//   "docHeight": number,
//   "viewportWidth": number
// }
//
// Bookmark JSON format (from markdown-kit.js WebKit path):
// [{ "title": "Section Title", "level": 1, "id": "section-title" }, ...]
//
// Coordinate mapping:
//   CSS: origin top-left, Y increases downward, units = CSS pixels
//   PDF: origin bottom-left, Y increases upward, units = points
//   Scale factor = 0.75 (CSS px to PDF pts = 72/96), fixed by Chromium spec
//
// Requires: macOS 11.0+ (for PDFKit)
//

import PDFKit
import Foundation

// ── JSON data structures ────────────────────────────────────────────────────

struct LinkSource: Codable {
    let sx: Double
    let sy: Double
    let sw: Double
    let sh: Double
    let targetId: String
}

struct AnchorRect: Codable {
    let x: Double
    let y: Double
    let w: Double
    let h: Double
}

struct LinkData: Codable {
    let links: [LinkSource]
    let anchors: [String: AnchorRect]
    let docHeight: Double?        // Retained for JSON compat; not used for scale
    let viewportWidth: Double?
}

// Heading data for bookmark mode (from markdown-kit.js WebKit path)
struct HeadingData: Codable {
    let title: String
    let level: Int
    let id: String
}

// ── Argument parsing ────────────────────────────────────────────────────────

let args = CommandLine.arguments

// Detect bookmark mode: add-links <pdf-path> --add-bookmarks <headings.json>
var bookmarkMode = false
var bookmarkJsonPath: String? = nil
if let idx = args.firstIndex(of: "--add-bookmarks"), idx + 1 < args.count {
    bookmarkMode = true
    bookmarkJsonPath = args[idx + 1]
}

guard args.count >= 3 else {
    fputs("""
    Usage: add-links <pdf-path> <link-data.json> [--pageless] [--remove-existing]
           add-links <pdf-path> --add-bookmarks <headings.json>

    Mode 1 — Links:
      Adds internal link annotations to a PDF using Apple PDFKit.
      Guarantees compatibility with macOS Preview.app and iOS.

    Mode 2 — Bookmarks:
      Adds PDF outline (bookmarks/table of contents) from heading data.
      Searches for heading text in the PDF to find precise Y positions.

    Options:
      --pageless          Single-page mode (default if PDF has 1 page)
      --remove-existing   Remove existing internal link annotations first
      --add-bookmarks     Add PDF outline from headings JSON file

    """, stderr)
    exit(1)
}

let pdfPath = args[1]
let forcePageless = args.contains("--pageless")
let removeExisting = args.contains("--remove-existing")

// ── Load PDF ──────────────────────────────────────────────────────────────

guard FileManager.default.fileExists(atPath: pdfPath) else {
    fputs("Error: PDF not found: \(pdfPath)\n", stderr)
    exit(1)
}

guard let pdfDoc = PDFDocument(url: URL(fileURLWithPath: pdfPath)) else {
    fputs("Error: Could not open PDF: \(pdfPath)\n", stderr)
    exit(1)
}

guard pdfDoc.pageCount > 0 else {
    fputs("Error: PDF has no pages\n", stderr)
    exit(1)
}

let pageCount = pdfDoc.pageCount
let isPageless = forcePageless || pageCount == 1

// ── Bookmark mode ─────────────────────────────────────────────────────────

if bookmarkMode {
    guard let bmJsonPath = bookmarkJsonPath else {
        fputs("Error: --add-bookmarks requires a JSON file path\n", stderr)
        exit(1)
    }

    guard FileManager.default.fileExists(atPath: bmJsonPath) else {
        fputs("Error: Headings JSON not found: \(bmJsonPath)\n", stderr)
        exit(1)
    }

    let bmJsonData = try Data(contentsOf: URL(fileURLWithPath: bmJsonPath))
    let headings = try JSONDecoder().decode([HeadingData].self, from: bmJsonData)

    if headings.isEmpty {
        print("{\"bookmarks\":0}")
        exit(0)
    }

    // ── Find heading positions via text search ──────────────────────────────
    // PDFDocument.findString() searches across all pages and returns
    // PDFSelection objects. We match headings in document order, requiring
    // each heading's match to come after the previous heading's position.
    // This prevents short titles like "Go" or "CSS" from matching body text
    // that appears earlier in the document.

    struct HeadingPosition {
        let heading: HeadingData
        let page: PDFPage
        let yPosition: CGFloat   // PDF Y coordinate (bottom-left origin)
    }

    var positions: [HeadingPosition] = []

    // Track the position of the last matched heading to enforce document order.
    // PDF coordinates: page index ascending, Y descending within each page
    // (Y increases upward, so earlier content has higher Y values).
    var lastPageIdx = 0
    var lastTopY = CGFloat.greatestFiniteMagnitude  // Start at "top" of document

    for heading in headings {
        let searchText = heading.title
        var found = false

        // PDFDocument.findString searches across all pages
        let selections = pdfDoc.findString(searchText, withOptions: .caseInsensitive)

        // Find the first match that comes AFTER the previous heading in document order.
        // Document order: page index ascending, then Y descending (top-to-bottom).
        var bestMatch: (page: PDFPage, pageIdx: Int, topY: CGFloat)? = nil

        for selection in selections {
            guard let selPage = selection.pages.first else { continue }
            let bounds = selection.bounds(for: selPage)
            guard bounds.width > 0 && bounds.height > 0 else { continue }

            let pageIdx = pdfDoc.index(for: selPage)
            let topY = bounds.origin.y + bounds.height

            // Check if this match comes after the previous heading in document order.
            // "After" means: later page, or same page with lower Y (further down).
            let isAfterPrev: Bool
            if pageIdx > lastPageIdx {
                isAfterPrev = true
            } else if pageIdx == lastPageIdx {
                // Same page: lower Y = further down the page (PDF Y increases upward)
                isAfterPrev = topY < lastTopY
            } else {
                isAfterPrev = false
            }

            if !isAfterPrev { continue }

            // Among valid matches, pick the one closest to the previous heading
            // (i.e., the earliest match after the previous position).
            if let current = bestMatch {
                if pageIdx < current.pageIdx {
                    bestMatch = (selPage, pageIdx, topY)
                } else if pageIdx == current.pageIdx && topY > current.topY {
                    // Same page: higher Y = closer to previous heading
                    bestMatch = (selPage, pageIdx, topY)
                }
            } else {
                bestMatch = (selPage, pageIdx, topY)
            }
        }

        if let match = bestMatch {
            positions.append(HeadingPosition(
                heading: heading,
                page: match.page,
                yPosition: match.topY
            ))
            lastPageIdx = match.pageIdx
            lastTopY = match.topY
            found = true
        }

        if !found {
            // Fallback: place at the last known position (keeps document order intact)
            if let page = pdfDoc.page(at: lastPageIdx) {
                positions.append(HeadingPosition(
                    heading: heading,
                    page: page,
                    yPosition: lastTopY
                ))
            }
        }
    }

    // ── Build hierarchical PDFOutline ────────────────────────────────────────
    // H1 = top-level, H2 = child of preceding H1, H3 = child of preceding H2.
    // Skipped levels (e.g., H1 then H3) make H3 a child of H1 — no synthetic nodes.

    let root = PDFOutline()

    // Track the outline node stack for nesting. Each entry is (level, outlineNode).
    // The root is at level 0.
    var stack: [(level: Int, node: PDFOutline)] = [(0, root)]

    for pos in positions {
        let item = PDFOutline()
        item.label = pos.heading.title
        item.destination = PDFDestination(
            page: pos.page,
            at: CGPoint(x: 0, y: pos.yPosition)
        )

        // Pop stack until we find a parent at a lower level
        while stack.count > 1 && stack.last!.level >= pos.heading.level {
            stack.removeLast()
        }

        let parent = stack.last!.node
        parent.insertChild(item, at: parent.numberOfChildren)
        stack.append((pos.heading.level, item))
    }

    // Set open/closed state: top-level items expanded, deeper items collapsed.
    // PDFOutline.isOpen controls whether children are visible in the sidebar.
    func setOpenState(_ node: PDFOutline, depth: Int) {
        // Top-level items (direct children of root) are open; deeper items closed
        node.isOpen = depth < 2
        for i in 0..<node.numberOfChildren {
            if let child = node.child(at: i) {
                setOpenState(child, depth: depth + 1)
            }
        }
    }
    setOpenState(root, depth: 0)

    pdfDoc.outlineRoot = root

    // Save
    guard let outputData = pdfDoc.dataRepresentation() else {
        fputs("Error: Could not serialize PDF\n", stderr)
        exit(1)
    }
    try outputData.write(to: URL(fileURLWithPath: pdfPath))

    print("{\"bookmarks\":\(positions.count)}")
    exit(0)
}

// ── Link mode: load link data JSON ────────────────────────────────────────

let jsonPath = args[2]

guard FileManager.default.fileExists(atPath: jsonPath) else {
    fputs("Error: JSON not found: \(jsonPath)\n", stderr)
    exit(1)
}

let jsonData = try Data(contentsOf: URL(fileURLWithPath: jsonPath))
let linkData = try JSONDecoder().decode(LinkData.self, from: jsonData)

// ── Remove existing internal link annotations ───────────────────────────────

var removedCount = 0

if removeExisting {
    for pageIdx in 0..<pageCount {
        guard let page = pdfDoc.page(at: pageIdx) else { continue }
        let toRemove = page.annotations.filter { annot in
            // Remove GoTo actions (internal links), keep URI actions (external links)
            if annot.action is PDFActionGoTo {
                return true
            }
            // Also remove annotations with /Dest (Chromium's named destinations)
            // PDFAnnotation doesn't expose raw /Dest directly, but we can check
            // if it has a destination property and no URL action
            if annot.destination != nil && annot.action == nil {
                return true
            }
            if annot.action == nil && annot.destination == nil {
                // Check if it's a link with no action — might be a broken one
                let subtype = annot.type
                if subtype == "Link" {
                    // Link with no action and no destination — remove it
                    return true
                }
            }
            return false
        }
        for annot in toRemove {
            page.removeAnnotation(annot)
            removedCount += 1
        }
    }
}

// ── Add internal link annotations ────────────────────────────────────────────

var addedCount = 0
var skippedCount = 0

if isPageless {
    // ── Pageless mode: single page ──────────────────────────────────────────
    guard let page = pdfDoc.page(at: 0) else {
        fputs("Error: Could not access page 0\n", stderr)
        exit(1)
    }

    let pageBounds = page.bounds(for: .mediaBox)
    let pageHeight = pageBounds.height

    // Scale factor: CSS pixels -> PDF points.
    // Chromium always converts CSS pixels to PDF points at 72/96 = 0.75.
    // This is a fixed ratio defined by the CSS spec (1px = 1/96 inch,
    // 1pt = 1/72 inch). Previous code computed scale as pageHeight/docHeight,
    // but scrollHeight can differ from the actual CSS height used for
    // page.pdf() (e.g., contentBottom + 10 vs scrollHeight), causing
    // annotation rects to drift from the rendered text — up to 20+ points
    // on tall documents.
    let scale = 0.75

    for link in linkData.links {
        guard let target = linkData.anchors[link.targetId] else {
            skippedCount += 1
            continue
        }

        // Source rect: convert from CSS (top-left origin) to PDF (bottom-left origin)
        let srcX = link.sx * scale
        let srcW = link.sw * scale
        let srcH = link.sh * scale
        let srcY = pageHeight - (link.sy * scale) - srcH

        let sourceBounds = CGRect(x: srcX, y: srcY, width: srcW, height: srcH)

        // Destination point: top of target element in PDF coordinates
        let destY = pageHeight - (target.y * scale)
        let destination = PDFDestination(page: page, at: CGPoint(x: 0, y: destY))

        // Create the annotation using PDFKit's native API.
        // Setting .destination (not just .action) causes PDFKit to write an
        // inline /Dest array on the annotation dictionary, which is the format
        // macOS Preview requires for reliable GoTo navigation.
        let annotation = PDFAnnotation(
            bounds: sourceBounds,
            forType: .link,
            withProperties: nil
        )
        annotation.destination = destination

        page.addAnnotation(annotation)
        addedCount += 1
    }
} else {
    // ── Paged mode: multiple pages (A4) ─────────────────────────────────────
    // Chromium re-lays out content for A4 pages. We compute the scale factor
    // from the PDF page dimensions and CSS viewport width.
    guard let firstPage = pdfDoc.page(at: 0) else {
        fputs("Error: Could not access page 0\n", stderr)
        exit(1)
    }

    let pageBounds = firstPage.bounds(for: .mediaBox)
    let pageWidth = pageBounds.width      // 595.92 pts for A4
    let pageHeightPts = pageBounds.height  // 841.92 pts for A4

    // A4 margins: 25mm top/bottom, 20mm left/right
    let marginTopPts = 25.0 * 72.0 / 25.4    // 70.87 pts
    let marginBotPts = 25.0 * 72.0 / 25.4
    let marginLeftPts = 20.0 * 72.0 / 25.4   // 56.69 pts
    let contentHeightPts = pageHeightPts - marginTopPts - marginBotPts
    let contentWidthPts = pageWidth - 2.0 * marginLeftPts

    // Scale: CSS viewport width -> PDF content width
    let viewportWidth = linkData.viewportWidth ?? 1400.0
    let xScale = contentWidthPts / viewportWidth
    let yScale = xScale  // Maintain aspect ratio

    // CSS content height per page
    let cssPerPage = contentHeightPts / yScale

    // Map CSS Y -> (pageIndex, pdfY)
    func cssToPaged(_ cssY: Double) -> (pageIndex: Int, pdfY: Double) {
        let pageIndex = Int(cssY / cssPerPage)
        let yWithinContent = cssY - Double(pageIndex) * cssPerPage
        let pdfYFromTop = marginTopPts + yWithinContent * yScale
        let pdfY = pageHeightPts - pdfYFromTop
        return (min(pageIndex, pageCount - 1), pdfY)
    }

    for link in linkData.links {
        guard let target = linkData.anchors[link.targetId] else {
            skippedCount += 1
            continue
        }

        // Source location
        let src = cssToPaged(link.sy)
        guard src.pageIndex >= 0, src.pageIndex < pageCount else { continue }
        guard let srcPage = pdfDoc.page(at: src.pageIndex) else { continue }

        let srcH = link.sh * yScale
        let srcX = marginLeftPts + link.sx * xScale
        let srcW = link.sw * xScale
        let srcY = src.pdfY - srcH

        // Clamp to page bounds
        let clampedX = max(0, min(srcX, pageWidth))
        let clampedY = max(0, min(srcY, pageHeightPts))
        let clampedX2 = max(0, min(srcX + srcW, pageWidth))
        let clampedY2 = max(0, min(srcY + srcH, pageHeightPts))
        guard clampedX2 > clampedX, clampedY2 > clampedY else { continue }

        let sourceBounds = CGRect(
            x: clampedX, y: clampedY,
            width: clampedX2 - clampedX, height: clampedY2 - clampedY
        )

        // Destination location
        let dest = cssToPaged(target.y)
        guard dest.pageIndex >= 0, dest.pageIndex < pageCount else { continue }
        guard let destPage = pdfDoc.page(at: dest.pageIndex) else { continue }

        let destination = PDFDestination(page: destPage, at: CGPoint(x: 0, y: dest.pdfY))

        let annotation = PDFAnnotation(
            bounds: sourceBounds,
            forType: .link,
            withProperties: nil
        )
        annotation.destination = destination

        srcPage.addAnnotation(annotation)
        addedCount += 1
    }
}

// ── Save the modified PDF ────────────────────────────────────────────────────

guard let outputData = pdfDoc.dataRepresentation() else {
    fputs("Error: Could not serialize PDF\n", stderr)
    exit(1)
}

try outputData.write(to: URL(fileURLWithPath: pdfPath))

// ── Output results as JSON for the calling script ───────────────────────────

let result: [String: Any] = [
    "added": addedCount,
    "removed": removedCount,
    "skipped": skippedCount,
]

// Simple JSON output (no Foundation JSONSerialization dependency issues)
print("{\"added\":\(addedCount),\"removed\":\(removedCount),\"skipped\":\(skippedCount)}")
