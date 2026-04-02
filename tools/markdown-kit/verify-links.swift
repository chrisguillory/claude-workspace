//
// verify-preview-links.swift — Verify PDF annotations work for Preview compatibility
//
// Checks that all GoTo link annotations have the inline /Dest array that
// macOS Preview.app requires for reliable navigation.
//

import PDFKit
import AppKit
import Foundation

func verify(path: String) -> Bool {
    let filename = (path as NSString).lastPathComponent

    guard let doc = PDFDocument(url: URL(fileURLWithPath: path)) else {
        print("FAIL: Cannot open \(filename)")
        return false
    }

    var totalGoTo = 0
    var withDest = 0
    var withAction = 0
    var withBoth = 0
    var validDests = 0
    var invalidDests = 0
    var totalLinks = 0
    var externalLinks = 0

    for pageIdx in 0..<doc.pageCount {
        guard let page = doc.page(at: pageIdx) else { continue }

        for annot in page.annotations {
            guard annot.type == "Link" else { continue }
            totalLinks += 1

            // External link
            if annot.action is PDFActionURL || annot.url != nil {
                externalLinks += 1
                continue
            }

            let hasDest = annot.destination != nil
            let hasGoTo = annot.action is PDFActionGoTo

            if hasDest || hasGoTo {
                totalGoTo += 1
                if hasDest { withDest += 1 }
                if hasGoTo { withAction += 1 }
                if hasDest && hasGoTo { withBoth += 1 }
            }

            // Verify destination resolves
            if let dest = annot.destination {
                if dest.page != nil {
                    validDests += 1
                } else {
                    invalidDests += 1
                }
            } else if let goTo = annot.action as? PDFActionGoTo {
                if goTo.destination.page != nil {
                    validDests += 1
                } else {
                    invalidDests += 1
                }
            }
        }
    }

    print("\(filename):")
    print("  Pages:       \(doc.pageCount)")
    print("  Total links: \(totalLinks)")
    print("  External:    \(externalLinks)")
    print("  Internal:    \(totalGoTo)")
    print("    with /Dest:    \(withDest)")
    print("    with /A GoTo:  \(withAction)")
    print("    with both:     \(withBoth)")
    print("  Valid dests: \(validDests)")
    print("  Invalid:     \(invalidDests)")

    let allHaveDest = withDest == totalGoTo
    let allValid = invalidDests == 0

    if allHaveDest && allValid {
        print("  STATUS: PASS - All internal links have inline /Dest (Preview-compatible)")
    } else if !allHaveDest {
        print("  STATUS: WARN - \(totalGoTo - withDest) links missing inline /Dest (may not work in Preview)")
    }
    if !allValid {
        print("  STATUS: FAIL - \(invalidDests) links have unresolvable destinations")
    }

    // Test navigation with PDFView
    let pdfView = PDFView(frame: NSRect(x: 0, y: 0, width: 800, height: 600))
    pdfView.document = doc

    if let page = doc.page(at: 0) {
        var navSuccess = 0
        var navFail = 0
        for annot in page.annotations {
            if let dest = annot.destination, dest.page != nil {
                pdfView.go(to: dest)
                navSuccess += 1
            } else if let goTo = annot.action as? PDFActionGoTo,
                      goTo.destination.page != nil {
                pdfView.go(to: goTo.destination)
                navSuccess += 1
            }
        }
        print("  PDFView nav: \(navSuccess) successful navigations")
    }

    print()
    return allHaveDest && allValid
}

let app = NSApplication.shared

print("PDF Link Verification Report")
print("============================\n")

let chromiumPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "~/claude-workspace/tools/markdown-kit/rendering-test.pdf"

let passed = verify(path: chromiumPath)
exit(passed ? 0 : 1)
