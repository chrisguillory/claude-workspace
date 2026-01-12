// noinspection JSAnnotator

/**
 * Smart text extraction from DOM elements.
 * Supports auto-detection of main content (main > article > body fallback).
 *
 * Arguments:
 *   arguments[0]: CSS selector or 'auto' for smart extraction
 *
 * Returns:
 *   Object with text, title, url, sourceElement, characterCount, and smart extraction metadata
 */
function extractFromElement(root) {
    const SKIP_ELEMENTS = new Set([
        'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS',
        'IFRAME', 'OBJECT', 'EMBED', 'AUDIO', 'VIDEO', 'MAP', 'HEAD'
    ]);
    const PREFORMATTED_ELEMENTS = new Set(['PRE', 'CODE', 'TEXTAREA']);

    // Block-level elements that create visual separation
    // These should have spaces between their content and adjacent content
    const BLOCK_ELEMENTS = new Set([
        'ADDRESS', 'ARTICLE', 'ASIDE', 'BLOCKQUOTE', 'DD', 'DETAILS', 'DIALOG',
        'DIV', 'DL', 'DT', 'FIELDSET', 'FIGCAPTION', 'FIGURE', 'FOOTER', 'FORM',
        'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HEADER', 'HGROUP', 'HR', 'LI',
        'MAIN', 'NAV', 'OL', 'P', 'PRE', 'SECTION', 'TABLE', 'TBODY', 'TD',
        'TFOOT', 'TH', 'THEAD', 'TR', 'UL', 'BUTTON', 'OPTION', 'SUMMARY'
    ]);

    const parts = [];
    let depth = 0;
    let inPreformatted = 0;
    const MAX_DEPTH = 100;

    function walk(node) {
        if (!node) return;
        depth++;
        if (depth > MAX_DEPTH) { depth--; return; }

        try {
            if (node.nodeType === Node.ELEMENT_NODE) {
                const tagName = node.tagName;
                if (SKIP_ELEMENTS.has(tagName)) return;

                // Handle IMG elements - emit marker with alt text
                if (tagName === 'IMG') {
                    const alt = node.getAttribute('alt');
                    const marker = alt && alt.trim()
                        ? '__IMG_ALT__' + alt.trim() + '__END_IMG__'
                        : '__IMG_ALT__(no alt)__END_IMG__';
                    parts.push({text: marker, pre: false, block: false});
                    return;  // IMG has no children to walk
                }

                const isPre = PREFORMATTED_ELEMENTS.has(tagName) ||
                    (window.getComputedStyle &&
                     ['pre', 'pre-wrap', 'pre-line'].includes(
                         window.getComputedStyle(node).whiteSpace));
                const isBlock = BLOCK_ELEMENTS.has(tagName);

                if (isPre) inPreformatted++;

                if (node.shadowRoot) {
                    for (const child of node.shadowRoot.childNodes) {
                        walk(child);
                    }
                }

                for (const child of node.childNodes) {
                    walk(child);
                }

                if (isPre) inPreformatted--;

                // After processing a block element's children, mark boundary
                // This ensures "<button>A</button><button>B</button>" becomes "A B"
                if (isBlock && parts.length > 0) {
                    parts[parts.length - 1].blockEnd = true;
                }
            }
            else if (node.nodeType === Node.TEXT_NODE) {
                const text = node.textContent;
                if (text) {
                    parts.push({text, pre: inPreformatted > 0, block: false});
                }
            }
            else if (node.nodeType === Node.DOCUMENT_FRAGMENT_NODE) {
                for (const child of node.childNodes) {
                    walk(child);
                }
            }
        } finally {
            depth--;
        }
    }

    walk(root);

    // Join parts, adding spaces only at block boundaries
    // - Block elements (div, button, p, etc.) get spaces between them
    // - Inline elements (span, strong, em) stay adjacent: "Hello<b>World</b>" -> "HelloWorld"
    let result = '';
    for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        let text = part.pre ? part.text : part.text.replace(/[\s\n\r\t\u00A0]+/g, ' ');

        // Add space before this part if previous part ended a block
        if (i > 0 && parts[i - 1].blockEnd) {
            // Ensure space between block boundaries
            if (result.length > 0 && !result.endsWith(' ') && text.length > 0 && !text.startsWith(' ')) {
                result += ' ';
            }
        }

        result += text;
    }

    // Final normalization: collapse multiple spaces, trim
    result = result.replace(/ +/g, ' ').trim();

    const MAX_SIZE = 5 * 1024 * 1024;
    if (result.length > MAX_SIZE) {
        result = result.substring(0, MAX_SIZE) + ' [Content truncated at 5MB limit]';
    }

    return result;
}

function extractAllText(requestedSelector) {
    const SMART_THRESHOLD = 500;  // Min chars for smart extraction to use an element

    // Calculate body character count for coverage calculation
    const bodyCharCount = extractFromElement(document.body).length;

    // Smart extraction mode
    if (requestedSelector === 'auto') {
        // Priority 1: Try <main> element or [role="main"]
        const main = document.querySelector('main, [role="main"]');
        if (main) {
            const mainText = extractFromElement(main);
            if (mainText.length >= SMART_THRESHOLD) {
                return {
                    text: mainText,
                    title: document.title || '',
                    url: window.location.href,
                    sourceElement: main.tagName.toLowerCase() === 'main' ? 'main' : '[role="main"]',
                    characterCount: mainText.length,
                    isSmartExtraction: true,
                    fallbackUsed: false,
                    bodyCharacterCount: bodyCharCount
                };
            }
        }

        // Priority 2: Try <article> element
        const article = document.querySelector('article');
        if (article) {
            const articleText = extractFromElement(article);
            if (articleText.length >= SMART_THRESHOLD) {
                return {
                    text: articleText,
                    title: document.title || '',
                    url: window.location.href,
                    sourceElement: 'article',
                    characterCount: articleText.length,
                    isSmartExtraction: true,
                    fallbackUsed: false,
                    bodyCharacterCount: bodyCharCount
                };
            }
        }

        // Fallback: Use body
        const bodyText = extractFromElement(document.body);
        return {
            text: bodyText,
            title: document.title || '',
            url: window.location.href,
            sourceElement: 'body',
            characterCount: bodyText.length,
            isSmartExtraction: true,
            fallbackUsed: true,
            bodyCharacterCount: bodyCharCount
        };
    }

    // Explicit selector mode
    const root = document.querySelector(requestedSelector);
    if (!root) {
        return {
            error: 'Selector not found: ' + requestedSelector,
            title: document.title || '',
            url: window.location.href
        };
    }

    const text = extractFromElement(root);
    return {
        text: text,
        title: document.title || '',
        url: window.location.href,
        sourceElement: requestedSelector,
        characterCount: text.length,
        isSmartExtraction: false
    };
}

return extractAllText(arguments[0]);
