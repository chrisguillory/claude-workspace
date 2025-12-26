// noinspection JSUnresolvedReference
/**
 * ARIA accessibility tree snapshot generator.
 * Implements accessible name computation per WAI-ARIA spec.
 *
 * Executed via Selenium's driver.execute_script() which wraps code in an
 * anonymous function. IIFE wrapper makes this valid standalone JavaScript.
 *
 * @param {string} arguments[0] - CSS selector for root element
 * @param {boolean} arguments[1] - Whether to include href values
 * @returns {Object} Hierarchical tree structure with roles, names, and states
 */
(function() {

function getAccessibilitySnapshot(rootSelector, includeUrls) {
    const root = document.querySelector(rootSelector);
    if (!root) return null;

    // Skip non-rendered elements
    const SKIP_TAGS = ['SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT'];

    function isVisible(el) {
        const style = window.getComputedStyle(el);
        return style.display !== 'none' &&
               style.visibility !== 'hidden' &&
               style.opacity !== '0';
    }

    // Shared whitespace normalization helper
    // Per WAI-ARIA 1.2 and CSS Text Module Level 3
    function normalize(text) {
        return text ? text.replace(/\s+/g, ' ').trim() : '';
    }

    // Accessible name computation per WAI-ARIA spec
    function computeAccessibleName(el) {
        // Step 1: aria-label
        if (el.getAttribute('aria-label')) {
            return normalize(el.getAttribute('aria-label'));
        }

        // Step 2: aria-labelledby
        if (el.getAttribute('aria-labelledby')) {
            const ids = el.getAttribute('aria-labelledby').split(/\s+/);
            return ids
                .map(id => {
                    const refEl = document.getElementById(id);
                    return refEl ? normalize(refEl.textContent) : '';
                })
                .filter(Boolean)
                .join(' ');
        }

        // Step 3: Label element association
        if (el.id) {
            const label = document.querySelector(`label[for="${el.id}"]`);
            if (label) return normalize(label.textContent);
        }

        // Step 4: Implicit label (form control inside label)
        if (el.closest('label')) {
            return normalize(el.closest('label').textContent);
        }

        // Step 5: Element content for links, buttons, headings
        const tagName = el.tagName.toLowerCase();
        if (['button', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
            return normalize(el.textContent);
        }

        // Step 6: Title attribute
        if (el.getAttribute('title')) {
            return normalize(el.getAttribute('title'));
        }

        // Step 7: Alt text for images
        if (tagName === 'img') {
            return normalize(el.getAttribute('alt') || '');
        }

        // Step 8: Placeholder for inputs
        if (['input', 'textarea'].includes(tagName)) {
            return normalize(el.placeholder || el.value || '');
        }

        return '';
    }

    // Implicit role mapping per HTML AAM spec
    function getImplicitRole(el) {
        const tagName = el.tagName.toLowerCase();
        const type = el.getAttribute('type')?.toLowerCase();

        const implicitRoles = {
            'a': 'link',
            'button': 'button',
            'h1': 'heading', 'h2': 'heading', 'h3': 'heading',
            'h4': 'heading', 'h5': 'heading', 'h6': 'heading',
            'header': 'banner',
            'footer': 'contentinfo',
            'nav': 'navigation',
            'main': 'main',
            'article': 'article',
            'section': 'region',
            'aside': 'complementary',
            'form': 'form',
            'p': 'paragraph',
            'input': type === 'checkbox' ? 'checkbox' : type === 'radio' ? 'radio' : 'textbox',
            'textarea': 'textbox',
            'select': 'combobox',
            'ul': 'list',
            'ol': 'list',
            'li': 'listitem',
            'table': 'table',
            'tr': 'row',
            'td': 'cell',
            'th': 'columnheader',
            'img': 'img',
            'strong': 'strong',
            'em': 'emphasis',
            'code': 'code'
        };

        return implicitRoles[tagName] || 'generic';
    }

    // Walk tree and build hierarchical snapshot (includes text nodes!)
    function walkTree(el, depth = 0) {
        if (depth > 50) return null; // Prevent infinite recursion

        // Handle text nodes
        if (el.nodeType === Node.TEXT_NODE) {
            const text = normalize(el.textContent);
            if (text) {
                return { type: 'text', content: text };
            }
            return null;
        }

        // Skip non-element nodes
        if (el.nodeType !== Node.ELEMENT_NODE) return null;

        // Skip non-rendered elements
        if (SKIP_TAGS.includes(el.tagName)) return null;

        // Skip hidden elements
        if (!isVisible(el)) return null;

        const role = el.getAttribute('role') || getImplicitRole(el);
        const name = computeAccessibleName(el);

        const node = { role: role };

        // Add name if available
        if (name) {
            node.name = name;
        }

        // Add description if available
        if (el.getAttribute('aria-description')) {
            node.description = el.getAttribute('aria-description');
        }

        // Add level for headings
        if (role === 'heading') {
            const match = el.tagName.match(/h([1-6])/i);
            if (match) {
                node.level = parseInt(match[1]);
            }
        }

        // Add checked state for checkboxes/radios
        if (['checkbox', 'radio', 'switch'].includes(role)) {
            if (el.hasAttribute('aria-checked')) {
                node.checked = el.getAttribute('aria-checked') === 'true';
            } else if (el.tagName === 'INPUT') {
                node.checked = el.checked;
            }
        }

        // Add disabled state
        if (el.hasAttribute('aria-disabled')) {
            node.disabled = el.getAttribute('aria-disabled') === 'true';
        } else if (['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) {
            node.disabled = el.disabled;
        }

        // Add URL for links if requested
        if (includeUrls && role === 'link' && el.href) {
            node.url = el.href;
        }

        // Process child NODES (not just elements - includes text!)
        const children = [];
        for (const child of el.childNodes) {
            const childNode = walkTree(child, depth + 1);
            if (childNode) {
                children.push(childNode);
            }
        }

        // Add children array if not empty
        if (children.length > 0) {
            node.children = children;
        }

        return node;
    }

    return walkTree(root);
}

return getAccessibilitySnapshot(arguments[0], arguments[1]);

}).apply(null, arguments);
