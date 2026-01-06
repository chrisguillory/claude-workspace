/**
 * ARIA accessibility tree snapshot generator.
 * Implements accessible name computation per WAI-ARIA spec.
 *
 * Arguments:
 *   arguments[0]: rootSelector - CSS selector for root element
 *   arguments[1]: includeUrls - boolean to include href values
 *   arguments[2]: includeHidden - boolean to include hidden content with markers
 *
 * Returns:
 *   Hierarchical tree structure with roles, names, and states.
 *   When includeHidden=true, hidden elements include a 'hidden' array
 *   indicating the hiding mechanism(s): 'aria', 'inert', or 'css'.
 */
function getAccessibilitySnapshot(rootSelector, includeUrls, includeHidden = false) {
    const root = document.querySelector(rootSelector);
    if (!root) return null;

    // Skip non-content elements (never included regardless of includeHidden)
    const SKIP_TAGS = ['SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT', 'TEMPLATE'];

    // Shared whitespace normalization helper
    // Per WAI-ARIA 1.2 and CSS Text Module Level 3
    function normalize(text) {
        return text ? text.replace(/\s+/g, ' ').trim() : '';
    }

    /**
     * Detect hiding mechanisms on an element.
     * Returns an array of hiding reasons: 'aria', 'inert', 'css'
     * Empty array means element is not hidden.
     *
     * Note: .sr-only patterns (clip, offscreen positioning) are NOT detected
     * as hidden because they ARE in the accessibility tree - they're just
     * visually hidden for sighted users.
     */
    function getHiddenReasons(el) {
        const reasons = [];

        // 1. aria-hidden="true" - explicit accessibility hiding
        // Note: aria-hidden="false" on children cannot override parent's true
        // But we don't need to check ancestors here because if parent is skipped,
        // we never reach children (recursive early return)
        if (el.getAttribute('aria-hidden') === 'true') {
            reasons.push('aria');
        }

        // 2. inert attribute - removes from accessibility tree AND prevents interaction
        // Check attribute directly - works in all browsers, reflects property changes
        if (el.hasAttribute('inert')) {
            reasons.push('inert');
        }

        // 3. CSS hiding - removes from visual rendering and accessibility tree
        // Note: We include opacity:0 here because from AI consumer perspective,
        // if something is invisible, it's "hidden" even if technically in AT.
        // This helps AI agents understand visual state.
        try {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' ||
                style.visibility === 'hidden' ||
                style.opacity === '0') {
                reasons.push('css');
            }
        } catch (e) {
            // getComputedStyle can fail on some elements, skip CSS check
        }

        return reasons;
    }

    // Accessible name computation per WAI-ARIA spec
    // IMPORTANT: This uses getElementById which accesses hidden elements.
    // Per W3C, aria-labelledby references MUST include hidden element content.
    function computeAccessibleName(el) {
        // Step 1: aria-label
        if (el.getAttribute('aria-label')) {
            return normalize(el.getAttribute('aria-label'));
        }

        // Step 2: aria-labelledby
        // Note: Referenced elements are accessed via getElementById regardless of
        // their hidden state. This is W3C-mandated behavior.
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

        // Always skip non-content elements
        if (SKIP_TAGS.includes(el.tagName)) return null;

        // Detect hidden state
        const hiddenReasons = getHiddenReasons(el);
        const isHidden = hiddenReasons.length > 0;

        // Skip hidden elements unless includeHidden is true
        // This is an early return that prevents recursion into hidden subtrees
        if (isHidden && !includeHidden) {
            return null;
        }

        const role = el.getAttribute('role') || getImplicitRole(el);
        const name = computeAccessibleName(el);

        const node = { role: role };

        // Add name if available
        if (name) {
            node.name = name;
        }

        // Add hidden markers when including hidden content
        // This tells the AI consumer WHY this element is hidden
        if (isHidden && includeHidden) {
            node.hidden = hiddenReasons;
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

return getAccessibilitySnapshot(arguments[0], arguments[1], arguments[2]);
