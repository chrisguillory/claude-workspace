/**
 * ARIA accessibility tree snapshot generator.
 * Implements accessible name computation per WAI-ARIA spec.
 *
 * Arguments:
 *   arguments[0]: rootSelector - CSS selector for root element
 *   arguments[1]: includeUrls - boolean to include href values
 *   arguments[2]: includeHidden - boolean to include content not in accessibility tree
 *
 * Returns:
 *   Hierarchical tree structure with roles, names, and states.
 *
 *   Visibility markers (single string, not array):
 *   - 'hidden' property: Element NOT in accessibility tree
 *     Values: 'display-none', 'visibility-hidden', 'visibility-collapse',
 *             'aria-hidden', 'inert', 'ancestor'
 *   - 'visuallyHidden' property: Element IN accessibility tree but not visible
 *     Values: 'opacity', 'clipped', 'offscreen'
 *
 *   With includeHidden=false (default): Excludes hidden:*, includes visuallyHidden:*
 *   With includeHidden=true: Includes all elements with appropriate markers
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
     * Detect sr-only clip patterns (Bootstrap, Tailwind, WordPress, etc.)
     * Requires position:absolute AND overflow:hidden as prerequisites.
     */
    function isSrOnlyClippedPattern(style) {
        if (style.position !== 'absolute' || style.overflow !== 'hidden') {
            return false;
        }

        // Check for clip or clip-path (most reliable indicator)
        if (style.clip && style.clip !== 'auto') return true;
        if (style.clipPath && style.clipPath !== 'none') return true;

        // Check for 1px x 1px dimensions (Bootstrap/Tailwind pattern)
        const width = parseFloat(style.width);
        const height = parseFloat(style.height);
        if (width <= 1 && height <= 1) return true;

        return false;
    }

    /**
     * Detect off-screen positioning patterns.
     * Requires position:absolute or position:fixed as prerequisite.
     */
    function isOffscreenPositioned(style) {
        if (style.position !== 'absolute' && style.position !== 'fixed') return false;

        const threshold = 1000; // pixels - anything beyond this is "off-screen"

        // Check left/right/top/bottom
        for (const prop of ['left', 'right', 'top', 'bottom']) {
            const value = style[prop];
            if (value && value !== 'auto') {
                const num = parseFloat(value);
                if (Math.abs(num) > threshold) return true;
            }
        }

        // Check transform translate (less common but used)
        // Pattern matches: translate(), translateX/Y/Z(), translate3d()
        if (style.transform && style.transform !== 'none') {
            const match = style.transform.match(/translate(?:3d|[XYZ])?\(([^)]+)\)/);
            if (match) {
                const values = match[1].split(',').map(v => parseFloat(v));
                if (values.some(v => Math.abs(v) > threshold)) return true;
            }
        }

        return false;
    }

    /**
     * Determine element's visibility state for accessibility tree purposes.
     *
     * Returns: { inAT: boolean, marker: string | null }
     *   - inAT: true if element is in accessibility tree
     *   - marker: visibility reason (or null if fully visible)
     *
     * Hidden markers (NOT in accessibility tree):
     *   'ancestor', 'aria-hidden', 'inert', 'display-none',
     *   'visibility-hidden', 'visibility-collapse'
     *
     * VisuallyHidden markers (IN accessibility tree but not visible):
     *   'opacity', 'clipped', 'offscreen'
     *
     * @param {Element} el - The element to check
     * @param {boolean} ancestorHidden - Whether an ancestor is hidden
     * @param {string|null} ancestorHiddenReason - Why ancestor is hidden (for visibility override check)
     */
    function getVisibilityState(el, ancestorHidden, ancestorHiddenReason) {
        // Handle visibility:visible override of inherited visibility:hidden
        // Per CSS spec, a child with visibility:visible IS visible even if parent has visibility:hidden
        // This ONLY applies to visibility inheritance, not to display:none, aria-hidden, inert, etc.
        if (ancestorHidden && (ancestorHiddenReason === 'visibility-hidden' || ancestorHiddenReason === 'visibility-collapse')) {
            let style;
            try {
                style = window.getComputedStyle(el);
            } catch (e) {
                return { inAT: false, marker: 'ancestor' };
            }
            // Child with visibility:visible overrides inherited visibility:hidden
            if (style.visibility === 'visible') {
                // Continue to check element's own conditions below (don't return ancestor marker)
                ancestorHidden = false;
            }
        }

        // Ancestor hidden (and not overridden) - propagate with 'ancestor' marker
        if (ancestorHidden) {
            return { inAT: false, marker: 'ancestor' };
        }

        // Check element's own NOT-in-AT conditions
        if (el.getAttribute('aria-hidden') === 'true') {
            return { inAT: false, marker: 'aria-hidden' };
        }
        if (el.hasAttribute('inert')) {
            return { inAT: false, marker: 'inert' };
        }

        // CSS properties that remove from accessibility tree
        let style;
        try {
            style = window.getComputedStyle(el);
        } catch (e) {
            // getComputedStyle can fail on some elements, treat as visible
            return { inAT: true, marker: null };
        }

        if (style.display === 'none') {
            return { inAT: false, marker: 'display-none' };
        }
        if (style.visibility === 'hidden') {
            return { inAT: false, marker: 'visibility-hidden' };
        }
        if (style.visibility === 'collapse') {
            return { inAT: false, marker: 'visibility-collapse' };
        }

        // Check IN-AT-but-visually-hidden conditions
        if (style.opacity === '0') {
            return { inAT: true, marker: 'opacity' };
        }
        if (isSrOnlyClippedPattern(style)) {
            return { inAT: true, marker: 'clipped' };
        }
        if (isOffscreenPositioned(style)) {
            return { inAT: true, marker: 'offscreen' };
        }

        // Fully visible and in AT
        return { inAT: true, marker: null };
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
    // ancestorHidden: true if any ancestor is not in accessibility tree
    // ancestorHiddenReason: why ancestor is hidden (for visibility:visible override check)
    function walkTree(el, depth = 0, ancestorHidden = false, ancestorHiddenReason = null) {
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

        // Detect visibility state (checks ancestor state and element's own state)
        const visState = getVisibilityState(el, ancestorHidden, ancestorHiddenReason);

        // For visibility:hidden/collapse, must traverse children for potential overrides.
        // CSS allows visibility:visible children to override inherited visibility:hidden.
        const isVisibilityHiddenType =
            visState.marker === 'visibility-hidden' || visState.marker === 'visibility-collapse';

        // Skip elements NOT in accessibility tree unless:
        // 1. includeHidden is true, OR
        // 2. Element has visibility:hidden/collapse (children may override)
        if (!visState.inAT && !includeHidden && !isVisibilityHiddenType) {
            return null;
        }

        const role = el.getAttribute('role') || getImplicitRole(el);
        const name = computeAccessibleName(el);

        const node = { role: role };

        // Add name if available
        if (name) {
            node.name = name;
        }

        // Add visibility markers based on state
        if (visState.marker) {
            if (visState.inAT) {
                // In accessibility tree but visually hidden
                node.visuallyHidden = visState.marker;
            } else {
                // Not in accessibility tree (only shown when includeHidden=true)
                node.hidden = visState.marker;
            }
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

        // Propagate hidden state to children
        // Children of hidden elements inherit the hidden state
        const childAncestorHidden = !visState.inAT || ancestorHidden;
        // Track the reason for hiding (needed for visibility:visible override check)
        // If this element is hidden, use its marker; otherwise propagate parent's reason
        const childAncestorReason = !visState.inAT ? visState.marker : ancestorHiddenReason;

        // Process child NODES (not just elements - includes text!)
        const children = [];
        for (const child of el.childNodes) {
            const childNode = walkTree(child, depth + 1, childAncestorHidden, childAncestorReason);
            if (childNode) {
                children.push(childNode);
            }
        }

        // Add children array if not empty
        if (children.length > 0) {
            node.children = children;
        }

        // For visibility:hidden without includeHidden, only include if we have visible children.
        // Return the children directly (skip the hidden parent wrapper).
        if (!visState.inAT && !includeHidden && isVisibilityHiddenType) {
            // Filter to only visible children (those without hidden marker)
            const visibleChildren = children.filter(c =>
                c.type === 'text' || !c.hidden
            );
            if (visibleChildren.length === 0) {
                return null;  // No visible children, skip this subtree
            }
            // Return visible children without hidden parent wrapper
            if (visibleChildren.length === 1) {
                return visibleChildren[0];
            }
            // Multiple visible children - wrap in structural generic
            return { role: 'generic', children: visibleChildren };
        }

        return node;
    }

    return walkTree(root);
}

return getAccessibilitySnapshot(arguments[0], arguments[1], arguments[2]);
