// noinspection JSAnnotator

/**
 * Visual tree snapshot generator.
 * Shows what sighted users see (complements ARIA tree which shows what AT sees).
 *
 * Key difference from ARIA tree:
 *   - aria-hidden elements ARE included (they're visible!)
 *   - opacity:0 elements are EXCLUDED (invisible to sighted users)
 *   - sr-only/clipped elements are EXCLUDED (invisible to sighted users)
 *
 * Arguments:
 *   arguments[0]: rootSelector - CSS selector for root element
 *   arguments[1]: includeUrls - boolean to include href values
 *   arguments[2]: includeHidden - boolean to include visually hidden content with markers
 *
 * Returns:
 *   Hierarchical tree structure with roles, names, and states.
 *
 *   Visibility markers (when includeHidden=true):
 *   - 'hidden' property with value: 'display-none', 'visibility-hidden',
 *     'visibility-collapse', 'opacity', 'clipped', 'offscreen', 'ancestor'
 *
 *   Note: aria-hidden is NOT a hiding mechanism in visual tree (it's an AT concept)
 */
function getVisualSnapshot(rootSelector, includeUrls, includeHidden = false) {
    const root = document.querySelector(rootSelector);
    if (!root) return null;

    // Skip non-content elements
    const SKIP_TAGS = ['SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT', 'TEMPLATE'];

    // Whitespace normalization
    function normalize(text) {
        return text ? text.replace(/\s+/g, ' ').trim() : '';
    }

    /**
     * Detect sr-only clip patterns (Bootstrap, Tailwind, WordPress, etc.)
     */
    function isSrOnlyClippedPattern(style) {
        if (style.position !== 'absolute' || style.overflow !== 'hidden') {
            return false;
        }
        if (style.clip && style.clip !== 'auto') return true;
        if (style.clipPath && style.clipPath !== 'none') return true;
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
        const threshold = 1000;
        for (const prop of ['left', 'right', 'top', 'bottom']) {
            const value = style[prop];
            if (value && value !== 'auto') {
                const num = parseFloat(value);
                if (Math.abs(num) > threshold) return true;
            }
        }
        // Check transform translate (pattern matches translate(), translateX/Y/Z(), translate3d())
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
     * Determine element's VISUAL visibility state.
     * Unlike ARIA tree, this checks what sighted users can see.
     *
     * Returns: { visible: boolean, marker: string | null }
     *
     * Markers (when not visible):
     *   'display-none', 'visibility-hidden', 'visibility-collapse',
     *   'opacity', 'clipped', 'offscreen', 'ancestor'
     *
     * Note: aria-hidden is NOT checked - it's an AT concept, not visual.
     */
    function getVisualState(el, ancestorHidden) {
        // Ancestor already hidden - propagate
        if (ancestorHidden) {
            return { visible: false, marker: 'ancestor' };
        }

        // Note: We do NOT check aria-hidden here - that's for AT, not visual rendering
        // aria-hidden elements ARE visible to sighted users

        // Check inert - removes from interaction AND visual focus indication
        // But the content is still visually rendered, so we don't hide it
        // (inert is more of an interaction concept than visual)

        // CSS properties that affect visual visibility
        let style;
        try {
            style = window.getComputedStyle(el);
        } catch (e) {
            return { visible: true, marker: null };
        }

        if (style.display === 'none') {
            return { visible: false, marker: 'display-none' };
        }
        if (style.visibility === 'hidden') {
            return { visible: false, marker: 'visibility-hidden' };
        }
        if (style.visibility === 'collapse') {
            return { visible: false, marker: 'visibility-collapse' };
        }
        // opacity:0 - invisible to sighted users (unlike ARIA tree where it's in AT)
        if (style.opacity === '0') {
            return { visible: false, marker: 'opacity' };
        }
        // sr-only clip patterns - invisible to sighted users
        if (isSrOnlyClippedPattern(style)) {
            return { visible: false, marker: 'clipped' };
        }
        // offscreen positioning - not in viewport
        if (isOffscreenPositioned(style)) {
            return { visible: false, marker: 'offscreen' };
        }

        return { visible: true, marker: null };
    }

    // Accessible name computation (same as ARIA tree)
    function computeAccessibleName(el) {
        if (el.getAttribute('aria-label')) {
            return normalize(el.getAttribute('aria-label'));
        }
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
        if (el.id) {
            const label = document.querySelector(`label[for="${el.id}"]`);
            if (label) return normalize(label.textContent);
        }
        if (el.closest('label')) {
            return normalize(el.closest('label').textContent);
        }
        const tagName = el.tagName.toLowerCase();
        if (['button', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
            return normalize(el.textContent);
        }
        if (el.getAttribute('title')) {
            return normalize(el.getAttribute('title'));
        }
        if (tagName === 'img') {
            return normalize(el.getAttribute('alt') || '');
        }
        if (['input', 'textarea'].includes(tagName)) {
            return normalize(el.placeholder || el.value || '');
        }
        return '';
    }

    // Implicit role mapping
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

    // Walk tree and build hierarchical snapshot
    function walkTree(el, depth = 0, ancestorHidden = false) {
        if (depth > 50) return null;

        // Handle text nodes
        if (el.nodeType === Node.TEXT_NODE) {
            const text = normalize(el.textContent);
            if (text) {
                return { type: 'text', content: text };
            }
            return null;
        }

        if (el.nodeType !== Node.ELEMENT_NODE) return null;
        if (SKIP_TAGS.includes(el.tagName)) return null;

        // Check VISUAL visibility (different from ARIA visibility!)
        const visState = getVisualState(el, ancestorHidden);

        // Skip visually hidden elements unless includeHidden is true
        if (!visState.visible && !includeHidden) {
            return null;
        }

        const role = el.getAttribute('role') || getImplicitRole(el);
        const name = computeAccessibleName(el);

        const node = { role: role };

        if (name) {
            node.name = name;
        }

        // Add hidden marker when including hidden content
        if (!visState.visible && visState.marker) {
            node.hidden = visState.marker;
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

        // Add checked state
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

        // Add URL for links
        if (includeUrls && role === 'link' && el.href) {
            node.url = el.href;
        }

        // Propagate hidden state to children
        const childAncestorHidden = !visState.visible || ancestorHidden;

        // Process children
        const children = [];
        for (const child of el.childNodes) {
            const childNode = walkTree(child, depth + 1, childAncestorHidden);
            if (childNode) {
                children.push(childNode);
            }
        }

        if (children.length > 0) {
            node.children = children;
        }

        return node;
    }

    return walkTree(root);
}

return getVisualSnapshot(arguments[0], arguments[1], arguments[2]);
