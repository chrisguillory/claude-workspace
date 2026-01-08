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
     *
     * @param {Element} el - The element to check
     * @param {boolean} ancestorHidden - Whether an ancestor is hidden
     * @param {string|null} ancestorHiddenReason - Why ancestor is hidden (for visibility override check)
     */
    function getVisualState(el, ancestorHidden, ancestorHiddenReason) {
        // Handle visibility:visible override of inherited visibility:hidden
        // Per CSS spec, a child with visibility:visible IS visible even if parent has visibility:hidden
        // This ONLY applies to visibility inheritance, not to display:none, opacity:0, etc.
        if (ancestorHidden && (ancestorHiddenReason === 'visibility-hidden' || ancestorHiddenReason === 'visibility-collapse')) {
            let style;
            try {
                style = window.getComputedStyle(el);
            } catch (e) {
                return { visible: false, marker: 'ancestor' };
            }
            // Child with visibility:visible overrides inherited visibility:hidden
            if (style.visibility === 'visible') {
                // Continue to check element's own conditions below (don't return ancestor marker)
                ancestorHidden = false;
            }
        }

        // Ancestor hidden (and not overridden) - propagate with 'ancestor' marker
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
    // ancestorHiddenReason: why ancestor is hidden (for visibility:visible override check)
    function walkTree(el, depth = 0, ancestorHidden = false, ancestorHiddenReason = null) {
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
        const visState = getVisualState(el, ancestorHidden, ancestorHiddenReason);

        // For visibility:hidden/collapse, must traverse children for potential overrides.
        // CSS allows visibility:visible children to override inherited visibility:hidden.
        const isVisibilityHiddenType =
            visState.marker === 'visibility-hidden' || visState.marker === 'visibility-collapse';

        // Skip visually hidden elements unless:
        // 1. includeHidden is true, OR
        // 2. Element has visibility:hidden/collapse (children may override)
        if (!visState.visible && !includeHidden && !isVisibilityHiddenType) {
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

        // Add checked state for checkboxes/radios/switches
        // Roles that use aria-checked attribute
        const CHECKED_ROLES = ['checkbox', 'radio', 'switch', 'menuitemcheckbox', 'menuitemradio'];
        // Roles that support mixed/indeterminate state (per WAI-ARIA spec)
        const MIXED_ALLOWED = ['checkbox', 'menuitemcheckbox'];

        if (CHECKED_ROLES.includes(role)) {
            // For native HTML inputs, IGNORE aria-checked per ARIA in HTML spec
            // Native semantics take precedence over ARIA attributes
            if (el.tagName === 'INPUT' && ['checkbox', 'radio'].includes(el.type)) {
                // Check indeterminate first (only checkboxes support this)
                if (el.type === 'checkbox' && el.indeterminate) {
                    node.checked = 'mixed';
                } else {
                    node.checked = el.checked;
                }
            }
            // For ARIA checkboxes (divs with role="checkbox", etc.)
            else if (el.hasAttribute('aria-checked')) {
                const val = el.getAttribute('aria-checked');
                if (val === 'mixed') {
                    // Mixed only valid for checkbox/menuitemcheckbox per spec
                    // Treat as false for other roles
                    node.checked = MIXED_ALLOWED.includes(role) ? 'mixed' : false;
                } else {
                    node.checked = val === 'true';
                }
            }
            // Role present but missing required aria-checked = assume unchecked
            else {
                node.checked = false;
            }
        }

        // Add selected state for selectable items (tabs, listbox options, tree items, etc.)
        // These use aria-selected instead of aria-checked
        const SELECTED_ROLES = ['option', 'tab', 'treeitem', 'gridcell', 'row', 'columnheader', 'rowheader'];
        if (SELECTED_ROLES.includes(role)) {
            if (el.hasAttribute('aria-selected')) {
                node.selected = el.getAttribute('aria-selected') === 'true';
            }
            // Note: Unlike aria-checked, aria-selected is not required
            // Absence means "not selectable" rather than "not selected"
        }

        // Add pressed state for toggle buttons
        // aria-pressed supports true/false/mixed (for partially pressed button groups)
        if (role === 'button' && el.hasAttribute('aria-pressed')) {
            const val = el.getAttribute('aria-pressed');
            if (val === 'mixed') {
                node.pressed = 'mixed';
            } else {
                node.pressed = val === 'true';
            }
        }

        // Add expanded state for disclosure widgets, accordions, menus, etc.
        if (el.hasAttribute('aria-expanded')) {
            node.expanded = el.getAttribute('aria-expanded') === 'true';
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
        // Track the reason for hiding (needed for visibility:visible override check)
        // If this element is hidden, use its marker; otherwise propagate parent's reason
        const childAncestorReason = !visState.visible ? visState.marker : ancestorHiddenReason;

        // Process children
        const children = [];
        for (const child of el.childNodes) {
            const childNode = walkTree(child, depth + 1, childAncestorHidden, childAncestorReason);
            if (childNode) {
                children.push(childNode);
            }
        }

        if (children.length > 0) {
            node.children = children;
        }

        // For visibility:hidden without includeHidden, only include if we have visible children.
        // Return the children directly (skip the hidden parent wrapper).
        if (!visState.visible && !includeHidden && isVisibilityHiddenType) {
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

return getVisualSnapshot(arguments[0], arguments[1], arguments[2]);
