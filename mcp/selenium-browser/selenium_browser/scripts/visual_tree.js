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

    // Page metadata counters (always collected, negligible overhead)
    const stats = {
        hidden: { total: 0, displayNone: 0, visibilityHidden: 0, opacity: 0, clipped: 0, offscreen: 0, other: 0 },
        images: { total: 0, withAlt: 0, withoutAlt: 0 },
        links: 0,
        totalNodes: 0,
        depthTruncated: 0
    };

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

    // Accessible name computation per WAI-ARIA AccName 1.2 spec (same as ARIA tree)
    function computeAccessibleName(el) {
        // Step 2B: aria-labelledby (before aria-label per spec)
        if (el.getAttribute('aria-labelledby')) {
            const ids = el.getAttribute('aria-labelledby').split(/\s+/);
            const name = ids
                .map(id => {
                    const refEl = document.getElementById(id);
                    return refEl ? normalize(refEl.textContent) : '';
                })
                .filter(Boolean)
                .join(' ');
            if (name) return name;
        }
        // Step 2D: aria-label
        if (el.getAttribute('aria-label')) {
            return normalize(el.getAttribute('aria-label'));
        }
        const tagName = el.tagName.toLowerCase();
        // Step 2E: Host language label — form controls via native API
        if (el.labels && el.labels.length) {
            return Array.from(el.labels)
                .map(label => normalize(label.textContent))
                .filter(Boolean)
                .join(' ');
        }
        // Fieldset: first legend child
        if (tagName === 'fieldset') {
            for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
                if (child.tagName.toLowerCase() === 'legend') {
                    return normalize(child.textContent);
                }
            }
        }
        // Figure: first figcaption child
        if (tagName === 'figure') {
            for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
                if (child.tagName.toLowerCase() === 'figcaption') {
                    return normalize(child.textContent);
                }
            }
        }
        // Table: first caption child
        if (tagName === 'table') {
            for (let child = el.firstElementChild; child; child = child.nextElementSibling) {
                if (child.tagName.toLowerCase() === 'caption') {
                    return normalize(child.textContent);
                }
            }
        }
        // Image: alt attribute
        if (tagName === 'img') {
            const alt = el.getAttribute('alt');
            if (alt != null) return normalize(alt);
        }
        // Name from content: buttons, links, headings
        if (['button', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
            return normalize(el.textContent);
        }
        // Input value/placeholder
        if (['input', 'textarea'].includes(tagName)) {
            return normalize(el.placeholder || el.value || '');
        }
        // Step 2I: Tooltip (title attribute) as last resort
        if (el.getAttribute('title')) {
            return normalize(el.getAttribute('title'));
        }
        return '';
    }

    // Implicit role mapping
    function getImplicitRole(el) {
        const tagName = el.tagName.toLowerCase();

        // Input types have complex mapping per HTML-AAM
        if (tagName === 'input') {
            const type = (el.getAttribute('type') || 'text').toLowerCase();
            switch (type) {
                case 'checkbox': return 'checkbox';
                case 'radio': return 'radio';
                case 'button': case 'submit': case 'reset': case 'image': return 'button';
                case 'number': return 'spinbutton';
                case 'range': return 'slider';
                case 'search': return 'searchbox';
                default: return 'textbox';
            }
        }

        // Context-sensitive: header/footer are landmarks only at page level
        if (tagName === 'header' || tagName === 'footer') {
            const sectioning = ['article', 'aside', 'main', 'nav', 'section'];
            let parent = el.parentElement;
            while (parent) {
                if (sectioning.includes(parent.tagName.toLowerCase())) return 'generic';
                parent = parent.parentElement;
            }
            return tagName === 'header' ? 'banner' : 'contentinfo';
        }

        // Context-sensitive: section is region only with an accessible name
        if (tagName === 'section') {
            return (el.getAttribute('aria-label') || el.getAttribute('aria-labelledby'))
                ? 'region' : 'generic';
        }

        // Context-sensitive: anchor is link only with href
        if (tagName === 'a') return el.hasAttribute('href') ? 'link' : 'generic';

        // Context-sensitive: summary is button only as direct child of details
        if (tagName === 'summary') {
            return el.parentElement?.tagName.toLowerCase() === 'details' ? 'button' : 'generic';
        }

        // Context-sensitive: select is listbox when multiple or size > 1
        if (tagName === 'select') {
            return (el.hasAttribute('multiple') || parseInt(el.getAttribute('size'), 10) > 1)
                ? 'listbox' : 'combobox';
        }

        // Static implicit role map
        const implicitRoles = {
            'button': 'button',
            'h1': 'heading', 'h2': 'heading', 'h3': 'heading',
            'h4': 'heading', 'h5': 'heading', 'h6': 'heading',
            'nav': 'navigation',
            'main': 'main',
            'article': 'article',
            'aside': 'complementary',
            'form': 'form',
            'p': 'paragraph',
            'textarea': 'textbox',
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
            'code': 'code',
            'fieldset': 'group',
            'details': 'group',
            'dt': 'term',
            'dd': 'definition',
            'figure': 'figure',
            'meter': 'meter',
            'progress': 'progressbar',
            'output': 'status',
            'hr': 'separator',
            'option': 'option',
            'optgroup': 'group',
            'address': 'group',
        };

        return implicitRoles[tagName] || 'generic';
    }

    // Walk tree and build hierarchical snapshot
    // ancestorHiddenReason: why ancestor is hidden (for visibility:visible override check)
    function walkTree(el, depth = 0, ancestorHidden = false, ancestorHiddenReason = null) {
        if (depth > 50) { stats.depthTruncated++; return null; }

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
            stats.hidden.total++;
            switch (visState.marker) {
                case 'display-none': stats.hidden.displayNone++; break;
                case 'visibility-hidden': case 'visibility-collapse':
                    stats.hidden.visibilityHidden++; break;
                case 'opacity': stats.hidden.opacity++; break;
                case 'clipped': stats.hidden.clipped++; break;
                case 'offscreen': stats.hidden.offscreen++; break;
                default: stats.hidden.other++; break;
            }
            return null;
        }

        const role = el.getAttribute('role') || getImplicitRole(el);
        const name = computeAccessibleName(el);

        // Count visible elements for page metadata
        stats.totalNodes++;
        if (el.tagName === 'IMG') {
            stats.images.total++;
            if (el.getAttribute('alt')) stats.images.withAlt++;
            else stats.images.withoutAlt++;
        }
        if (role === 'link') stats.links++;

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
        } else if (el.tagName.toLowerCase() === 'details') {
            node.expanded = el.open;
        } else if (el.tagName.toLowerCase() === 'summary' && el.parentElement?.tagName.toLowerCase() === 'details') {
            node.expanded = el.parentElement.open;
        }

        // Add disabled state
        if (el.hasAttribute('aria-disabled')) {
            node.disabled = el.getAttribute('aria-disabled') === 'true';
        } else if (['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) {
            node.disabled = el.disabled;
        }

        // Add URL for links: always if no name (provides essential info),
        // or if includeUrls explicitly requested
        if (role === 'link' && el.href && (includeUrls || !name)) {
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
                stats.hidden.total++;
                stats.hidden.visibilityHidden++;
                return null;
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

    // Pre-walk: count iframes and shadow roots (may be inside hidden containers)
    const iframeCount = root.querySelectorAll('iframe').length;
    let shadowRootCount = 0;
    const shadowWalker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    while (shadowWalker.nextNode()) {
        if (shadowWalker.currentNode.shadowRoot) shadowRootCount++;
    }

    const tree = walkTree(root);
    return {
        tree,
        stats: {
            hidden: stats.hidden,
            iframes: iframeCount,
            shadowRoots: shadowRootCount,
            images: stats.images,
            links: stats.links,
            totalNodes: stats.totalNodes,
            depthTruncated: stats.depthTruncated
        }
    };
}

return getVisualSnapshot(arguments[0], arguments[1], arguments[2]);
