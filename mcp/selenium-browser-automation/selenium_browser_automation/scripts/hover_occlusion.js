/**
 * Occlusion check via elementFromPoint at element center.
 *
 * Computes center from getBoundingClientRect() (viewport-relative,
 * which is what elementFromPoint requires). Self-contained: callers
 * pass only the element.
 *
 * arguments[0]: target element
 */
const target = arguments[0];
const rect = target.getBoundingClientRect();
const x = rect.left + rect.width / 2;
const y = rect.top + rect.height / 2;
const atPoint = document.elementFromPoint(x, y);
if (!atPoint) return 'no_element';
if (atPoint === target) return 'ok';
if (target.contains(atPoint)) return 'ok';
return 'obscured';