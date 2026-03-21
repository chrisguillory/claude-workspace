/**
 * Smooth scroll helper with scrollend + rAF + timeout waiting.
 *
 * Waits for a smooth scroll animation to complete using three signals:
 * 1. scrollend event (primary — fired by browser after animation completes)
 * 2. Double-rAF no-movement check (early exit if already at target position, ~32ms)
 * 3. setTimeout fallback (safety net if scrollend never fires, 1000ms)
 *
 * @param {Object} opts
 * @param {EventTarget} opts.eventTarget - Where to listen for scrollend (window or element)
 * @param {function} opts.scrollAction - Initiates the smooth scroll
 * @param {function} opts.hasMoved - Returns true if scroll position changed from before
 * @param {function} opts.measure - Returns the result object with final position data
 * @returns {Promise<Object>} Result from measure(), called after scroll completes
 */
function smoothScroll(opts) {
    var resolved = false;
    return new Promise(function(resolve) {
        function finish() {
            if (resolved) return;
            resolved = true;
            resolve(opts.measure());
        }
        opts.eventTarget.addEventListener('scrollend', finish, {once: true});
        opts.scrollAction();
        // Early exit: if no scroll occurred (already at boundary), scrollend won't fire.
        // Double-rAF detects this in ~32ms instead of waiting for the 1s timeout.
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                if (!opts.hasMoved()) {
                    opts.eventTarget.removeEventListener('scrollend', finish);
                    finish();
                }
            });
        });
        // Hard fallback in case scrollend never fires
        setTimeout(function() {
            opts.eventTarget.removeEventListener('scrollend', finish);
            finish();
        }, 1000);
    });
}
