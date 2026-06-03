/**
 * Smooth scroll helper that resolves only after the animation has settled.
 *
 * Correctness rests on position stability, not on any single browser event:
 * a rAF loop polls the scroll position every frame and resolves once the
 * position has held steady across STABLE_FRAMES consecutive frames (the
 * animation has stopped). scrollend and the timeout are accelerators/safety
 * nets layered on top — never the sole signal.
 *
 *   1. Stability poll (primary) — resolve after the position is unchanged for
 *      STABLE_FRAMES frames *and* it has moved at least once. This guarantees
 *      measure() runs after the CSS smooth animation finishes, never mid-flight.
 *   2. No-op grace (NOOP_GRACE_FRAMES) — if the position never moves within the
 *      grace window, the scroll was a genuine no-op (already at the boundary,
 *      where scrollend never fires); resolve immediately. The window is wide
 *      enough that a real animation whose first paint is merely delayed is not
 *      misread as a no-op — the bug the old double-rAF early-exit caused.
 *   3. scrollend (accelerator) — when it fires, settle on the next frame instead
 *      of waiting out the remaining stability frames.
 *   4. Timeout (TIMEOUT_MS) — hard safety net for pathologically long scrolls.
 *
 * @param {Object} opts
 * @param {EventTarget} opts.eventTarget - Where to listen for scrollend (window or element)
 * @param {function} opts.scrollAction - Initiates the smooth scroll
 * @param {function} opts.hasMoved - Returns true if scroll position changed from before
 * @param {function} opts.position - Returns a scalar scroll position to track for stability
 * @param {function} opts.measure - Returns the result object with final position data
 * @returns {Promise<Object>} Result from measure(), called after scroll settles
 */
function smoothScroll(opts) {
    var STABLE_FRAMES = 3;       // consecutive unchanged frames => animation settled
    var NOOP_GRACE_FRAMES = 10;  // frames to allow the animation to start before declaring no-op
    var TIMEOUT_MS = 3000;       // hard fallback (Chrome's 7500+ px smooth scrolls can exceed 1s)
    return new Promise(function(resolve) {
        var resolved = false, frames = 0, stable = 0, moved = false, prev = null;
        function finish() {
            if (resolved) return;
            resolved = true;
            opts.eventTarget.removeEventListener('scrollend', onScrollend);
            resolve(opts.measure());
        }
        // Accelerator: settle on the frame after scrollend rather than waiting
        // out the remaining stability frames. One extra frame lets the final
        // position commit before measure().
        function onScrollend() {
            requestAnimationFrame(finish);
        }
        opts.eventTarget.addEventListener('scrollend', onScrollend, {once: true});
        opts.scrollAction();
        var start = performance.now();
        function tick() {
            if (resolved) return;
            frames++;
            if (opts.hasMoved()) moved = true;
            var p = opts.position();
            stable = (prev !== null && p === prev) ? stable + 1 : 0;
            prev = p;
            // Settled: it moved and has now held steady for STABLE_FRAMES frames.
            if (moved && stable >= STABLE_FRAMES) { finish(); return; }
            // Genuine no-op: never moved within the grace window (scrollend won't fire).
            if (!moved && frames >= NOOP_GRACE_FRAMES) { finish(); return; }
            // Safety net for pathologically long animations.
            if (performance.now() - start > TIMEOUT_MS) { finish(); return; }
            requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
}
