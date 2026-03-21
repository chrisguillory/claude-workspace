/**
 * Multi-signal stability check for hover actionability.
 *
 * Uses requestAnimationFrame for position polling + Web Animations API for
 * CSS animation detection. Returns a Promise that ChromeDriver auto-resolves.
 *
 * arguments[0]: target element
 */
const el = arguments[0];
const DISTANCE_THRESHOLD = 5;  // Pixels - matches Cypress default
const MAX_CHECKS = 10;  // ~160ms at 60fps

return new Promise((resolve) => {
    // Check for running animations using Web Animations API
    let animations = [];
    let hasInfiniteAnimation = false;
    try {
        animations = el.getAnimations();
        hasInfiniteAnimation = animations.some(a => {
            const effect = a.effect;
            if (effect && effect.getTiming) {
                const timing = effect.getTiming();
                return timing.iterations === Infinity;
            }
            return false;
        });
    } catch (e) {
        // getAnimations not supported, proceed without
    }

    const runningAnimations = animations.filter(a => a.playState === 'running');

    // Two-frame stability check using requestAnimationFrame
    let prevRect = el.getBoundingClientRect();
    let checkCount = 0;
    let consecutiveStable = 0;

    function checkStability() {
        checkCount++;
        const currRect = el.getBoundingClientRect();

        // Calculate distance moved
        const dx = currRect.x - prevRect.x;
        const dy = currRect.y - prevRect.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Check size stability
        const sizeStable = (
            Math.abs(currRect.width - prevRect.width) < 1 &&
            Math.abs(currRect.height - prevRect.height) < 1
        );

        // Element is stable if moved less than threshold and size unchanged
        const isStable = distance < DISTANCE_THRESHOLD && sizeStable;

        if (isStable) {
            consecutiveStable++;
            // Require 2 consecutive stable frames (like Playwright)
            if (consecutiveStable >= 2) {
                resolve({
                    stable: true,
                    framesChecked: checkCount,
                    runningAnimations: runningAnimations.length,
                    hasInfiniteAnimation: hasInfiniteAnimation,
                    finalDistance: distance
                });
                return;
            }
        } else {
            consecutiveStable = 0;
        }

        prevRect = currRect;

        // Max checks reached - report as unstable or proceed with warning
        if (checkCount >= MAX_CHECKS) {
            resolve({
                stable: false,
                framesChecked: checkCount,
                runningAnimations: runningAnimations.length,
                hasInfiniteAnimation: hasInfiniteAnimation,
                finalDistance: distance,
                reason: 'timeout'
            });
            return;
        }

        requestAnimationFrame(checkStability);
    }

    // Start checking on next frame
    requestAnimationFrame(checkStability);
});
