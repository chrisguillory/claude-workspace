/**
 * Core Web Vitals collection using Performance APIs.
 * Collects FCP, LCP, TTFB, CLS, and INP metrics with ratings.
 *
 * Arguments:
 *   arguments[0]: timeoutMs - max wait time for metric collection
 *   arguments[arguments.length - 1]: callback - Selenium async callback
 *
 * Returns via callback:
 *   Object with fcp, lcp, ttfb, cls, inp metrics (or null if unavailable)
 */
var callback = arguments[arguments.length - 1];
var timeoutMs = arguments[0];

(async function collectWebVitals() {
    var results = { fcp: null, lcp: null, ttfb: null, cls: null, inp: null };

    // FCP - immediate from paint entries
    try {
        var fcpEntry = performance.getEntriesByName('first-contentful-paint', 'paint')[0];
        if (fcpEntry) {
            results.fcp = {
                name: 'FCP',
                value: fcpEntry.startTime,
                rating: fcpEntry.startTime <= 1800 ? 'good' : fcpEntry.startTime <= 3000 ? 'needs-improvement' : 'poor'
            };
        }
    } catch (e) { results.fcp = { error: e.toString() }; }

    // TTFB - immediate from navigation timing
    try {
        var navEntries = performance.getEntriesByType('navigation');
        var navEntry = navEntries[0];
        if (navEntry) {
            results.ttfb = {
                name: 'TTFB',
                value: navEntry.responseStart,
                rating: navEntry.responseStart <= 800 ? 'good' : navEntry.responseStart <= 1800 ? 'needs-improvement' : 'poor',
                phases: {
                    dns: navEntry.domainLookupEnd - navEntry.domainLookupStart,
                    tcp: navEntry.connectEnd - navEntry.connectStart,
                    request: navEntry.responseStart - navEntry.requestStart
                }
            };
        }
    } catch (e) { results.ttfb = { error: e.toString() }; }

    // LCP - use PerformanceObserver with buffered flag
    try {
        results.lcp = await new Promise(function(resolve) {
            var lastEntry = null;
            var observer = new PerformanceObserver(function(list) {
                var entries = list.getEntries();
                lastEntry = entries[entries.length - 1];
            });
            observer.observe({ type: 'largest-contentful-paint', buffered: true });
            setTimeout(function() {
                observer.disconnect();
                if (lastEntry) {
                    resolve({
                        name: 'LCP',
                        value: lastEntry.startTime,
                        size: lastEntry.size,
                        element_id: lastEntry.id || null,
                        url: lastEntry.url || null,
                        rating: lastEntry.startTime <= 2500 ? 'good' : lastEntry.startTime <= 4000 ? 'needs-improvement' : 'poor'
                    });
                } else { resolve(null); }
            }, Math.min(timeoutMs, 3000));
        });
    } catch (e) { results.lcp = { error: e.toString() }; }

    // CLS - collect layout shifts
    try {
        results.cls = await new Promise(function(resolve) {
            var sessionValue = 0;
            var sessionEntries = [];
            var observer = new PerformanceObserver(function(list) {
                var entries = list.getEntries();
                for (var i = 0; i < entries.length; i++) {
                    var entry = entries[i];
                    if (!entry.hadRecentInput) {
                        sessionValue += entry.value;
                        var sources = [];
                        if (entry.sources) {
                            for (var j = 0; j < entry.sources.length; j++) {
                                var s = entry.sources[j];
                                sources.push({
                                    node: s.node ? s.node.tagName : null
                                });
                            }
                        }
                        sessionEntries.push({
                            value: entry.value,
                            time: entry.startTime,
                            sources: sources
                        });
                    }
                }
            });
            observer.observe({ type: 'layout-shift', buffered: true });
            setTimeout(function() {
                observer.disconnect();
                resolve({
                    name: 'CLS',
                    value: sessionValue,
                    rating: sessionValue <= 0.1 ? 'good' : sessionValue <= 0.25 ? 'needs-improvement' : 'poor',
                    entries: sessionEntries
                });
            }, Math.min(timeoutMs, 2000));
        });
    } catch (e) { results.cls = { error: e.toString() }; }

    // INP - collect event timing (requires user interaction)
    try {
        results.inp = await new Promise(function(resolve) {
            var worstInteraction = null;
            var observer = new PerformanceObserver(function(list) {
                var entries = list.getEntries();
                for (var i = 0; i < entries.length; i++) {
                    var entry = entries[i];
                    if (!worstInteraction || entry.duration > worstInteraction.duration) {
                        worstInteraction = {
                            duration: entry.duration,
                            name: entry.name,
                            start_time: entry.startTime,
                            input_delay: entry.processingStart - entry.startTime,
                            processing_time: entry.processingEnd - entry.processingStart,
                            presentation_delay: entry.duration - (entry.processingEnd - entry.startTime)
                        };
                    }
                }
            });
            observer.observe({ type: 'event', durationThreshold: 40, buffered: true });
            setTimeout(function() {
                observer.disconnect();
                if (worstInteraction) {
                    resolve({
                        name: 'INP',
                        value: worstInteraction.duration,
                        rating: worstInteraction.duration <= 200 ? 'good' : worstInteraction.duration <= 500 ? 'needs-improvement' : 'poor',
                        details: worstInteraction
                    });
                } else { resolve(null); }
            }, Math.min(timeoutMs, 1000));
        });
    } catch (e) { results.inp = { error: e.toString() }; }

    return results;
})().then(callback).catch(function(err) { callback({ error: err.toString() }); });
