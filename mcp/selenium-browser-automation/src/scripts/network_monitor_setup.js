/**
 * Network activity monitor setup.
 * Instruments Fetch API and XMLHttpRequest to track active requests.
 *
 * Creates window.__networkMonitor with:
 *   - activeRequests: count of in-flight requests
 *   - lastRequestTime: timestamp of last completed request
 *
 * No arguments or return value - side effect only.
 */
window.__networkMonitor = {
    activeRequests: 0,
    lastRequestTime: null,

    increment() {
        this.activeRequests++;
        this.lastRequestTime = Date.now();
    },

    decrement() {
        this.activeRequests = Math.max(0, this.activeRequests - 1);
    }
};

// Instrument Fetch API
const origFetch = window.fetch;
window.fetch = function(...args) {
    window.__networkMonitor.increment();
    return origFetch.apply(this, args)
        .then(r => { window.__networkMonitor.decrement(); return r; })
        .catch(e => { window.__networkMonitor.decrement(); throw e; });
};

// Instrument XMLHttpRequest
const origOpen = XMLHttpRequest.prototype.open;
const origSend = XMLHttpRequest.prototype.send;

XMLHttpRequest.prototype.open = function() {
    window.__networkMonitor.increment();
    return origOpen.apply(this, arguments);
};

XMLHttpRequest.prototype.send = function() {
    this.addEventListener('loadend', () => window.__networkMonitor.decrement());
    return origSend.apply(this, arguments);
};
