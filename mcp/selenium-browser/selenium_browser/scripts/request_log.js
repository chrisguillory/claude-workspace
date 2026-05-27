/**
 * Request log for download_resource header replay.
 * Intercepts fetch() and XMLHttpRequest to record outgoing request headers
 * (and method, URL, timestamp) into a ring buffer at window.__downloadRequestLog.
 *
 * download_resource queries this buffer for the most recent request matching
 * the target URL's origin, takes its headers verbatim, and uses them as the
 * base for the httpx download. This propagates SPA HttpInterceptor-injected
 * headers (tenant scoping, CSRF tokens, market/feature flags, traceparent,
 * etc.) — anything the page's own HTTP client would attach.
 *
 * Configuration:
 *   - MAX_ENTRIES: 50 (FIFO eviction)
 *
 * Limitations:
 *   - Service Workers: Intercept before this code runs (separate scope)
 *   - Web Workers: Separate global scope, not intercepted
 *   - Request bodies are NOT captured (only headers are needed for replay)
 */
(function() {
    'use strict';

    if (window.__downloadRequestLogInstalled) {
        return;
    }
    window.__downloadRequestLogInstalled = true;

    const MAX_ENTRIES = 50;

    window.__downloadRequestLog = window.__downloadRequestLog || [];

    function addEntry(entry) {
        if (window.__downloadRequestLog.length >= MAX_ENTRIES) {
            window.__downloadRequestLog.shift();
        }
        window.__downloadRequestLog.push(entry);
    }

    function normalizeUrl(url) {
        try {
            return new URL(url, window.location.href).href;
        } catch (e) {
            return String(url);
        }
    }

    // Lowercase header keys at the write boundary. HTTP/2 lowercases on the
    // wire anyway, and consistent casing prevents two same-semantic keys
    // (X-Tenant-Id vs x-tenant-id) from landing in the same captured dict on
    // pages that mix XHR and fetch+Headers APIs.
    function setHeader(map, name, value) {
        map[String(name).toLowerCase()] = value;
    }

    // =========================================================================
    // XMLHttpRequest Interception
    // =========================================================================
    const origXHROpen = XMLHttpRequest.prototype.open;
    const origXHRSend = XMLHttpRequest.prototype.send;
    const origXHRSetHeader = XMLHttpRequest.prototype.setRequestHeader;

    XMLHttpRequest.prototype.open = function(method, url) {
        this.__dlReqLog = {
            url: normalizeUrl(url),
            method: (method || 'GET').toUpperCase(),
            headers: {},
            timestamp: null,
        };
        return origXHROpen.apply(this, arguments);
    };

    XMLHttpRequest.prototype.setRequestHeader = function(name, value) {
        if (this.__dlReqLog) {
            setHeader(this.__dlReqLog.headers, name, value);
        }
        return origXHRSetHeader.apply(this, arguments);
    };

    XMLHttpRequest.prototype.send = function() {
        if (this.__dlReqLog) {
            this.__dlReqLog.timestamp = Date.now();
            addEntry(this.__dlReqLog);
        }
        return origXHRSend.apply(this, arguments);
    };

    // =========================================================================
    // Fetch API Interception
    // =========================================================================
    const origFetch = window.fetch;

    window.fetch = function(input, init) {
        let url, method;
        const headers = {};
        const initObj = init || {};

        if (input instanceof Request) {
            url = input.url;
            method = (input.method || 'GET').toUpperCase();
            input.headers.forEach(function(v, k) { setHeader(headers, k, v); });
        } else {
            url = String(input);
            method = (initObj.method || 'GET').toUpperCase();
        }

        if (initObj.headers) {
            if (initObj.headers instanceof Headers) {
                initObj.headers.forEach(function(v, k) { setHeader(headers, k, v); });
            } else if (Array.isArray(initObj.headers)) {
                for (const pair of initObj.headers) {
                    if (pair && pair.length >= 2) {
                        setHeader(headers, pair[0], pair[1]);
                    }
                }
            } else {
                for (const k of Object.keys(initObj.headers)) {
                    setHeader(headers, k, initObj.headers[k]);
                }
            }
        }

        addEntry({
            url: normalizeUrl(url),
            method: method,
            headers: headers,
            timestamp: Date.now(),
        });

        return origFetch.apply(this, arguments);
    };

})();
