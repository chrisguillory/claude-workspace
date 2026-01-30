/**
 * Response body capture for HAR export.
 * Intercepts fetch() and XMLHttpRequest to capture response bodies in real-time,
 * before Chrome garbage collects them.
 *
 * Creates window.__responseBodies array with captured responses.
 * Each entry contains: url, method, status, contentType, requestBody, responseBody,
 * timestamp, base64Encoded, truncated.
 *
 * Configuration:
 *   - MAX_ENTRIES: 1000 (FIFO eviction)
 *   - MAX_BODY_SIZE: 10MB per response (matches export_har default)
 *
 * Limitations:
 *   - Service Workers: Intercept before this code runs (documented limitation)
 *   - Web Workers: Separate global scope, not intercepted (documented limitation)
 *   - Streaming responses: Only captures up to MAX_BODY_SIZE
 */
(function() {
    'use strict';

    // Prevent double-installation
    if (window.__responseBodyCaptureInstalled) {
        return;
    }
    window.__responseBodyCaptureInstalled = true;

    const MAX_ENTRIES = 1000;
    const MAX_BODY_SIZE = 10 * 1024 * 1024; // 10MB - matches export_har default

    // Initialize storage
    window.__responseBodies = window.__responseBodies || [];

    /**
     * Add entry with FIFO eviction at MAX_ENTRIES.
     */
    function addEntry(entry) {
        if (window.__responseBodies.length >= MAX_ENTRIES) {
            window.__responseBodies.shift();
        }
        window.__responseBodies.push(entry);
    }

    /**
     * Normalize URL to absolute form for consistent matching with CDP.
     * Handles relative URLs and encoding differences.
     */
    function normalizeUrl(url) {
        try {
            return new URL(url, window.location.href).href;
        } catch (e) {
            // If URL parsing fails, return as-is
            return String(url);
        }
    }

    /**
     * Convert ArrayBuffer to base64 string.
     */
    function arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        const chunkSize = 32768; // Process in chunks to avoid call stack limits
        for (let i = 0; i < bytes.length; i += chunkSize) {
            const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
            for (let j = 0; j < chunk.length; j++) {
                binary += String.fromCharCode(chunk[j]);
            }
        }
        return btoa(binary);
    }

    /**
     * Serialize request body for capture.
     * Handles: string, ArrayBuffer, FormData, URLSearchParams, Blob.
     */
    async function serializeRequestBody(body) {
        if (!body) return null;

        if (typeof body === 'string') {
            if (body.length > MAX_BODY_SIZE) {
                return { data: body.substring(0, MAX_BODY_SIZE), type: 'string', truncated: true };
            }
            return { data: body, type: 'string' };
        }

        if (body instanceof ArrayBuffer) {
            if (body.byteLength > MAX_BODY_SIZE) {
                return { data: arrayBufferToBase64(body.slice(0, MAX_BODY_SIZE)), type: 'base64', truncated: true };
            }
            return { data: arrayBufferToBase64(body), type: 'base64' };
        }

        if (body instanceof FormData) {
            const obj = {};
            for (const [key, value] of body.entries()) {
                if (value instanceof File) {
                    obj[key] = '[File: ' + value.name + ', ' + value.size + ' bytes, ' + value.type + ']';
                } else {
                    obj[key] = value;
                }
            }
            return { data: JSON.stringify(obj), type: 'formdata' };
        }

        if (body instanceof URLSearchParams) {
            return { data: body.toString(), type: 'urlsearchparams' };
        }

        if (body instanceof Blob) {
            try {
                const buffer = await body.arrayBuffer();
                if (buffer.byteLength > MAX_BODY_SIZE) {
                    return { data: arrayBufferToBase64(buffer.slice(0, MAX_BODY_SIZE)), type: 'base64', truncated: true, size: buffer.byteLength };
                }
                return { data: arrayBufferToBase64(buffer), type: 'base64' };
            } catch (e) {
                return { data: null, type: 'blob', error: e.message };
            }
        }

        // Unknown type - try to stringify
        try {
            const str = JSON.stringify(body);
            if (str.length > MAX_BODY_SIZE) {
                return { data: str.substring(0, MAX_BODY_SIZE), type: 'json', truncated: true };
            }
            return { data: str, type: 'json' };
        } catch (e) {
            return { data: String(body), type: 'unknown' };
        }
    }

    /**
     * Capture response body from a cloned Response object.
     * Handles text, JSON, binary responses with truncation.
     */
    async function captureResponseBody(response, contentType) {
        try {
            const clone = response.clone();

            // Determine if binary based on content-type
            const isBinary = contentType && (
                contentType.includes('image/') ||
                contentType.includes('audio/') ||
                contentType.includes('video/') ||
                contentType.includes('application/octet-stream') ||
                contentType.includes('application/pdf') ||
                contentType.includes('application/zip') ||
                contentType.includes('application/gzip')
            );

            if (isBinary) {
                const buffer = await clone.arrayBuffer();
                if (buffer.byteLength > MAX_BODY_SIZE) {
                    return {
                        body: arrayBufferToBase64(buffer.slice(0, MAX_BODY_SIZE)),
                        base64Encoded: true,
                        truncated: true,
                        originalSize: buffer.byteLength
                    };
                }
                return {
                    body: arrayBufferToBase64(buffer),
                    base64Encoded: true,
                    truncated: false
                };
            }

            // Text-based response
            const text = await clone.text();
            if (text.length > MAX_BODY_SIZE) {
                return {
                    body: text.substring(0, MAX_BODY_SIZE),
                    base64Encoded: false,
                    truncated: true,
                    originalSize: text.length
                };
            }
            return {
                body: text,
                base64Encoded: false,
                truncated: false
            };
        } catch (e) {
            return {
                body: null,
                base64Encoded: false,
                truncated: false,
                error: e.message
            };
        }
    }

    // =========================================================================
    // Fetch API Interception
    // =========================================================================
    const origFetch = window.fetch;

    window.fetch = async function(...args) {
        const startTime = Date.now();

        // Parse request info
        let url, method, requestBody;
        const input = args[0];
        const init = args[1] || {};

        if (input instanceof Request) {
            url = input.url;
            method = input.method;
            // Clone request to read body without consuming it
            try {
                const clonedReq = input.clone();
                requestBody = await clonedReq.text();
            } catch (e) {
                requestBody = null;
            }
        } else {
            url = String(input);
            method = init.method || 'GET';
            requestBody = init.body || null;
        }

        // Normalize URL to absolute form for consistent CDP matching
        const normalizedUrl = normalizeUrl(url);

        // Prepare entry
        const entry = {
            api: 'fetch',
            url: normalizedUrl,
            method: method.toUpperCase(),
            timestamp: startTime,
            requestBody: null,
            status: null,
            contentType: null,
            responseBody: null,
            base64Encoded: false,
            truncated: false,
            error: null
        };

        // Serialize request body
        if (requestBody) {
            try {
                entry.requestBody = await serializeRequestBody(requestBody);
            } catch (e) {
                entry.requestBody = { error: e.message };
            }
        }

        try {
            // Make the actual request
            const response = await origFetch.apply(this, args);

            // Capture response metadata
            entry.status = response.status;
            entry.contentType = response.headers.get('content-type') || '';

            // Capture response body
            const bodyResult = await captureResponseBody(response, entry.contentType);
            entry.responseBody = bodyResult.body;
            entry.base64Encoded = bodyResult.base64Encoded;
            entry.truncated = bodyResult.truncated;
            if (bodyResult.originalSize) {
                entry.originalSize = bodyResult.originalSize;
            }
            if (bodyResult.error) {
                entry.error = bodyResult.error;
            }

            addEntry(entry);
            return response;

        } catch (e) {
            entry.error = e.message || String(e);
            addEntry(entry);
            throw e;
        }
    };

    // =========================================================================
    // XMLHttpRequest Interception
    // =========================================================================
    const origXHROpen = XMLHttpRequest.prototype.open;
    const origXHRSend = XMLHttpRequest.prototype.send;

    XMLHttpRequest.prototype.open = function(method, url) {
        // Normalize URL to absolute form for consistent CDP matching
        const normalizedUrl = normalizeUrl(url);

        this.__capture = {
            api: 'xhr',
            url: normalizedUrl,
            method: (method || 'GET').toUpperCase(),
            timestamp: null,
            requestBody: null,
            status: null,
            contentType: null,
            responseBody: null,
            base64Encoded: false,
            truncated: false,
            error: null
        };
        return origXHROpen.apply(this, arguments);
    };

    XMLHttpRequest.prototype.send = function(body) {
        const xhr = this;

        if (xhr.__capture) {
            xhr.__capture.timestamp = Date.now();

            // Capture request body
            if (body) {
                if (typeof body === 'string') {
                    if (body.length > MAX_BODY_SIZE) {
                        xhr.__capture.requestBody = { data: body.substring(0, MAX_BODY_SIZE), type: 'string', truncated: true };
                    } else {
                        xhr.__capture.requestBody = { data: body, type: 'string' };
                    }
                } else if (body instanceof FormData) {
                    const obj = {};
                    for (const [key, value] of body.entries()) {
                        if (value instanceof File) {
                            obj[key] = '[File: ' + value.name + ', ' + value.size + ' bytes, ' + value.type + ']';
                        } else {
                            obj[key] = value;
                        }
                    }
                    xhr.__capture.requestBody = { data: JSON.stringify(obj), type: 'formdata' };
                } else if (body instanceof ArrayBuffer || body instanceof Uint8Array) {
                    const buffer = body instanceof Uint8Array ? body.buffer : body;
                    if (buffer.byteLength > MAX_BODY_SIZE) {
                        xhr.__capture.requestBody = { data: arrayBufferToBase64(buffer.slice(0, MAX_BODY_SIZE)), type: 'base64', truncated: true };
                    } else {
                        xhr.__capture.requestBody = { data: arrayBufferToBase64(buffer), type: 'base64' };
                    }
                } else if (body instanceof Blob) {
                    xhr.__capture.requestBody = { data: null, type: 'blob', note: 'Blob sent via XHR - async read not performed' };
                } else {
                    try {
                        const str = JSON.stringify(body);
                        if (str.length > MAX_BODY_SIZE) {
                            xhr.__capture.requestBody = { data: str.substring(0, MAX_BODY_SIZE), type: 'json', truncated: true };
                        } else {
                            xhr.__capture.requestBody = { data: str, type: 'json' };
                        }
                    } catch (e) {
                        xhr.__capture.requestBody = { data: String(body), type: 'unknown' };
                    }
                }
            }

            // error/abort/timeout handlers only set the error message.
            // loadend is the ONLY place we call addEntry() to avoid duplicates,
            // since loadend fires AFTER error/abort/timeout per WHATWG XHR spec.
            xhr.addEventListener('error', function() {
                xhr.__capture.error = 'XHR network error';
            });

            xhr.addEventListener('abort', function() {
                xhr.__capture.error = 'XHR aborted';
            });

            xhr.addEventListener('timeout', function() {
                xhr.__capture.error = 'XHR timeout';
            });

            // loadend fires after ALL terminal events (load, error, abort, timeout).
            // This is the single place we capture and add the entry.
            xhr.addEventListener('loadend', function() {
                try {
                    xhr.__capture.status = xhr.status;
                    xhr.__capture.contentType = xhr.getResponseHeader('content-type') || '';

                    const responseType = xhr.responseType || 'text';

                    // Only capture response body if no error was set by error/abort/timeout handlers
                    if (!xhr.__capture.error) {
                        if (responseType === '' || responseType === 'text') {
                            const text = xhr.responseText || '';
                            if (text.length > MAX_BODY_SIZE) {
                                xhr.__capture.responseBody = text.substring(0, MAX_BODY_SIZE);
                                xhr.__capture.truncated = true;
                                xhr.__capture.originalSize = text.length;
                            } else {
                                xhr.__capture.responseBody = text;
                            }
                        } else if (responseType === 'json') {
                            try {
                                const str = JSON.stringify(xhr.response);
                                if (str.length > MAX_BODY_SIZE) {
                                    xhr.__capture.responseBody = str.substring(0, MAX_BODY_SIZE);
                                    xhr.__capture.truncated = true;
                                } else {
                                    xhr.__capture.responseBody = str;
                                }
                            } catch (e) {
                                xhr.__capture.error = 'JSON stringify failed: ' + e.message;
                            }
                        } else if (responseType === 'arraybuffer' && xhr.response) {
                            const buffer = xhr.response;
                            if (buffer.byteLength > MAX_BODY_SIZE) {
                                xhr.__capture.responseBody = arrayBufferToBase64(buffer.slice(0, MAX_BODY_SIZE));
                                xhr.__capture.truncated = true;
                                xhr.__capture.originalSize = buffer.byteLength;
                            } else {
                                xhr.__capture.responseBody = arrayBufferToBase64(buffer);
                            }
                            xhr.__capture.base64Encoded = true;
                        } else if (responseType === 'blob') {
                            xhr.__capture.error = 'Blob responseType not captured (async read required)';
                        } else if (responseType === 'document') {
                            try {
                                xhr.__capture.responseBody = xhr.response ? xhr.response.documentElement.outerHTML : null;
                            } catch (e) {
                                xhr.__capture.error = 'Document serialization failed: ' + e.message;
                            }
                        }
                    }

                    addEntry(xhr.__capture);
                } catch (e) {
                    xhr.__capture.error = 'Capture failed: ' + e.message;
                    addEntry(xhr.__capture);
                }
            });
        }

        return origXHRSend.apply(this, arguments);
    };

})();