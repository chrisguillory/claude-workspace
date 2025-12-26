/**
 * Extract resource timing entries from Performance API.
 *
 * Returns:
 *   Array of request objects with URL, timing breakdown, and size
 */
var entries = performance.getEntriesByType('resource');
return entries.map(function(r) {
    return {
        url: r.name,
        method: 'GET',
        resource_type: r.initiatorType,
        started_at: r.startTime,
        duration_ms: r.duration,
        encoded_data_length: r.transferSize || 0,
        timing: {
            blocked: Math.max(0, r.fetchStart - r.startTime),
            dns: Math.max(0, r.domainLookupEnd - r.domainLookupStart),
            connect: Math.max(0, r.connectEnd - r.connectStart),
            ssl: r.secureConnectionStart > 0 ? Math.max(0, r.connectEnd - r.secureConnectionStart) : 0,
            send: Math.max(0, r.requestStart - r.connectEnd),
            wait: Math.max(0, r.responseStart - r.requestStart),
            receive: Math.max(0, r.responseEnd - r.responseStart)
        }
    };
});
