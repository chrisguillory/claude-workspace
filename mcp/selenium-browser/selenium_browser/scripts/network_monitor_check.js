// noinspection JSAnnotator

/**
 * Check network monitor status.
 * Must be called after network_monitor_setup.js has been executed.
 *
 * Returns:
 *   Object with activeRequests, lastRequestTime, and currentTime
 */
return {
    activeRequests: window.__networkMonitor?.activeRequests || 0,
    lastRequestTime: window.__networkMonitor?.lastRequestTime,
    currentTime: Date.now()
};
