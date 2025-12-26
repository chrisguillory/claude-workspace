/**
 * Check network monitor status.
 * Must be called after network_monitor_setup.js has been executed.
 *
 * Executed via Selenium's driver.execute_script() which wraps code in an
 * anonymous function. IIFE wrapper makes this valid standalone JavaScript.
 *
 * @returns {Object} activeRequests, lastRequestTime, and currentTime
 */
(function() {
    return {
        activeRequests: window.__networkMonitor?.activeRequests || 0,
        lastRequestTime: window.__networkMonitor?.lastRequestTime,
        currentTime: Date.now()
    };
})();