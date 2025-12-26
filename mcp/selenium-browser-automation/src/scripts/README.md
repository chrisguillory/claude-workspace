# Selenium JavaScript Scripts

JavaScript executed via Selenium's `driver.execute_script()`.

## Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `text_extraction.js` | Smart DOM text extraction | `arguments[0]`: selector |
| `aria_snapshot.js` | ARIA accessibility tree | `arguments[0]`: selector, `arguments[1]`: includeUrls |
| `network_monitor_setup.js` | Inject Fetch/XHR instrumentation | none |
| `network_monitor_check.js` | Poll network idle status | none |
| `web_vitals.js` | Core Web Vitals collection | `arguments[0]`: timeoutMs |
| `resource_timing.js` | Performance API timing | none |
| `safe_serialize.js` | Safe value serialization | used via `__init__.py` builder |

## IIFE Pattern

Selenium wraps scripts in an anonymous function at runtime, making bare `return` valid. But IDEs see files in isolation and report "return outside function".

We wrap in IIFEs for valid standalone JavaScript:

```javascript
// noinspection JSUnresolvedReference
(function() {
    return extractAllText(arguments[0]);
}).apply(null, arguments);
```

**`// noinspection JSUnresolvedReference`** - Suppresses PyCharm warning about `arguments` being unresolved (it comes from Selenium's wrapper at runtime).

## Loading

Scripts are loaded at import time by `__init__.py`. The loader validates all files exist (fail-fast).
