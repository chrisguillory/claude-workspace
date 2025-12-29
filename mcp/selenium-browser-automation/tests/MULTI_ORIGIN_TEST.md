# Multi-Origin localStorage Test

Integration test for multi-origin localStorage capture and restore via CDP.

## Setup

Terminal 1 - Start test servers:
```bash
cd /Users/chris/claude-workspace/mcp/selenium-browser-automation
uv run tests/serve_test_pages.py
```

This starts servers on ports 8001, 8002, 8003.

## Test 1: Multi-Origin Capture

**Goal:** Verify `save_storage_state()` captures localStorage from ALL visited origins.

### Steps

1. Navigate to first origin, set localStorage:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({origin: 'port8001', key1: 'value1'})")
```

2. Navigate to second origin, set localStorage:
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("window.setTestData({origin: 'port8002', key2: 'value2'})")
```

3. Navigate to third origin, set localStorage:
```
navigate("http://localhost:8003/storage-test-page.html")
execute_javascript("window.setTestData({origin: 'port8003', key3: 'value3'})")
```

4. Save storage state:
```
save_storage_state("multi_origin_test.json")
```

### Verify

Check the result shows:
- `tracked_origins` contains all 3: `["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"]`
- `origins_count` is 3

Read the JSON file and verify it contains localStorage for all 3 origins:
```json
{
  "origins": [
    {"origin": "http://localhost:8001", "localStorage": [{"name": "origin", "value": "\"port8001\""}, ...]},
    {"origin": "http://localhost:8002", "localStorage": [{"name": "origin", "value": "\"port8002\""}, ...]},
    {"origin": "http://localhost:8003", "localStorage": [{"name": "origin", "value": "\"port8003\""}, ...]}
  ]
}
```

---

## Test 2: Multi-Origin Lazy Restore

**Goal:** Verify lazy restore pattern - localStorage is restored for each origin as we navigate there (not upfront).

### Steps

1. Navigate to first origin with storage state (fresh browser clears previous state):
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True, storage_state_file="multi_origin_test.json")
```

2. Verify localStorage on current origin (8001):
```
execute_javascript("window.getTestData()")
```
Should return: `{origin: "port8001", key1: "value1"}`

3. Navigate to second origin (WITHOUT fresh_browser, so storage state persists):
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("window.getTestData()")
```
Should return: `{origin: "port8002", key2: "value2"}`

4. Navigate to third origin:
```
navigate("http://localhost:8003/storage-test-page.html")
execute_javascript("window.getTestData()")
```
Should return: `{origin: "port8003", key3: "value3"}`

### Verify

All 3 origins have their localStorage restored lazily as we navigate to each.
The `pending_storage_state` persists in memory and triggers restore for each new origin.

**Technical note:** CDP `DOMStorage.setDOMStorageItem` requires an active frame for the target origin,
so we cannot restore localStorage upfront. Lazy restore solves this by restoring on-demand.

---

## Test 3: Origin Tracking Isolation

**Goal:** Verify only VISITED origins are tracked and captured.

### Steps

1. Fresh browser, visit ONLY port 8001:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({origin: 'only8001'})")
```

2. Save storage state:
```
save_storage_state("single_origin_test.json")
```

### Verify

- `tracked_origins` contains ONLY `["http://localhost:8001"]`
- `origins_count` is 1
- JSON file only has one origin entry

---

## Test 4: Fresh Browser Clears Tracking

**Goal:** Verify `fresh_browser=True` resets origin tracking.

### Steps

1. Visit all 3 origins:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
navigate("http://localhost:8002/storage-test-page.html")
navigate("http://localhost:8003/storage-test-page.html")
```

2. Fresh browser, visit only 8001:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({afterFresh: 'yes'})")
```

3. Save storage state:
```
save_storage_state("after_fresh_test.json")
```

### Verify

- `tracked_origins` contains ONLY `["http://localhost:8001"]` (8002 and 8003 were cleared)
- JSON only has port 8001

---

## Test 5: Empty Origin Handling

**Goal:** Verify origins with no localStorage are tracked but not saved.

### Steps

1. Visit 8001 and set data, visit 8002 but DON'T set data:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({hasData: true})")
navigate("http://localhost:8002/storage-test-page.html")
# Don't set any data on 8002
```

2. Save storage state:
```
save_storage_state("empty_origin_test.json")
```

### Verify

- `tracked_origins` contains BOTH 8001 and 8002
- `origins_count` is 1 (only 8001 has data)
- JSON only contains 8001 (8002 had no localStorage to save)

---

## Cleanup

Stop the test servers with Ctrl+C in Terminal 1.

Delete test files:
```bash
rm -f multi_origin_test.json single_origin_test.json after_fresh_test.json empty_origin_test.json
```