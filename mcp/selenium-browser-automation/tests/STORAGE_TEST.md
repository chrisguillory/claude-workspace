# Multi-Origin Storage Tests

Comprehensive integration tests for browser storage capture and restore via CDP and JavaScript.

## Overview

This document tests the **storage persistence** feature of the Selenium Browser Automation MCP server. The feature captures and restores browser storage across all visited origins, enabling session persistence for authenticated workflows.

### Storage Types Tested

| Type | Persistence | Scope | Capture Method | Restore Method |
|------|-------------|-------|----------------|----------------|
| **localStorage** | Persistent | Origin | CDP DOMStorage | CDP DOMStorage |
| **sessionStorage** | Session-only | Origin + Tab | CDP DOMStorage | CDP DOMStorage |
| **IndexedDB** | Persistent | Origin | JavaScript | JavaScript |

### Core Patterns

1. **Pre-capture**: Cache storage before navigating away (CDP can't query departed origins)
2. **Lazy restore**: Restore on-demand when arriving at each origin (CDP write requires active frame)
3. **Origin tracking**: Track all visited origins for multi-origin capture

### Implementation References

- `_capture_current_origin_storage()` - Captures all three storage types before navigation
- `_restore_pending_storage_for_current_origin()` - Restores all three types lazily
- `save_storage_state()` - Saves cookies + multi-origin storage to JSON
- `BrowserState` caches: `local_storage_cache`, `session_storage_cache`, `indexed_db_cache`

---

## Test Infrastructure

### Test Servers

Three HTTP servers on different ports (different origins):
- `http://localhost:8001` - Origin 1
- `http://localhost:8002` - Origin 2
- `http://localhost:8003` - Origin 3

### Test Page API

The test page (`storage-test-page.html`) exposes these functions:

```javascript
// localStorage
window.setTestData(data)           // Set localStorage data
window.getTestData()               // Get all localStorage
window.clearTestData()             // Clear localStorage
window.verifyTestData(expected)    // Verify localStorage matches expected

// sessionStorage
window.setSessionData(data)        // Set sessionStorage data
window.getSessionData()            // Get all sessionStorage
window.clearSessionData()          // Clear sessionStorage
window.verifySessionData(expected) // Verify sessionStorage matches expected

// IndexedDB (all async - use await)
window.setIndexedDBData(dbName, storeName, records, keyPath='id')
window.getIndexedDBData(dbName, storeName)
window.getIndexedDBInfo()          // List all databases with metadata
window.clearIndexedDB()            // Delete all IndexedDB databases
window.verifyIndexedDBData(dbName, storeName, expected)

// Combined
window.setAllData(localData, sessionData, indexedDBData)
window.getAllData()                // Returns {localStorage, sessionStorage, indexedDB}
window.clearAllData()              // Clear all storage types
```

### MCP Tools Used

- `navigate(url, fresh_browser, storage_state_file)` - Navigation with optional state import
- `execute_javascript(code)` - Run JavaScript in page context
- `save_storage_state(filename, include_indexeddb)` - Export storage state

---

## Setup

### Terminal 1 - Start Test Servers

```bash
cd /Users/chris/claude-workspace/mcp/selenium-browser-automation
uv run tests/serve_test_pages.py
```

Expected output:
```
Starting test servers...
  http://localhost:8001 -> tests/fixtures/
  http://localhost:8002 -> tests/fixtures/
  http://localhost:8003 -> tests/fixtures/
Press Ctrl+C to stop
```

### Terminal 2 - Connect MCP

Ensure the selenium-browser-automation MCP server is connected:
```bash
claude mcp list  # Should show selenium-browser-automation
```

---

## Part 1: localStorage Tests

These tests verify multi-origin localStorage capture and restore.

### Test L1: Multi-Origin Capture

**Goal:** Verify `save_storage_state()` captures localStorage from ALL visited origins, not just the current one.

**Why this matters:** CDP DOMStorage can only query the current origin's frame. We cache localStorage before navigating away so departed origins aren't lost.

#### Steps

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
save_storage_state("test_L1_localstorage.json")
```

#### Verify

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

#### Success Criteria
- [ ] Result shows 3 tracked origins
- [ ] Result shows 3 origins with data
- [ ] JSON file contains localStorage for all 3 origins
- [ ] Each origin's data matches what was set

---

### Test L2: Multi-Origin Lazy Restore

**Goal:** Verify lazy restore pattern - localStorage is restored for each origin as we navigate there (not upfront).

**Why this matters:** CDP `DOMStorage.setDOMStorageItem` requires an active frame for the target origin. We can't restore all origins upfront; we must restore on-demand.

#### Steps

1. Navigate to first origin with storage state (fresh browser clears previous state):
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True, storage_state_file="test_L1_localstorage.json")
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

#### Verify

- All 3 origins have their localStorage restored lazily as we navigate to each
- Check stderr logs for: `[storage] Restored N localStorage for http://localhost:800X`

#### Success Criteria
- [ ] Origin 8001 data restored on initial navigation
- [ ] Origin 8002 data restored on second navigation
- [ ] Origin 8003 data restored on third navigation
- [ ] No double-restore (each origin restored exactly once)

---

### Test L3: Origin Tracking Isolation

**Goal:** Verify only VISITED origins are tracked and captured.

#### Steps

1. Fresh browser, visit ONLY port 8001:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({origin: 'only8001'})")
```

2. Save storage state:
```
save_storage_state("test_L3_single.json")
```

#### Verify

- `tracked_origins` contains ONLY `["http://localhost:8001"]`
- `origins_count` is 1
- JSON file only has one origin entry

#### Success Criteria
- [ ] Only visited origin is tracked
- [ ] Unvisited origins (8002, 8003) are not in output

---

### Test L4: Fresh Browser Clears Tracking

**Goal:** Verify `fresh_browser=True` resets origin tracking and caches.

#### Steps

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
save_storage_state("test_L4_fresh.json")
```

#### Verify

- `tracked_origins` contains ONLY `["http://localhost:8001"]` (8002 and 8003 were cleared)
- JSON only has port 8001

#### Success Criteria
- [ ] fresh_browser=True clears origin tracking
- [ ] Previous origins not included in new save

---

### Test L5: Empty Origin Handling

**Goal:** Verify origins with no localStorage are tracked but not saved in origins array.

#### Steps

1. Visit 8001 and set data, visit 8002 but DON'T set data:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({hasData: true})")
navigate("http://localhost:8002/storage-test-page.html")
# Don't set any data on 8002
```

2. Save storage state:
```
save_storage_state("test_L5_empty.json")
```

#### Verify

- `tracked_origins` contains BOTH 8001 and 8002
- `origins_count` is 1 (only 8001 has data)
- JSON only contains 8001 (8002 had no localStorage to save)

#### Success Criteria
- [ ] Empty origins are tracked
- [ ] Empty origins are NOT included in saved JSON
- [ ] Non-empty origins are saved correctly

---

## Part 2: sessionStorage Tests

sessionStorage follows the same patterns as localStorage, but with session-scoped persistence.

**Important:** sessionStorage is cleared when the browser closes. Restored sessionStorage will persist only for the current browser session.

### Test S1: Multi-Origin sessionStorage Capture

**Goal:** Verify `save_storage_state()` captures sessionStorage from all visited origins.

#### Steps

1. Navigate and set sessionStorage on multiple origins:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setSessionData({session: 'port8001', visitId: 'abc'})")

navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("window.setSessionData({session: 'port8002', visitId: 'def'})")
```

2. Save storage state:
```
save_storage_state("test_S1_session.json")
```

#### Verify

Read the JSON file and verify sessionStorage is captured:
```json
{
  "origins": [
    {"origin": "http://localhost:8001", "localStorage": [], "sessionStorage": [{"name": "session", "value": "\"port8001\""}, ...]},
    {"origin": "http://localhost:8002", "localStorage": [], "sessionStorage": [{"name": "session", "value": "\"port8002\""}, ...]}
  ]
}
```

#### Success Criteria
- [ ] sessionStorage captured for both origins
- [ ] Data matches what was set

---

### Test S2: Multi-Origin sessionStorage Restore

**Goal:** Verify lazy restore pattern for sessionStorage.

#### Steps

1. Navigate with storage state:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True, storage_state_file="test_S1_session.json")
execute_javascript("window.getSessionData()")
```
Should return: `{session: "port8001", visitId: "abc"}`

2. Navigate to second origin:
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("window.getSessionData()")
```
Should return: `{session: "port8002", visitId: "def"}`

#### Verify

- Check stderr logs for: `[storage] Restored N sessionStorage for http://localhost:800X`

#### Success Criteria
- [ ] sessionStorage restored on 8001
- [ ] sessionStorage restored on 8002 (lazy)

---

### Test S3: Session Scope Documentation

**Goal:** Document that sessionStorage is session-scoped (this is expected browser behavior).

**Note:** This is not a failure case - it's how browsers work.

#### Behavior

1. Save sessionStorage via `save_storage_state()`
2. Close browser (`fresh_browser=True` on next navigate)
3. Restore via `storage_state_file`
4. sessionStorage is restored and works during the session
5. When browser closes again, sessionStorage is cleared

**Use case:** Useful for preserving mid-session state (form data, navigation state, etc.) within a single automation run.

**Alternative:** For cross-session persistence, use localStorage or cookies instead.

---

### Test S4: Empty sessionStorage Handling

**Goal:** Verify origins with no sessionStorage are handled correctly.

#### Steps

1. Set sessionStorage only on 8001:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setSessionData({hasSession: true})")
navigate("http://localhost:8002/storage-test-page.html")
# Don't set sessionStorage on 8002
```

2. Save and verify:
```
save_storage_state("test_S4_empty.json")
```

#### Verify

- JSON contains sessionStorage only for 8001
- 8002 may appear in origins array with empty sessionStorage, or not appear at all (implementation detail)

#### Success Criteria
- [ ] Empty sessionStorage doesn't cause errors
- [ ] Non-empty sessionStorage is saved correctly

---

## Part 3: IndexedDB Tests

IndexedDB has unique characteristics:
- Complex data structures (databases → object stores → records)
- Async API (requires `await` in execute_javascript)
- No CDP write API (restoration uses JavaScript)
- Supports complex types (Date, Map, Set, typed arrays)

### Test I1: Multi-Origin IndexedDB Capture

**Goal:** Verify `save_storage_state(include_indexeddb=True)` captures IndexedDB from all visited origins.

#### Steps

1. Navigate and create IndexedDB on first origin:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("await window.setIndexedDBData('testDB_8001', 'users', [{id: 1, name: 'Alice from 8001'}, {id: 2, name: 'Bob from 8001'}])")
```

2. Navigate and create IndexedDB on second origin:
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("await window.setIndexedDBData('testDB_8002', 'users', [{id: 1, name: 'Alice from 8002'}])")
```

3. Navigate and create IndexedDB on third origin:
```
navigate("http://localhost:8003/storage-test-page.html")
execute_javascript("await window.setIndexedDBData('testDB_8003', 'items', [{id: 100, product: 'Widget'}])")
```

4. Save storage state with IndexedDB:
```
save_storage_state("test_I1_indexeddb.json", include_indexeddb=True)
```

#### Verify

Check the result shows:
- `indexeddb_databases_count` is 3 (one per origin)
- `indexeddb_records_count` is 4 (2 + 1 + 1)

Read the JSON file and verify IndexedDB structure:
```json
{
  "origins": [
    {
      "origin": "http://localhost:8001",
      "localStorage": [],
      "indexedDB": [{
        "databaseName": "testDB_8001",
        "version": 1,
        "objectStores": [{
          "name": "users",
          "keyPath": "id",
          "records": [
            {"key": 1, "value": {"id": 1, "name": "Alice from 8001"}},
            {"key": 2, "value": {"id": 2, "name": "Bob from 8001"}}
          ]
        }]
      }]
    },
    ...
  ]
}
```

#### Success Criteria
- [ ] IndexedDB captured for all 3 origins
- [ ] Database names correct
- [ ] Object store names correct
- [ ] Records match what was set
- [ ] Record count in result matches

---

### Test I2: Multi-Origin IndexedDB Restore

**Goal:** Verify lazy restore pattern for IndexedDB.

#### Steps

1. Navigate with storage state:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True, storage_state_file="test_I1_indexeddb.json")
```

2. Verify IndexedDB on 8001:
```
execute_javascript("await window.getIndexedDBData('testDB_8001', 'users')")
```
Should return: `[{id: 1, name: "Alice from 8001"}, {id: 2, name: "Bob from 8001"}]`

3. Navigate to second origin:
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("await window.getIndexedDBData('testDB_8002', 'users')")
```
Should return: `[{id: 1, name: "Alice from 8002"}]`

4. Navigate to third origin:
```
navigate("http://localhost:8003/storage-test-page.html")
execute_javascript("await window.getIndexedDBData('testDB_8003', 'items')")
```
Should return: `[{id: 100, product: "Widget"}]`

#### Verify

- Check stderr logs for: `[storage] Restored N IndexedDB databases for http://localhost:800X`

#### Success Criteria
- [ ] IndexedDB restored on 8001 (initial navigation)
- [ ] IndexedDB restored on 8002 (lazy)
- [ ] IndexedDB restored on 8003 (lazy)
- [ ] All records match original data

---

### Test I3: IndexedDB Type Serialization

**Goal:** Verify complex JavaScript types survive the capture/restore cycle.

**Types tested:** Date, nested objects, arrays

#### Steps

1. Create IndexedDB with complex types:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript(`
  await window.setIndexedDBData('complexDB', 'data', [
    {
      id: 1,
      created: new Date('2024-12-29T12:00:00Z'),
      tags: ['alpha', 'beta', 'gamma'],
      metadata: {
        nested: {
          deep: 'value'
        }
      }
    }
  ])
`)
```

2. Save storage state:
```
save_storage_state("test_I3_types.json", include_indexeddb=True)
```

3. Fresh browser, restore:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True, storage_state_file="test_I3_types.json")
```

4. Verify types:
```
execute_javascript(`
  const records = await window.getIndexedDBData('complexDB', 'data');
  const record = records[0];
  return {
    hasDate: record.created instanceof Date || typeof record.created === 'object',
    dateValue: record.created,
    tagsLength: record.tags.length,
    nestedValue: record.metadata.nested.deep
  };
`)
```

#### Verify

- Date is serialized/deserialized (may be string or Date object depending on serialization)
- Arrays are preserved
- Nested objects are preserved

#### Success Criteria
- [ ] Complex types don't cause errors
- [ ] Arrays preserved correctly
- [ ] Nested objects preserved correctly
- [ ] Dates serialized (as ISO string or Date object)

---

### Test I4: Empty IndexedDB Handling

**Goal:** Verify origins with no IndexedDB are handled correctly.

#### Steps

1. Create IndexedDB only on 8001:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("await window.setIndexedDBData('testDB', 'store', [{id: 1}])")
navigate("http://localhost:8002/storage-test-page.html")
# Don't create any IndexedDB on 8002
```

2. Save with IndexedDB:
```
save_storage_state("test_I4_empty.json", include_indexeddb=True)
```

#### Verify

- JSON contains IndexedDB only for 8001
- 8002 has no indexedDB array (or empty array)
- No errors during capture

#### Success Criteria
- [ ] Empty IndexedDB doesn't cause errors
- [ ] Non-empty IndexedDB is saved correctly
- [ ] Empty origins don't have indexedDB in output

---

### Test I5: Multiple Databases per Origin

**Goal:** Verify multiple IndexedDB databases on the same origin are captured.

#### Steps

1. Create multiple databases:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("await window.setIndexedDBData('db1', 'store1', [{id: 1, from: 'db1'}])")
execute_javascript("await window.setIndexedDBData('db2', 'store2', [{id: 2, from: 'db2'}])")
```

2. Save storage state:
```
save_storage_state("test_I5_multi_db.json", include_indexeddb=True)
```

#### Verify

- JSON contains both databases for 8001
- `indexeddb_databases_count` is 2

#### Success Criteria
- [ ] Both databases captured
- [ ] Each database has correct data

---

## Part 4: Combined Storage Tests

### Test C1: All Storage Types Together

**Goal:** Verify localStorage, sessionStorage, and IndexedDB all work together.

#### Steps

1. Set all storage types on 8001:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({local: 'data8001'})")
execute_javascript("window.setSessionData({session: 'data8001'})")
execute_javascript("await window.setIndexedDBData('combinedDB', 'store', [{id: 1, name: 'Combined8001'}])")
```

2. Set all storage types on 8002:
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("window.setTestData({local: 'data8002'})")
execute_javascript("window.setSessionData({session: 'data8002'})")
execute_javascript("await window.setIndexedDBData('combinedDB', 'store', [{id: 2, name: 'Combined8002'}])")
```

3. Save storage state:
```
save_storage_state("test_C1_combined.json", include_indexeddb=True)
```

4. Fresh browser, restore:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True, storage_state_file="test_C1_combined.json")
```

5. Verify all three on 8001:
```
execute_javascript("window.getTestData()")
execute_javascript("window.getSessionData()")
execute_javascript("await window.getIndexedDBData('combinedDB', 'store')")
```

6. Navigate to 8002 and verify all three:
```
navigate("http://localhost:8002/storage-test-page.html")
execute_javascript("window.getTestData()")
execute_javascript("window.getSessionData()")
execute_javascript("await window.getIndexedDBData('combinedDB', 'store')")
```

#### Verify

- All three storage types restored on both origins
- Data matches what was set

#### Success Criteria
- [ ] localStorage restored on 8001 and 8002
- [ ] sessionStorage restored on 8001 and 8002
- [ ] IndexedDB restored on 8001 and 8002
- [ ] All data matches original

---

### Test C2: Selective Capture (IndexedDB opt-in)

**Goal:** Verify `include_indexeddb=False` (default) doesn't capture IndexedDB.

#### Steps

1. Set all storage types:
```
navigate("http://localhost:8001/storage-test-page.html", fresh_browser=True)
execute_javascript("window.setTestData({local: 'yes'})")
execute_javascript("window.setSessionData({session: 'yes'})")
execute_javascript("await window.setIndexedDBData('testDB', 'store', [{id: 1}])")
```

2. Save WITHOUT include_indexeddb:
```
save_storage_state("test_C2_no_indexeddb.json")
```

#### Verify

- JSON has localStorage and sessionStorage
- JSON does NOT have indexedDB
- `indexeddb_databases_count` is None in result

#### Success Criteria
- [ ] localStorage captured
- [ ] sessionStorage captured
- [ ] IndexedDB NOT captured (opt-in required)

---

## Cleanup

### Stop Test Servers

Press Ctrl+C in Terminal 1 where `serve_test_pages.py` is running.

### Delete Test Files

```bash
cd /Users/chris/claude-workspace/mcp/selenium-browser-automation
rm -f test_L*.json test_S*.json test_I*.json test_C*.json
```

---

## Technical Notes

### CDP DOMStorage Limitation

CDP `DOMStorage.setDOMStorageItem` requires an active frame for the target origin. This is why we use the lazy restore pattern - we can't restore localStorage/sessionStorage for origins we haven't navigated to yet.

### IndexedDB CDP Limitation

CDP IndexedDB domain has READ APIs but NO WRITE APIs. This is documented Chromium behavior (issues #40884867, #40887598). Therefore:
- **Capture:** Uses JavaScript (`indexeddb_capture.js`)
- **Restore:** Uses JavaScript (`indexeddb_restore.js`)

### Type Serialization

IndexedDB supports complex types that don't serialize to JSON directly. The capture script uses `__type` markers:

```javascript
// Serialization
Date → { __type: 'Date', __value: '2024-12-29T12:00:00.000Z' }
Map  → { __type: 'Map', __value: [[key1, val1], [key2, val2]] }
Set  → { __type: 'Set', __value: [val1, val2, val3] }
Uint8Array → { __type: 'Uint8Array', __value: 'base64...' }
```

The restore script deserializes these back to native types.

### Playwright Compatibility

The storage state JSON format is compatible with Playwright's `storageState`:
- Cookies: Same format
- localStorage: Same format
- sessionStorage: Our extension (Playwright doesn't support this - open issue #31108)
- IndexedDB: Our extension (Playwright added in v1.51 with `indexedDB: true`)

### Performance

- localStorage/sessionStorage capture: ~1ms per origin
- IndexedDB capture: ~100ms per origin (JavaScript execution)
- For sites with large IndexedDB databases, capture may take longer

### Chrome Version Requirement

`indexedDB.databases()` requires Chrome 71+ (released December 2018).

---

## Troubleshooting

### "Frame not found for the given storage id"

This error occurs when trying to set storage for an origin without an active frame. The lazy restore pattern should prevent this, but if you see it:
1. Ensure you've navigated to the origin before trying to restore
2. Check that the URL matches the origin in the storage state file

### IndexedDB Not Restoring

1. Check that `include_indexeddb=True` was used when saving
2. Check stderr logs for `[storage] Restored N IndexedDB databases`
3. Verify the database name matches exactly

### Empty Storage State

If `save_storage_state()` returns 0 origins:
1. Verify you set data before navigating away
2. Check that `fresh_browser=True` wasn't called between setting data and saving
3. Verify the storage was actually set (use the test page UI to check)

### Test Servers Not Running

If navigation fails with connection refused:
1. Ensure `uv run tests/serve_test_pages.py` is running in Terminal 1
2. Check that ports 8001, 8002, 8003 are not in use by other processes