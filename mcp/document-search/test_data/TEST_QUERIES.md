# Test Queries for Document Search Validation

## Test Files Summary

| File | Type | Purpose | Expected Chunks |
|------|------|---------|-----------------|
| `test_simple_email.eml` | Email | Plain text email | 1-2 (content only) |
| `test_multipart_email.eml` | Email | MIME multipart | 1-2 (content, NO boundaries) |
| `test_complex_email.eml` | Email | Nested + base64 | 1-2 (content, NO base64) |
| `test_short_boilerplate.txt` | Text | Short lines | 0 (all filtered) |
| `test_project_docs.md` | Markdown | Technical docs | 3-5 |
| `test_company_info.json` | JSON | Company data | 2-4 |
| `test_employees.csv` | CSV | Employee records | 2-4 |
| `test_technical_spec.txt` | Text | Tech spec | 2-3 |

---

## Issue #2: Email MIME Parsing

### Query 1: Simple Email
```
Query: "Q4 budget review meeting revenue projections"
Expected: Find test_simple_email.eml
Verify: Results contain "Total revenue exceeded projections by 12%"
Verify: NO MIME boundaries in results
```

### Query 2: Multipart Email
```
Query: "Order 12345 Bluetooth Headphones XZ-500"
Expected: Find test_multipart_email.eml
Verify: Results contain "Tracking Number: 1Z999AA10123456784"
Verify: NO "------=_Part_ABC123" in results
Verify: NO HTML tags in results
```

### Query 3: Complex Email with Attachment
```
Query: "Contract Amendment SA-2025-001 liability cap"
Expected: Find test_complex_email.eml
Verify: Results contain "Updated liability cap to $5,000,000"
Verify: NO base64 strings like "JVBERi0xLjcK"
Verify: NO "NextPart_000" boundaries
```

---

## Issue #3: Boilerplate Filtering (MIN_CHUNK_LENGTH=50)

### Query 4: Short Boilerplate
```
Query: "Page 1 of 5 confidential draft"
Expected: NO results from test_short_boilerplate.txt
Reason: All content is under 50 characters per chunk
```

### Query 5: Verify stats
```
After indexing, run get_index_stats()
Check that test_short_boilerplate.txt contributed 0 chunks
```

---

## Sanity Checks: Good Content Still Works

### Query 6: Markdown
```
Query: "Apache Kafka partitions real-time streaming"
Expected: Find test_project_docs.md
Verify: heading_context includes "Project Phoenix"
```

### Query 7: JSON
```
Query: "Jennifer Martinez CEO TechStartup CloudSync"
Expected: Find test_company_info.json
Verify: json_path metadata present
```

### Query 8: CSV
```
Query: "James Wilson Engineering salary 140000"
Expected: Find test_employees.csv
```

### Query 9: Text
```
Query: "XR-7000 Industrial Controller ARM Cortex"
Expected: Find test_technical_spec.txt
```

---

## Exact Match Test (Hybrid Search BM25)

### Query 10: Exact product name
```
Query: "DataVault Enterprise"
Expected: Score close to 1.0 (BM25 exact match boost)
```

### Query 11: VIN-like unique identifier
```
Query: "1Z999AA10123456784"
Expected: Score = 1.0 (exact match)
```
