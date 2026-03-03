# Possible Improvements

* Collection verification on startup - detect when Qdrant collection is empty/missing but index state exists
  in Redis (e.g., after Docker container deletion); currently this mismatch causes files to be skipped despite
  missing vectors. `ensure_collection` creates collections but doesn't verify vector counts match index state.

* Query transformation techniques for improved retrieval:
  - **HyDE** (Hypothetical Document Embeddings) - generate a hypothetical answer, embed that instead of the query;
    bridges the query-document style gap in embedding space
  - **Query expansion** - add synonyms, related terms, or rephrase the query to capture more relevant results
  - **Multi-query** - generate multiple query variations, retrieve for each, merge results; improves recall for
    ambiguous queries
  - **Step-back prompting** - abstract the query to a higher-level concept before retrieval; helps with specific
    questions that need general background

* `embed_batch_bytes` in bm25-rs has no timing instrumentation - returns raw results without wall_secs/cpu_secs.
  No production callers currently, but should be added if the bytes API is used.

* Dashboard `server.py` is 1,700 lines of mixed Python + inline HTML/JS/CSS. Extracting the frontend to a
  separate file would improve maintainability (JS linting, syntax highlighting in editors, etc.)

* Embedding cache scaling refinements (relevant if keyspace reaches millions):
  - **Collection-level reverse index** - `embed-idx:collection:{name}` → Set of file reverse index keys;
    avoids Redis SCAN for collection-level cache invalidation (SCAN throughput degrades significantly at
    scale with concurrent operations)
  - **SSCAN for large reverse index sets** - replace SMEMBERS with cursor-based SSCAN for files producing
    thousands of chunks (e.g., large JSONL files with 77K+ chunks); prevents large single-response payloads
  - **Lua scripts for atomic invalidation** - server-side SMEMBERS + UNLINK in a single Lua script prevents
    partial invalidation on client crash mid-operation; currently benign (orphans TTL away) but matters if
    cache correctness becomes critical
  - **Extend content hash from 64-bit to 128/256-bit** - current `sha256[:16]` has birthday collision at
    ~2^32 entries (safe at current scale); extending eliminates theoretical collisions but forces
    re-embedding of all existing cached content (migration path or dual-read fallback needed)