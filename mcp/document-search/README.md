# Document Search MCP Server

Semantic search over local documents using Gemini embeddings and Qdrant.

## Possible Improvements
* Use `pyrate-limiter` for proactive throttling (instead of `tenacity` retry with exponential backoff on API errors)
* SQLite for state persistence (instead of JSON file) - enables atomic single-row updates, crash recovery without
  full-file rewrites, and queryable index metadata (e.g., "files indexed since X"); current JSON approach writes entire
  state file per batch which scales poorly beyond 10K files