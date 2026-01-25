# Document Search MCP Server

Semantic search over local documents using Gemini embeddings and Qdrant.

## Prerequisites

**Qdrant** (vector database):
```bash
cd mcp/document-search && docker compose up -d
```

**Gemini API key**:
```bash
mkdir -p ~/.claude-workspace/secrets
echo "your-key" > ~/.claude-workspace/secrets/document_search_api_key
```

## Installation

```bash
uv tool install --editable /path/to/claude-workspace/mcp/document-search
claude mcp add --scope user document-search -- mcp-docsearch-server
```

## Tools

| Tool               | Description                                                |
|--------------------|------------------------------------------------------------|
| `index_directory`  | Index documents in a directory (markdown, text, JSON, PDF) |
| `search_documents` | Semantic search with file type and path filters            |
| `get_index_stats`  | Get index statistics                                       |

## State

- Index state: `~/.claude-workspace/cache/document_search_index_state.json`
- Qdrant collection: `document_chunks` on `localhost:6333`

## Possible Improvements

* SQLite for state persistence (instead of JSON file) - enables atomic single-row updates, crash recovery without
  full-file rewrites, and queryable index metadata; current JSON approach scales poorly beyond 10K files
* Use `pyrate-limiter` for proactive throttling (instead of `tenacity` retry with exponential backoff on API errors)
* Progress callbacks during file processing - currently `on_progress` is only called during scanning phase; for CLI
  usage, periodic updates during the worker processing phase would improve user experience
* Collection verification on startup - detect when Qdrant collection is empty/missing but state file exists (e.g.,
  after Docker container deletion); currently this mismatch causes files to be skipped despite missing vectors
* Query transformation techniques for improved retrieval:
  - **HyDE** (Hypothetical Document Embeddings) - generate a hypothetical answer, embed that instead of the query;
    bridges the query-document style gap in embedding space
  - **Query expansion** - add synonyms, related terms, or rephrase the query to capture more relevant results
  - **Multi-query** - generate multiple query variations, retrieve for each, merge results; improves recall for
    ambiguous queries
  - **Step-back prompting** - abstract the query to a higher-level concept before retrieval; helps with specific
    questions that need general background