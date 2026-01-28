# Document Search MCP Server

Semantic search over local documents using hybrid vector search (Gemini embeddings + BM25) and Qdrant.

## Prerequisites

**Qdrant** (vector database):
```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant-data:/qdrant/storage qdrant/qdrant:v1.16.2
```

**Gemini API key** ([Google AI Studio](https://aistudio.google.com/app/apikey)):
```bash
mkdir -p ~/.claude-workspace/secrets
echo "your-key" > ~/.claude-workspace/secrets/document_search_api_key
```

> **Rate Limits**: Free tier is limited to **100 requests/day**. For production use,
> [enable billing](https://ai.google.dev/gemini-api/docs/rate-limits) to unlock Tier 1
> (1,000 RPD, 300 RPM). See [Rate Limits](#gemini-api-rate-limits) below.

**MCP timeout** (default 2min is too short for large directories):
```bash
# Add to ~/.claude/settings.json under "env":
"MCP_TOOL_TIMEOUT": "1800000"  # 30 minutes (in ms)
```

**Tesseract** (optional, for OCR on scanned PDFs):
```bash
brew install tesseract
```

## Installation

For users installing from GitHub:

```bash
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/document-search
claude mcp add --scope user document-search -- mcp-document-search
```

To upgrade to the latest version:
```bash
uv tool upgrade document-search-mcp
```

For local development with editable install, see the [workspace README](../../README.md).

## Architecture

Document Search extends Claude Code with semantic retrieval, forming a distributed RAG system:

![Architecture](docs/architecture.svg)

**The RAG pattern emerges naturally:**
- **R**etrieval: Document Search finds relevant chunks
- **A**ugmented: Chunks flow back to Claude's context
- **G**eneration: Claude synthesizes the answer

**Two use cases, one pattern:**

| Use Case       | What's Indexed               | What You Get                               |
|----------------|------------------------------|--------------------------------------------|
| Project search | Code, docs, configs          | Semantic understanding of your codebase    |
| Session memory | `~/.claude/projects/*.jsonl` | Past decisions and context across sessions |

**Search type tradeoffs:**

| Type      | Path                | Best For                                |
|-----------|---------------------|-----------------------------------------|
| hybrid    | Gemini API + Qdrant | Most queries (semantic + keyword)       |
| lexical   | Qdrant only         | Exact terms, identifiers, works offline |
| embedding | Gemini API + Qdrant | Pure semantic similarity                |

## Supported File Types

markdown, text, pdf, json, jsonl, csv, email (.eml), images (placeholder for future multimodal)

## Tools

| Tool               | Description                                        |
|--------------------|----------------------------------------------------|
| `index_documents`  | Index file or directory (incremental, respects .gitignore) |
| `clear_documents`  | Remove documents from the index                    |
| `search_documents` | Search with configurable strategy and filters      |
| `list_documents`   | List indexed documents with optional filtering     |
| `get_info`         | Index health and statistics                        |

## Search Types

- **hybrid** (default): Combines semantic + keyword matching. Best for most queries.
- **lexical**: BM25 keyword search. Best for exact terms, identifiers, symbols.
- **embedding**: Dense vectors only. Useful for comparison/debugging.

## State

- Index state: `~/.claude-workspace/cache/document_search_index_state.json`
- Embedding config: `~/.claude-workspace/config/document_search.json`
- Qdrant collection: `document_chunks` on `localhost:6333`

## Gemini API Rate Limits

**Before January 2026**: `text-embedding-004` had no explicit daily request cap—only RPM limits (~10-20 RPM free tier). High-volume batch indexing worked without issues.

**After January 2026**: Google shutdown `text-embedding-004` and enforced strict daily caps on `gemini-embedding-001` due to fraud/abuse.

| Tier                          | Requests/Day | Requests/Min | Cost            |
|-------------------------------|--------------|--------------|-----------------|
| Free                          | **100**      | ~100         | $0              |
| Tier 1 (billing enabled)      | 1,000        | 300          | $0.15/1M tokens |
| Tier 2 ($250+ spend, 30 days) | 10,000+      | 1,000+       | $0.15/1M tokens |

**For RAG/indexing workloads**: Free tier (100 RPD) is insufficient for indexing more than ~100 documents. Enable billing to unlock Tier 1.
