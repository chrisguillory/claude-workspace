# Document Search MCP Server

Semantic search over local documents using hybrid vector search (Gemini embeddings + BM25) and Qdrant.

## Prerequisites

**Qdrant** (vector database):
```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant-data:/qdrant/storage qdrant/qdrant:v1.16.2
```

**Gemini API key** ([free at Google AI Studio](https://aistudio.google.com/app/apikey)):
```bash
mkdir -p ~/.claude-workspace/secrets
echo "your-key" > ~/.claude-workspace/secrets/document_search_api_key
```

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

## Supported File Types

markdown, text, pdf, json, jsonl, csv, email (.eml), images (placeholder for future multimodal)

## Tools

| Tool                  | Description                                        |
|-----------------------|----------------------------------------------------|
| `index_directory`     | Index documents (incremental, respects .gitignore) |
| `search_documents`    | Search with configurable strategy and filters      |
| `get_index_stats`     | Collection statistics                              |
| `get_index_breakdown` | Chunk counts by file type                          |
| `is_file_indexed`     | Check if a specific file is indexed                |
| `list_indexed_files`  | List all indexed files                             |

## Search Types

- **hybrid** (default): Combines semantic + keyword matching. Best for most queries.
- **lexical**: BM25 keyword search. Best for exact terms, identifiers, symbols.
- **embedding**: Dense vectors only. Useful for comparison/debugging.

## State

- Index state: `~/.claude-workspace/cache/document_search_index_state.json`
- Qdrant collection: `document_chunks` on `localhost:6333`
