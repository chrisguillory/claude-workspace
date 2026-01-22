# Stubs

Custom type stubs for third-party libraries missing official stubs or py.typed markers.

## Purpose

Provides type information for external packages that lack typing support, enabling full mypy validation and IDE
autocomplete.

## Current Stubs

- `ccl_chromium_reader/` - LocalStoreDb and SessionStoreDb for Chrome browser data extraction
- `dfindexeddb/` - FolderReader and record types for Chrome IndexedDB parsing
- `fitz/` - PyMuPDF Document/Page types (supplements incomplete pymupdf-stubs, see [#3361](https://github.com/pymupdf/PyMuPDF/issues/3361))
- `google/` - google-genai Client for Gemini API
- `langchain_text_splitters/` - Text splitters for chunking
- `pdfplumber/` - PDF table extraction (py.typed missing from wheel, see [#698](https://github.com/jsvine/pdfplumber/issues/698))
- `qdrant_client/` - Qdrant vector database client
- `tenacity/` - Retry decorators

## Guidelines

Stubs can be partial. Complete coverage is ideal but not required.

**For incomplete stubs:**
- Document missing coverage at top of file
- Use `__getattr__(self, name: str) -> Any: ...` for dynamic/unmapped attributes
- Add specific methods as needed for better type inference
- Avoid stubbing every method when `__getattr__` suffices

**Principles:**
- Type commonly used APIs explicitly
- Let `__getattr__` handle the rest
- Maintainability over exhaustiveness
