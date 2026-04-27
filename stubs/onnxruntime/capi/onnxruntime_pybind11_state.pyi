"""Type stubs for onnxruntime C-extension exceptions.

Only NoSuchFile is exposed because that's the only symbol we catch
(see mcp/document-search/document_search/services/reranker.py).
"""

class NoSuchFile(Exception): ...
