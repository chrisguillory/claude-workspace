"""Type stubs for bm25_rs native extension."""

from collections.abc import Sequence

class BM25Model:
    """BM25 sparse embedding model using Rust + rayon parallelism.

    Thread-safe: immutable model with rayon thread-local mutable state.
    Releases the GIL during computation via py.allow_threads().
    """

    def __init__(
        self,
        k: float = 1.2,
        b: float = 0.75,
        avg_doc_len: float = 256.0,
    ) -> None:
        """Create BM25 model with parameters matching fastembed Qdrant/bm25.

        Args:
            k: Term frequency saturation parameter (must be >= 0).
            b: Document length normalization.
            avg_doc_len: Assumed average document length in tokens (must be > 0).

        Raises:
            ValueError: If k < 0 or avg_doc_len <= 0.
        """
        ...
    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[list[tuple[list[int], list[float]]], float, float]:
        """Embed texts in parallel using rayon.

        Returns:
            (results, wall_secs, cpu_secs) where:
            - results: Sparse vectors as (token_ids, tf_scores) per text
            - wall_secs: Wall-clock time for the parallel section
            - cpu_secs: Sum of per-task durations (total CPU across all cores)
        """
        ...
    @staticmethod
    def thread_count() -> int:
        """Number of rayon worker threads in the global thread pool."""
        ...
    def embed_batch_bytes(
        self,
        texts: Sequence[str],
    ) -> tuple[bytes, bytes, bytes]:
        """Embed texts returning packed native-endian bytes.

        Returns:
            (offsets_bytes, indices_bytes, values_bytes) where:
            - offsets_bytes: u32 array, len(texts)+1 entries
            - indices_bytes: u32 array, flat packed token IDs
            - values_bytes: f64 array, flat packed BM25 TF scores

            Text i spans indices[offsets[i]:offsets[i+1]].

        Possible improvements:
            - Add per-task timing instrumentation (wall_secs, cpu_secs)
              matching embed_batch, if this API is adopted for production use.
        """
        ...
    def query_embed(self, text: str) -> tuple[list[int], list[float]]:
        """Embed a single query text (values are all 1.0, deduplicated tokens).

        Unlike embed_batch, no TF weighting — matches fastembed query_embed behavior.
        """
        ...
