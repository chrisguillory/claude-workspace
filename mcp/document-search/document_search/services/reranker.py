"""Cross-encoder reranking service using fastembed ONNX backend."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

import onnxruntime.capi.onnxruntime_pybind11_state
from fastembed.rerank.cross_encoder import TextCrossEncoder

from document_search.schemas import SearchHit, SearchResult

__all__ = [
    'RerankerService',
]

logger = logging.getLogger(__name__)


class RerankerService:
    """Cross-encoder reranker using ms-marco model via ONNX Runtime."""

    MODEL_NAME = 'Xenova/ms-marco-MiniLM-L-6-v2'

    def __init__(self) -> None:
        """Initialize the reranker model, self-healing on cache corruption."""
        try:
            self._encoder = TextCrossEncoder(model_name=self.MODEL_NAME)
        except onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile:
            # TODO: remove this workaround once https://github.com/qdrant/fastembed/issues/569 lands auto-recovery.
            logger.exception('Reranker cache corrupt — wiping and re-downloading')
            _wipe_model_cache(self.MODEL_NAME)
            self._encoder = TextCrossEncoder(model_name=self.MODEL_NAME)

    async def rerank(
        self,
        query: str,
        result: SearchResult,
        top_k: int | None = None,
    ) -> SearchResult:
        """Rerank search results using cross-encoder.

        Args:
            query: Original search query.
            result: Search results from hybrid search.
            top_k: Return only top k results after reranking (None = all).

        Returns:
            SearchResult with hits reordered by cross-encoder scores.
        """
        if not result.hits:
            return result

        documents = [hit.text for hit in result.hits]

        # fastembed rerank returns scores in document order (Iterable[float])
        scores = await asyncio.to_thread(
            lambda: list(self._encoder.rerank(query, documents)),
        )

        # Pair each hit with its reranker score, sort descending
        scored_hits = sorted(
            zip(scores, result.hits, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )

        reranked_hits: list[SearchHit] = [hit.model_copy(update={'score': score}) for score, hit in scored_hits]

        if top_k is not None:
            reranked_hits = reranked_hits[:top_k]

        return SearchResult(hits=reranked_hits, total=result.total)


def _wipe_model_cache(model_name: str) -> None:
    """Delete the fastembed cache dir for a model so the next load re-downloads.

    Workaround for a known upstream issue where fastembed detects but does not
    recover from a corrupted cache. The most common trigger on macOS is the
    per-user TMPDIR reaper (``dirhelper``) wiping ``/var/folders/*/T/``'s blob
    files while leaving the HuggingFace-style symlinks dangling; interrupted
    downloads produce the same wedged state on any platform.

    fastembed emits "Local file sizes do not match the metadata" and proceeds
    anyway, so ONNX Runtime raises ``NoSuchFile`` on load. Upstream's
    recommendation is to pass a persistent ``cache_dir``, which this service
    deliberately does not do — it would fork the per-user cache shared with any
    other fastembed tool on the same account. Users who want a persistent
    location can set ``FASTEMBED_CACHE_PATH`` in their environment; fastembed
    honors it (see ``fastembed/common/utils.py:define_cache_dir``).

    Upstream tracking:

    - https://github.com/qdrant/fastembed/issues/569 — default tmp-cache reaped
    - https://github.com/qdrant/fastembed/issues/88  — interrupted downloads corrupt cache
    - https://github.com/qdrant/fastembed/issues/127 — empty cache dir / missing model.onnx

    Cache layout follows HuggingFace's convention:
    ``{cache_root}/models--{org}--{name}/{blobs,snapshots,refs}``. Removing the
    whole model subdir clears both the blobs and the dangling symlinks so the
    next ``TextCrossEncoder`` init starts clean.
    """
    cache_root = os.environ.get('FASTEMBED_CACHE_PATH') or os.path.join(tempfile.gettempdir(), 'fastembed_cache')
    # HF convention: 'org/name' → 'models--org--name'
    model_dir = Path(cache_root) / f'models--{model_name.replace("/", "--")}'
    if model_dir.exists():
        shutil.rmtree(model_dir)
        logger.info('Removed corrupt cache dir: %s', model_dir)
