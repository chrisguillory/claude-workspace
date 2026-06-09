---
area: document-search
title: Gemini Batch API path for bulk embedding
---

**What.** A bulk-embedding path that submits chunks to Gemini's asynchronous **Batch API** (`:batchEmbedContents` job inputs) instead of the synchronous `embed_content` endpoint — for latency-tolerant jobs: full reindex, corpus migration, provider bake-offs.

**Why.** It sidesteps the real-time rate ceiling entirely — the Batch API draws from a *separate* enqueued-token pool that's immune to the per-minute 429s the sync path hits (gmail Tier-2 caps at 5,000 embeddings/min, so a full ~12.2M-vector mesh reindex is ~41h on the sync path). It's also **50% cheaper** ($0.075 vs $0.15 per 1M tokens). The sync path stays for interactive search + incremental indexing; this is for the rare-but-real bulk work.

**Sketch.** A sibling to the sync `clients/gemini.py` (e.g. `clients/gemini_batch.py`): submit a batch job, poll for completion, ingest results into Qdrant. Touches the `EmbeddingClient` protocol (`clients/protocols.py`), the client factory (`clients/__init__.py`), and the indexing service (`services/embedding.py`). Latency-tolerant by design — not for query-time embeds.

<sub>Claude Code session <code>d14ec85e-6891-4510-bae1-9a38d8e02167</code></sub>
