"""Temporal / reasoning-intensive retrieval plugin — DIVER + ReasonIR, merged.

One plugin that owns both halves of reasoning-intensive temporal retrieval:

  • the *embedder* — ReasonIR-8B (arXiv:2504.20595), a bidirectional 8B reasoning
    retriever, served over an OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM,
    ``LlamaBidirectionalModel`` + mean pooling); and
  • the *retrieval pipeline* — DIVER (arXiv:2508.07995): LLM query expansion → hybrid
    retrieve (union over sub-queries) → LLM listwise rerank (:func:`diver_search`).

Together they form the strongest single retriever we measured (TEMPO/workplace
NDCG@10 0.197 BM25 → 0.33 hybrid → ~0.43 DIVER; ReasonIR beats bge on the dense/hybrid
stage). Exposed as one arm so Context Runtime's bandit can *select* it per query
against the cheaper ``hybrid_search`` configs — see
``context_runtime.integrations.redevops_rag`` for the tuner that does the selecting.

Usage::

    emb   = ReasonIREmbedder("http://host:8012/v1/embeddings")
    store = Store(emb, "corpus.duckdb")           # docs + queries embed via ReasonIR
    plug  = TemporalReasoningRetriever(reason_llm)  # reason_llm(system, user) -> str
    hits  = plug.search(store, query, limit=8)
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

from .retrieve import diver_search, hybrid_search


class ReasonIREmbedder:
    """ReasonIR-8B embeddings over an OpenAI-compatible endpoint (vLLM on GPU).

    Drop-in for :class:`redevops_rag.embed.Embedder` — exposes ``encode`` + ``dim`` so a
    :class:`Store` built with it embeds *both* documents and queries via ReasonIR. Serve
    ReasonIR bidirectionally (it underperforms a small encoder if served causal)::

        vllm serve /reasonir-8b --served-model-name reasonir --trust-remote-code \\
          --hf-overrides '{"architectures":["LlamaBidirectionalModel"],"pooling":"avg"}'
    """

    def __init__(self, url: str = "http://127.0.0.1:8012/v1/embeddings",
                 model: str = "reasonir", dim: int = 4096, batch: int = 48,
                 max_chars: int = 6000, timeout: float = 180.0):
        self.url, self.model, self.dim = url, model, dim
        self.batch, self.max_chars, self.timeout = batch, max_chars, timeout

    def encode(self, texts) -> list[list[float]]:
        texts, out = list(texts), []
        for i in range(0, len(texts), self.batch):
            chunk = [str(t)[: self.max_chars] for t in texts[i:i + self.batch]]
            req = urllib.request.Request(
                self.url, data=json.dumps({"model": self.model, "input": chunk}).encode(),
                headers={"Content-Type": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            out += [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        return out


class TemporalReasoningRetriever:
    """The merged DIVER + ReasonIR temporal retrieval plugin.

    ``reason_llm(system, user) -> str`` is the reasoning model used for DIVER's query
    expansion and listwise rerank. ``search`` runs the full DIVER pipeline over a store
    that should be ReasonIR-embedded (:class:`ReasonIREmbedder`). With ``reason_llm``
    None it degrades to plain :func:`hybrid_search`, so it is a safe bandit arm even on
    cold start / no-LLM budget.
    """

    name = "temporal_reasoning"

    def __init__(self, reason_llm: Callable[[str, str], str] | None = None,
                 pool: int = 25, n_subqueries: int = 3):
        self.reason_llm = reason_llm
        self.pool = pool
        self.n_subqueries = n_subqueries

    def search(self, store, query: str, limit: int = 8, *, reranker=None,
               document_ids: list | None = None,
               recency_half_life_days: float = 0.0) -> list[dict[str, Any]]:
        if self.reason_llm is None:
            return hybrid_search(store, query, limit=limit, pool=self.pool,
                                 document_ids=document_ids, reranker=reranker,
                                 recency_half_life_days=recency_half_life_days)
        return diver_search(store, query, self.reason_llm, limit=limit, pool=self.pool,
                            n_subqueries=self.n_subqueries, reranker=reranker,
                            document_ids=document_ids,
                            recency_half_life_days=recency_half_life_days)

    # callable form so it slots straight into a bandit arm's retrieve hook
    __call__ = search
