"""Temporal / reasoning-intensive retrieval — DIVER (the recommended default) plus two
opt-in, non-default building blocks (the combined DIVER+ReasonIR retriever and the
ReasonIR embedder).

The recommended temporal reasoning retriever is **DIVER** (:func:`redevops_rag.diver_search`)
— LLM query expansion → hybrid retrieve (union over sub-queries) → LLM listwise rerank —
run over a *cheap* embedder. It is embedder-agnostic and is the strongest, cheapest option
we measured. Neither class below is a redevops-rag default; import them explicitly.

Ablation (TEMPO/workplace, matched limit=10/pool=25, NDCG@10):

    sim/BM25 ............... 0.197
    hybrid · bge .......... 0.297      hybrid · ReasonIR ... 0.325   (+ReasonIR helps hybrid)
    DIVER · bge (solo) .... 0.448      DIVER · ReasonIR .... 0.337   (+ReasonIR HURTS DIVER)

So: **DIVER over a cheap encoder (bge) is the default** — best and cheapest. ReasonIR lifts
*plain hybrid* (+0.03) but *degrades* DIVER (−0.11), because DIVER feeds the embedder
already-expanded sub-queries rather than the reasoning-heavy original ReasonIR is tuned for.

:class:`TemporalReasoningRetriever` — DIVER over any store, exposed as a single Context
Runtime bandit arm (see ``context_runtime.integrations.redevops_rag``). Pair it with a cheap
embedder; do NOT pair it with ReasonIR (that's the losing combo above).

:class:`ReasonIREmbedder` — ReasonIR-8B (arXiv:2504.20595) over a vLLM ``/v1/embeddings``
endpoint. Opt-in embedder for the *hybrid* arm only, where its reasoning-tuned vectors help;
not a default, and not for the DIVER path.

Usage::

    from redevops_rag import Embedder, Store, TemporalReasoningRetriever
    store = Store(Embedder(), "corpus.duckdb")        # cheap bge — the recommended pairing
    plug  = TemporalReasoningRetriever(reason_llm)    # reason_llm(system, user) -> str
    hits  = plug.search(store, query, limit=8)        # DIVER over bge = the default temporal retriever
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

from .retrieve import diver_search, hybrid_search


class ReasonIREmbedder:
    """ReasonIR-8B embeddings over an OpenAI-compatible endpoint (vLLM on GPU). **Opt-in,
    non-default** — best for the *plain hybrid* arm (helps ~+0.03 NDCG@10 on TEMPO); do NOT
    use it under DIVER, where it degrades results (~−0.11) vs a cheap encoder.

    Drop-in for :class:`redevops_rag.embed.Embedder` — exposes ``encode`` + ``dim`` so a
    :class:`Store` built with it embeds *both* documents and queries via ReasonIR. Serve
    ReasonIR bidirectionally (it underperforms a small encoder if served causal)::

        vllm serve /reasonir-8b --served-model-name reasonir --trust-remote-code \\
          --hf-overrides '{"architectures":["LlamaBidirectionalModel"],"pooling":"avg"}'
    """

    backend = "reasonir"

    #: ReasonIR is instruction-tuned (arXiv:2504.20595): queries carry a task instruction,
    #: documents don't. Applied on the query side only, via ``encode_queries``.
    QUERY_INSTRUCTION = ("Instruct: Given a query, retrieve the passages that best answer it\nQuery: ")

    def __init__(self, url: str | None = None,
                 model: str = "reasonir", dim: int = 4096, batch: int = 48,
                 max_chars: int = 6000, timeout: float = 180.0):
        # URL resolves from REDEVOPS_RAG_REASONIR_URL (mirrors NemotronEmbedder), so make_embedder
        # picks up an off-box endpoint from the env instead of the localhost default.
        import os
        url = url or os.environ.get("REDEVOPS_RAG_REASONIR_URL", "http://127.0.0.1:8012/v1/embeddings")
        self.url, self.model, self.dim = url, model, dim
        self.batch, self.max_chars, self.timeout = batch, max_chars, timeout

    def _post(self, texts) -> list[list[float]]:
        texts, out = list(texts), []
        for i in range(0, len(texts), self.batch):
            chunk = [str(t)[: self.max_chars] for t in texts[i:i + self.batch]]
            req = urllib.request.Request(
                self.url, data=json.dumps({"model": self.model, "input": chunk}).encode(),
                headers={"Content-Type": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            out += [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        return out

    def encode(self, texts) -> list[list[float]]:
        return self._post(texts)

    def encode_queries(self, queries) -> list[list[float]]:
        """Embed queries WITH the reasoning instruction prefix (documents stay raw). DIVER sends
        the reasoning-heavy *original* query here; sub-query fragments go through plain ``encode``."""
        return self._post([self.QUERY_INSTRUCTION + str(q) for q in queries])


class TemporalReasoningRetriever:
    """DIVER temporal reasoning retriever, exposed as one Context Runtime bandit arm.

    ``reason_llm(system, user) -> str`` is the reasoning model used for DIVER's query
    expansion and listwise rerank. ``search`` runs the full DIVER pipeline over the given
    store. With ``reason_llm`` None it degrades to plain :func:`hybrid_search`, so it is a
    safe bandit arm on cold start / no-LLM budget.

    Recommended pairing: a **cheap embedder** (:class:`redevops_rag.embed.Embedder`, bge) —
    that's DIVER-solo, the best/cheapest config in the ablation. Do NOT build the store with
    :class:`ReasonIREmbedder`: the combined DIVER+ReasonIR is the losing arm (0.337 vs 0.448).
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

    def insert(self, store, docs, *, embedder=None, reindex: bool = True) -> int:
        """Incrementally add documents to the store DIVER retrieves over — NO index rebuild. DIVER is
        index-free (query-time expand → retrieve → rerank), so a new document is simply embedded and
        upserted, and the very next ``search`` sees it. This is the streaming/live-corpus update path
        (parity with the graph engines' incremental ``insert``): a corpus that grows — new sessions,
        edited docs — extends in place instead of rebuilding. ``docs`` = list of ``{text, document_id?,
        metadata?}``; returns the number of chunks added. Uses the store's own embedder by default so the
        new documents live in the SAME vector space the queries are encoded in (see encoder routing)."""
        emb = embedder or store.embedder
        chunks = [{"document_id": d.get("document_id") or d.get("chunk_id"),
                   "text": d["text"], "metadata": d.get("metadata") or {}} for d in docs]
        if not chunks:
            return 0
        for c, e in zip(chunks, emb.encode([c["text"] for c in chunks])):
            c["embedding"] = e
        return store.add_chunks(chunks, reindex=reindex)

    # callable form so it slots straight into a bandit arm's retrieve hook
    __call__ = search
