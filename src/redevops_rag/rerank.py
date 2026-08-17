"""Optional cross-encoder rerank stage (BAAI/bge-reranker-v2-m3), matching the final
rerank in rag-saas-platform. Needs the ``[rerank]`` extra (FlagEmbedding + torch).

Two rerankers share one duck-typed contract — ``rerank(query, candidates) -> list[dict]`` (mutates
``rerank_score`` in, sorts best-first) — so either can back the final rerank stage in
:func:`~redevops_rag.retrieve.hybrid_search`:

  * :class:`Reranker`              — the local cross-encoder (bge-reranker-v2-m3; pulls torch).
  * :class:`NemoRetrieverReranker` — NVIDIA **NeMo Retriever** reranking NIM (e.g.
    ``nvidia/llama-3.2-nv-rerankqa``) over its OpenAI-compatible ``/v1/ranking`` endpoint. A
    capability behind the RAG contract — ReDevOps RAG still owns the ranking *evidence*, so the
    NIM reranker records the pre-rank and post-rank order (see :meth:`rerank_with_evidence`)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _cand_id(c: dict) -> str:
    """Stable identity for a candidate row across the rerank boundary (chunk id first)."""
    return str(c.get("chunk_id") or c.get("id") or c.get("document_id")
               or (c.get("filename"), c.get("chunk_index")))


def _digest(obj) -> str:
    """SHA-256 over a canonical JSON encoding — request/response provenance for the envelope."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Reranker:
    def __init__(self, model_name: str | None = None, use_fp16: bool = True):
        from FlagEmbedding import FlagReranker  # lazy: pulls torch

        self.model_name = model_name or os.environ.get("REDEVOPS_RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self.model = FlagReranker(self.model_name, use_fp16=use_fp16)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        pairs = [[query, c.get("text") or ""] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
        return candidates


class NemoRetrieverReranker:
    """NVIDIA **NeMo Retriever** reranking NIM (e.g. ``nvidia/llama-3.2-nv-rerankqa``) over its
    OpenAI-compatible ``/v1/ranking`` endpoint. Drop-in for :class:`Reranker` — same
    ``rerank(query, candidates) -> list[dict]`` contract — so it can back the final rerank stage in
    :func:`~redevops_rag.retrieve.hybrid_search` without touching call sites.

    A capability behind the RAG contract, not a replacement: ReDevOps RAG owns the ranking
    *evidence*, so every rerank records the pre-rank order (as fused) and the post-rank order (as
    the NIM returned it). :meth:`rerank` stays interface-compatible and stashes that on
    :attr:`last_evidence`; :meth:`rerank_with_evidence` returns the full ``provider-capability``
    result envelope (``provider``/``service_kind``/``model_id``/digests/``latency_ms``/``evidence``).

    Serve it on GPU (``docker run … nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2``); the container
    exposes ``/v1/ranking`` taking ``{"model", "query": {"text"}, "passages": [{"text"}, …]}`` and
    returning ``{"rankings": [{"index", "logit"}, …]}`` sorted best-first.
    """

    #: provider identity for the cross-repo ``provider-capability/v1`` result envelope.
    provider = "nvidia-nemo-retriever"
    service_kind = "nemo-retriever"

    def __init__(self, url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0):
        self.url = (url or os.environ.get("REDEVOPS_RAG_NEMO_RERANK_URL")
                    or "http://127.0.0.1:8015/v1/ranking")
        self.model = model or os.environ.get("REDEVOPS_RAG_NEMO_RERANK_MODEL",
                                             "nvidia/llama-3.2-nv-rerankqa-1b-v2")
        self.api_key = api_key if api_key is not None else os.environ.get("REDEVOPS_RAG_NEMO_API_KEY", "")
        self.timeout = timeout
        #: evidence from the most recent :meth:`rerank` (``None`` before the first call).
        self.last_evidence: dict | None = None

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        return json.load(urllib.request.urlopen(req, timeout=self.timeout))

    def rerank_with_evidence(self, query: str, candidates: list[dict]) -> dict:
        """Rerank ``candidates`` and return the full ``provider-capability`` result envelope.

        ``evidence`` records ``pre_rank_order`` (candidate ids as fused, before the NIM) and
        ``post_rank_order`` (the same ids after the NIM's ranking) plus the per-candidate
        ``rerank_score`` — the audit trail that ReDevOps RAG, not the provider, owns. The ranked
        candidate list is under ``ranked`` (the same objects, mutated with ``rerank_score`` and
        sorted best-first).
        """
        pre_order = [_cand_id(c) for c in candidates]
        if not candidates:
            self.last_evidence = {"pre_rank_order": [], "post_rank_order": []}
            return self._envelope(query, candidates, self.last_evidence, request={}, response={})
        passages = [{"text": c.get("text") or ""} for c in candidates]
        body = {"model": self.model, "query": {"text": query}, "passages": passages}
        t0 = time.time()
        resp = self._post(body)
        latency_ms = (time.time() - t0) * 1000.0
        # NIM returns rankings best-first; each carries the passage index + a relevance logit.
        rankings = resp.get("rankings") or []
        ranked: list[dict] = []
        seen: set[int] = set()
        for r in rankings:
            i = r.get("index")
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                c = candidates[i]
                c["rerank_score"] = float(r.get("logit", 0.0))
                ranked.append(c)
        for i, c in enumerate(candidates):  # stable fallback: keep any the NIM didn't return
            if i not in seen:
                c.setdefault("rerank_score", 0.0)
                ranked.append(c)
        evidence = {
            "pre_rank_order": pre_order,
            "post_rank_order": [_cand_id(c) for c in ranked],
            "scores": {_cand_id(c): c.get("rerank_score", 0.0) for c in ranked},
        }
        self.last_evidence = evidence
        env = self._envelope(query, ranked, evidence, request=body, response=resp)
        env["latency_ms"] = latency_ms
        return env

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Interface-compatible with :meth:`Reranker.rerank`: returns the candidates ranked
        best-first (mutated with ``rerank_score``); the ranking evidence is stashed on
        :attr:`last_evidence`."""
        if not candidates:
            self.last_evidence = {"pre_rank_order": [], "post_rank_order": []}
            return candidates
        return self.rerank_with_evidence(query, candidates)["ranked"]

    def _envelope(self, query: str, ranked: list[dict], evidence: dict,
                  request: dict, response: dict) -> dict:
        return {
            "provider": self.provider,
            "service_kind": self.service_kind,
            "model_id": self.model,
            "request_digest": _digest(request),
            "response_digest": _digest(response),
            "cost_usd": 0.0,       # self-hosted NIM: no per-call price signal
            "latency_ms": 0.0,
            "evidence": evidence,
            "ranked": ranked,
        }
