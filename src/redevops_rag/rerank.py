"""Optional cross-encoder rerank stage (BAAI/bge-reranker-v2-m3), matching the final
rerank in rag-saas-platform. Needs the ``[rerank]`` extra (FlagEmbedding + torch)."""
from __future__ import annotations

import os

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


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
