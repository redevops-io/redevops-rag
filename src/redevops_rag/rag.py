"""Top-level RAG facade: index a folder, hybrid-search it, optionally synthesize an answer
against any OpenAI-compatible LLM (local MLX/llama.cpp, OpenAI, or Anthropic via a gateway)."""
from __future__ import annotations

import os
from typing import Any

from .embed import Embedder
from .ingest import ingest as _ingest
from .retrieve import hybrid_search
from .store import Store

_ANSWER_SYSTEM = (
    "Answer the question using ONLY the provided context. Cite sources inline like [1], [2]. "
    "If the context is insufficient, say so plainly — do not invent facts."
)


class RAG:
    def __init__(self, db_path: str = "./redevops_rag.duckdb", embed_model: str | None = None,
                 use_reranker: bool = False, rerank_model: str | None = None):
        self.embedder = Embedder(embed_model)
        self.store = Store(self.embedder, db_path)
        self.reranker = None
        if use_reranker:
            from .rerank import Reranker
            self.reranker = Reranker(rerank_model)

    def index(self, path: str, size: int = 1000, overlap: int = 150) -> dict:
        result = _ingest(self.store, self.embedder, path, size=size, overlap=overlap)
        self.store.reindex_fts()
        return result

    def search(self, query: str, k: int = 8, **kw) -> list[dict[str, Any]]:
        return hybrid_search(self.store, query, limit=k, reranker=self.reranker, **kw)

    def ask(self, query: str, k: int = 8, model: str | None = None, base_url: str | None = None,
            api_key: str | None = None, system: str | None = None) -> dict[str, Any]:
        hits = self.search(query, k=k)
        context = "\n\n".join(f"[{i + 1}] {h['filename']}: {h['text']}" for i, h in enumerate(hits))
        base_url = base_url or os.environ.get("REDEVOPS_RAG_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("REDEVOPS_RAG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not (base_url or api_key):
            return {"answer": None, "context": context, "sources": hits}
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
            model = model or os.environ.get("REDEVOPS_RAG_LLM_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model, temperature=0.1,
                messages=[
                    {"role": "system", "content": system or _ANSWER_SYSTEM},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
            )
            return {"answer": resp.choices[0].message.content, "context": context, "sources": hits}
        except Exception as e:
            return {"answer": None, "error": str(e), "context": context, "sources": hits}

    def close(self) -> None:
        self.store.close()
