"""Sentence-Transformers embedder (normalized → cosine-ready). Mirrors rag-saas-platform's
shared-singleton model loading, but self-contained."""
from __future__ import annotations

import os
from typing import Iterable

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, strong for retrieval, small enough for CPU


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model_name = model_name or os.environ.get("REDEVOPS_RAG_EMBED_MODEL", DEFAULT_MODEL)
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        embs = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in embs]
