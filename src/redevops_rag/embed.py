"""Sentence-Transformers embedder (normalized → cosine-ready). Mirrors rag-saas-platform's
shared-singleton model loading, but self-contained.

Two embedders share one duck-typed contract — ``encode(texts) -> list[list[float]]`` + ``.dim`` — so
either can back a :class:`~redevops_rag.store.Store`:

  * :class:`Embedder`         — the cheap CPU default (bge-small-en, 384-d).
  * :class:`NemotronEmbedder` — NVIDIA Nemotron-3-Embed-8B (4096-d, RTEB #1) over an OpenAI-compatible
    ``/v1/embeddings`` endpoint (NIM / vLLM on GPU). Opt-in — a stronger, heavier arm to benchmark
    against bge before deciding whether to make it a default.

Pick the backend with :func:`make_embedder` (env ``REDEVOPS_RAG_EMBED_BACKEND``) so the bench harness
and the Context Runtime adapters can swap encoders without touching call sites."""
from __future__ import annotations

import json
import os
import urllib.request
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


class NemotronEmbedder:
    """NVIDIA **Nemotron-3-Embed-8B** over an OpenAI-compatible ``/v1/embeddings`` endpoint (serve on
    GPU via NIM or vLLM). Drop-in for :class:`Embedder` — exposes ``encode`` + ``dim`` so a
    :class:`Store` built with it embeds *both* documents and queries via Nemotron.

    Opt-in, not a default: 8B / 4096-d is much heavier than bge-small (384-d) — bigger vectors, GPU
    serving — so it is wired behind a flag and meant to be *measured* against bge across datasets +
    retrieval methods first (see ``benchmarks/eval_embedders.py``).

    Serve it (example)::

        # NIM:  docker run … nvcr.io/nim/nvidia/nemotron-3-embed-8b  (exposes /v1/embeddings)
        # vLLM: vllm serve nvidia/Nemotron-3-Embed-8B-BF16 --task embed --served-model-name nemotron-embed

    Nemotron is a retrieval model that expects a task **instruction on queries** (documents get none).
    ``encode`` is symmetric (used for both sides, like :class:`Embedder`); pass a query through
    :meth:`encode_queries` when you want the instruction prefix applied.
    """

    #: NV/Nemotron retrieval instruction; prepended to queries only (documents are embedded raw).
    QUERY_INSTRUCTION = ("Instruct: Given a query, retrieve the passages that best answer it\nQuery: ")

    def __init__(self, url: str | None = None, model: str | None = None, dim: int | None = None,
                 api_key: str | None = None, batch: int = 32, max_chars: int = 24000,
                 timeout: float = 180.0):
        self.url = (url or os.environ.get("REDEVOPS_RAG_NEMOTRON_URL")
                    or "http://127.0.0.1:8013/v1/embeddings")
        self.model = model or os.environ.get("REDEVOPS_RAG_NEMOTRON_MODEL", "nemotron-embed")
        self.dim = int(dim or os.environ.get("REDEVOPS_RAG_NEMOTRON_DIM", "4096"))
        self.api_key = api_key if api_key is not None else os.environ.get("REDEVOPS_RAG_NEMOTRON_API_KEY", "")
        self.batch, self.max_chars, self.timeout = batch, max_chars, timeout

    def _post(self, inputs: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(inputs), self.batch):
            chunk = [str(t)[: self.max_chars] for t in inputs[i:i + self.batch]]
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                self.url, data=json.dumps({"model": self.model, "input": chunk}).encode(), headers=headers)
            data = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            out += [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        return out

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        return self._post(list(texts))

    def encode_queries(self, queries: Iterable[str]) -> list[list[float]]:
        """Embed queries WITH the retrieval instruction prefix (Nemotron's asymmetric query side)."""
        return self._post([self.QUERY_INSTRUCTION + str(q) for q in queries])


def make_embedder(backend: str | None = None, **kw):
    """Return the embedder for ``backend`` (env ``REDEVOPS_RAG_EMBED_BACKEND``; default ``bge``).

    ``bge`` → :class:`Embedder` · ``nemotron`` → :class:`NemotronEmbedder` · ``reasonir`` →
    :class:`~redevops_rag.temporal.ReasonIREmbedder`. One factory so the bench and the CR adapters
    select an encoder from the environment without branching at every call site."""
    backend = (backend or os.environ.get("REDEVOPS_RAG_EMBED_BACKEND", "bge")).strip().lower()
    if backend in ("nemotron", "nemotron-embed", "nemo"):
        return NemotronEmbedder(**kw)
    if backend == "reasonir":
        from .temporal import ReasonIREmbedder
        return ReasonIREmbedder(**kw)
    return Embedder(**kw)
