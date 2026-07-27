"""Document-visual retrieval — the ColPali / ColQwen arm for text-dense page images.

**Why this exists.** For scanned / PDF corpora we first tried generic CLIP over page-images
(PDF → page-images → CLIP + multilingual-CLIP). On text-dense Russian lecture pages it landed
at ~0.525 doc-level — *below* plain text retrieval. Generic CLIP embeds a page as a natural
image; it does not read the text laid out on it, so it is the wrong model for document pages.

The right model is a **document-visual late-interaction** encoder — ColPali (arXiv:2407.01449)
or ColQwen — which tiles the page into patches and produces a *multi-vector* representation
scored against the query tokens by MaxSim (ColBERT-style). It reads text-as-layout, so it
recovers the text-dense pages generic CLIP loses.

**Integration status (measure-first, like Nemotron was).** This client is a first, opt-in
integration so the arm can be benchmarked against text retrieval. The redevops-rag
:class:`~redevops_rag.store.Store` is single-vector cosine, so by default this **mean-pools** the
page's patch vectors to one vector to drop straight in. That pooling discards the late-interaction
signal that is the whole point of ColPali — so treat pooled numbers as a floor. Full MaxSim
scoring (a multi-vector store + a MaxSim retriever) is the follow-up; see ``encode_multivector``,
which returns the un-pooled patch vectors for a future late-interaction store.

Serve it (example)::

    # ColQwen / ColPali over vLLM or a small FastAPI shim exposing /v1/embeddings that accepts
    # {"input":[{"image": "<base64>"}...]} for pages and {"input":["query text"...]} for queries.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Iterable


class ColVisionEmbedder:
    """ColPali / ColQwen document-visual embedder over an HTTP ``/v1/embeddings`` endpoint.

    Duck-typed like :class:`redevops_rag.embed.Embedder` (``encode`` + ``.dim``) so it can back a
    :class:`~redevops_rag.store.Store`, plus:
      * ``encode`` / ``encode_image`` — embed page IMAGES (documents): base64 strings, ``data:``
        URLs, or filesystem paths to PNG/JPG page renders;
      * ``encode_queries`` — embed TEXT queries (the asymmetric query side);
      * ``encode_multivector`` — the un-pooled patch vectors, for a future MaxSim store.

    Opt-in, not a default. Single-vector pooling is a simplification of ColPali's late interaction
    (documented above) — wire it behind the ``colpali``/``colqwen`` backend and *measure* before
    trusting it for text-dense document corpora."""

    backend = "colpali"

    def __init__(self, url: str | None = None, model: str | None = None, dim: int | None = None,
                 api_key: str | None = None, batch: int = 8, timeout: float = 180.0,
                 backend: str | None = None):
        self.url = (url or os.environ.get("REDEVOPS_RAG_COLPALI_URL")
                    or "http://127.0.0.1:8014/v1/embeddings")
        self.model = model or os.environ.get("REDEVOPS_RAG_COLPALI_MODEL", backend or "colpali")
        self.dim = int(dim or os.environ.get("REDEVOPS_RAG_COLPALI_DIM", "128"))
        self.api_key = api_key if api_key is not None else os.environ.get("REDEVOPS_RAG_COLPALI_API_KEY", "")
        self.batch, self.timeout = batch, timeout
        if backend:
            self.backend = backend

    # ── HTTP ────────────────────────────────────────────────────────────────────────────────
    def _post(self, inputs: list[Any]) -> list[list[list[float]]]:
        """POST a batch, returning one MULTI-vector (list of patch/token vectors) per input."""
        out: list[list[list[float]]] = []
        for i in range(0, len(inputs), self.batch):
            chunk = inputs[i:i + self.batch]
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                self.url, data=json.dumps({"model": self.model, "input": chunk}).encode(), headers=headers)
            data = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            for d in sorted(data["data"], key=lambda x: x["index"]):
                emb = d["embedding"]
                # normalize to a list-of-vectors: a server may return a single vector (already
                # pooled) or a multi-vector (patch/token vectors) — carry both as multi-vector.
                out.append(emb if emb and isinstance(emb[0], list) else [emb])
        return out

    @staticmethod
    def _mean_pool(mv: list[list[float]]) -> list[float]:
        if not mv:
            return []
        n = len(mv)
        dim = len(mv[0])
        acc = [0.0] * dim
        for vec in mv:
            for j in range(dim):
                acc[j] += vec[j]
        return [x / n for x in acc]

    @staticmethod
    def _as_image_payload(img: Any) -> dict:
        """Coerce a page image (path / data-URL / raw base64) into the server's image input item."""
        if isinstance(img, (str, Path)) and Path(str(img)).exists():
            raw = Path(str(img)).read_bytes()
            b64 = base64.b64encode(raw).decode()
            return {"image": f"data:image/png;base64,{b64}"}
        return {"image": str(img)}   # already a data URL or base64 string

    # ── embedder contract ────────────────────────────────────────────────────────────────────
    def encode(self, images: Iterable[Any]) -> list[list[float]]:
        """Embed page IMAGES (documents) → one pooled vector each (drops into the cosine Store)."""
        mvs = self._post([self._as_image_payload(x) for x in images])
        return [self._mean_pool(mv) for mv in mvs]

    #: page images are the document side.
    encode_image = encode

    def encode_queries(self, queries: Iterable[str]) -> list[list[float]]:
        """Embed TEXT queries → one pooled vector each (asymmetric query side)."""
        mvs = self._post([str(q) for q in queries])
        return [self._mean_pool(mv) for mv in mvs]

    def encode_multivector(self, items: Iterable[Any], *, images: bool = True) -> list[list[list[float]]]:
        """Un-pooled patch/token vectors per item — the input a future MaxSim late-interaction
        store needs. ``images=True`` treats items as page images, else as text queries."""
        payload = [self._as_image_payload(x) for x in items] if images else [str(x) for x in items]
        return self._post(payload)
