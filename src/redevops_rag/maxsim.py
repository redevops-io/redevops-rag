"""Late-interaction (MaxSim) retrieval — the scoring half of the ColPali / ColQwen doc-visual arm.

Single-vector cosine (the DuckDB/pg :class:`Store`) collapses a page to ONE vector; ColPali/ColQwen
keep a **multi-vector** representation (one vector per image patch / per token) and score a query
against a document by **MaxSim** (ColBERT, arXiv:2004.12832): for each query vector take its
best-matching document vector, then sum. This recovers the text-dense pages a single pooled vector
loses — the reason generic-CLIP-pooled retrieval underperformed text on the Russian nutrition scans.

    score(Q, D) = Σ_i  max_j  (q_i · d_j)          # Q:(nq,dim)  D:(nd,dim), L2-normalized rows

:class:`MaxSimStore` holds per-document multi-vectors and ranks them for a query multi-vector. Pair
it with :class:`redevops_rag.multimodal.ColVisionEmbedder` — ``encode_multivector`` for page images
(documents) and ``encode_multivector(..., images=False)`` for text queries. This is the follow-up to
the pooled first cut: pooled numbers were a floor; MaxSim is the real late-interaction score.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

MultiVector = Sequence[Sequence[float]]   # (n_vectors, dim)


def _normalize(mat):
    import numpy as np
    a = np.asarray(mat, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


def maxsim_score(query_mv: MultiVector, doc_mv: MultiVector) -> float:
    """ColBERT MaxSim between two multi-vectors (rows L2-normalized → dot = cosine). 0 if either
    side is empty."""
    import numpy as np
    q = _normalize(query_mv)
    d = _normalize(doc_mv)
    if q.size == 0 or d.size == 0:
        return 0.0
    sim = q @ d.T                      # (nq, nd)
    return float(sim.max(axis=1).sum())


class MaxSimStore:
    """In-memory multi-vector store with ColBERT-style MaxSim ranking.

    ``embedder`` (optional) is a multi-vector encoder — :class:`ColVisionEmbedder` or anything with
    ``encode_multivector(items, images=bool)``. It lets :meth:`add` embed raw page images and
    :meth:`search` embed a raw text query; pass pre-computed multi-vectors to skip it.

    Not a drop-in for :class:`Store` (single-vector + BM25); a distinct late-interaction retriever.
    Results carry a ``maxsim_score`` and ``source_type='maxsim'`` so downstream code can tell them
    apart from vector/bm25 hits."""

    def __init__(self, embedder=None):
        self.embedder = embedder
        self._meta: list[dict] = []      # per-doc {chunk_id, document_id, filename, text, metadata}
        self._mv: list[Any] = []         # per-doc np.ndarray (n_vec, dim), L2-normalized at add time

    # ── ingest ───────────────────────────────────────────────────────────────────────────────
    def add(self, docs: Iterable[dict], *, images: bool = True) -> int:
        """Add documents. Each ``doc`` carries identity (``document_id``/``id``, optional
        ``filename``/``text``/``metadata``) plus EITHER a precomputed ``multivector`` OR an
        ``image`` (path / data-URL / base64) embedded via the store's ``encode_multivector``.
        ``images`` selects the modality when embedding raw inputs."""
        import numpy as np
        docs = list(docs)
        to_embed, embed_at = [], []
        mvs: list[Any] = [None] * len(docs)
        for i, d in enumerate(docs):
            if d.get("multivector") is not None:
                mvs[i] = d["multivector"]
            else:
                src = d.get("image") if images else d.get("text")
                if src is None:
                    raise ValueError("doc needs 'multivector' or an 'image'/'text' to embed")
                to_embed.append(src)
                embed_at.append(i)
        if to_embed:
            if self.embedder is None:
                raise ValueError("MaxSimStore has no embedder; pass precomputed 'multivector' per doc")
            for j, mv in zip(embed_at, self.embedder.encode_multivector(to_embed, images=images)):
                mvs[j] = mv
        for d, mv in zip(docs, mvs):
            norm = _normalize(mv)
            if norm.size == 0:
                continue
            self._meta.append({
                "chunk_id": d.get("id") or d.get("chunk_id") or d.get("document_id"),
                "document_id": d.get("document_id") or d.get("id"),
                "filename": d.get("filename"),
                "text": d.get("text"),
                "metadata": d.get("metadata") or {},
            })
            self._mv.append(norm.astype(np.float32))
        return len(self._mv)

    #: pages are the document side.
    add_pages = add

    # ── search ───────────────────────────────────────────────────────────────────────────────
    def search(self, query, top_k: int = 10, *, images: bool = False) -> list[dict]:
        """Rank documents for ``query`` (a text string / raw input embedded via the store's encoder,
        or a precomputed query multi-vector) by MaxSim. ``images`` selects the query modality when a
        raw input is embedded (default text)."""
        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("MaxSimStore has no embedder; pass a precomputed query multi-vector")
            qmv = self.embedder.encode_multivector([query], images=images)[0]
        elif query and isinstance(query[0], (int, float)):
            qmv = [query]                                   # a single vector → 1×dim multi-vector
        else:
            qmv = query                                     # already a multi-vector
        qn = _normalize(qmv)
        scored = [(maxsim_score(qn, dmv), i) for i, dmv in enumerate(self._mv)]
        scored.sort(key=lambda s: s[0], reverse=True)
        out = []
        for score, i in scored[:top_k]:
            row = dict(self._meta[i])
            row["maxsim_score"] = float(score)
            row["source_type"] = "maxsim"
            out.append(row)
        return out

    def count(self) -> int:
        return len(self._mv)

    # ── persistence (ragged multi-vectors → one packed array + offsets) ───────────────────────
    def save(self, path: str) -> None:
        """Persist to a ``.npz`` (packed vectors + offsets + JSON meta). Reload with :meth:`load`."""
        import numpy as np
        if self._mv:
            packed = np.concatenate(self._mv, axis=0).astype(np.float32)
            lengths = np.asarray([m.shape[0] for m in self._mv], dtype=np.int64)
        else:
            packed = np.zeros((0, 0), dtype=np.float32)
            lengths = np.zeros((0,), dtype=np.int64)
        np.savez(path, packed=packed, lengths=lengths, meta=np.asarray(json.dumps(self._meta)))

    @classmethod
    def load(cls, path: str, embedder=None) -> "MaxSimStore":
        import numpy as np
        p = path if str(path).endswith(".npz") else f"{path}.npz"
        data = np.load(p, allow_pickle=False)
        store = cls(embedder=embedder)
        store._meta = json.loads(str(data["meta"]))
        packed, lengths = data["packed"], data["lengths"]
        off = 0
        for n in lengths.tolist():
            store._mv.append(packed[off:off + n].astype(np.float32))
            off += n
        return store
