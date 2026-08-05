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
    #: backend tag persisted in the index so query-time reconstructs the SAME encoder (see open_store).
    backend = "bge"
    #: default cosine floor for the dense leg — the noise threshold BELOW which a hit is discarded. This
    #: is PER-ENCODER: bge sims are well-separated (~0.85 on-topic), so 0.4 is a real floor. A single
    #: global 0.4 silently kills encoders whose sim distribution is compressed (see NemotronEmbedder) —
    #: their vector leg returns nothing and "hybrid" degenerates to BM25-only, discarding the paid-for
    #: encoder. Stores resolve threshold=None to this value.
    sim_floor = 0.4

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

    backend = "nemotron"
    #: Nemotron-3-Embed's cosine sims are COMPRESSED vs bge (measured ~0.28–0.54 on-topic, not ~0.85),
    #: so the bge-calibrated 0.4 floor throws the gold chunk (which it ranks #1) out. A low floor keeps
    #: the top ranks; RRF/rerank order them. Verified: at 0.4 the Nemotron vector leg returns ~nothing.
    sim_floor = 0.1

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


class NemoRetrieverEmbedder:
    """NVIDIA **NeMo Retriever** embedding NIM (e.g. ``nvidia/llama-3.2-nv-embedqa``) over its
    OpenAI-compatible ``/v1/embeddings`` endpoint. Drop-in for :class:`Embedder` — exposes
    ``encode`` + ``dim`` — so a :class:`Store` built with it embeds both documents and queries
    through the NIM.

    This is a *capability behind the existing RAG contract*, not a replacement: ReDevOps RAG keeps
    owning chunk identity, provenance and ranking evidence; the NIM only supplies vectors. Serve it
    on GPU (``docker run … nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2``); the container exposes
    ``/v1/embeddings``.

    Like the other NVIDIA retrieval embedders the embedqa NIM is **asymmetric**: it takes an
    ``input_type`` (``passage`` for documents, ``query`` for queries) rather than an instruction
    prefix. ``encode`` embeds passages; :meth:`encode_queries` flips ``input_type`` to ``query``.
    """

    backend = "nemo-retriever"
    #: embedqa's cosine sims run compressed vs bge (like Nemotron), so the bge-calibrated 0.4 floor
    #: would discard the top-ranked chunk. A low floor keeps the top ranks; RRF/rerank order them.
    sim_floor = 0.1

    def __init__(self, url: str | None = None, model: str | None = None, dim: int | None = None,
                 api_key: str | None = None, batch: int = 32, max_chars: int = 24000,
                 timeout: float = 180.0):
        self.url = (url or os.environ.get("REDEVOPS_RAG_NEMO_URL")
                    or "http://127.0.0.1:8014/v1/embeddings")
        self.model = model or os.environ.get("REDEVOPS_RAG_NEMO_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
        self.dim = int(dim or os.environ.get("REDEVOPS_RAG_NEMO_DIM", "2048"))
        self.api_key = api_key if api_key is not None else os.environ.get("REDEVOPS_RAG_NEMO_API_KEY", "")
        self.batch, self.max_chars, self.timeout = batch, max_chars, timeout

    def _post(self, inputs: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(inputs), self.batch):
            chunk = [str(t)[: self.max_chars] for t in inputs[i:i + self.batch]]
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            body = {"model": self.model, "input": chunk, "input_type": input_type}
            req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
            data = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            out += [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        return out

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        return self._post(list(texts), "passage")

    def encode_queries(self, queries: Iterable[str]) -> list[list[float]]:
        """Embed queries with ``input_type="query"`` (embedqa's asymmetric query side)."""
        return self._post([str(q) for q in queries], "query")


def make_embedder(backend: str | None = None, **kw):
    """Return the embedder for ``backend`` (env ``REDEVOPS_RAG_EMBED_BACKEND``; default ``bge``).

    ``bge`` → :class:`Embedder` · ``nemo``/``nemo-retriever`` → :class:`NemoRetrieverEmbedder` ·
    ``nemotron`` → :class:`NemotronEmbedder` · ``reasonir`` →
    :class:`~redevops_rag.temporal.ReasonIREmbedder` · ``colpali``/``colqwen`` →
    :class:`~redevops_rag.multimodal.ColVisionEmbedder` (the doc-visual arm). One factory so the
    bench and the CR adapters select an encoder from the environment without branching at every
    call site. To route by corpus instead of a fixed backend, see :func:`encoder_for`."""
    backend = (backend or os.environ.get("REDEVOPS_RAG_EMBED_BACKEND", "bge")).strip().lower()
    if backend in ("nemo", "nemo-retriever"):
        return NemoRetrieverEmbedder(**kw)
    if backend in ("nemotron", "nemotron-embed"):
        return NemotronEmbedder(**kw)
    if backend == "reasonir":
        from .temporal import ReasonIREmbedder
        return ReasonIREmbedder(**kw)
    if backend in ("colpali", "colqwen", "colvision", "colbert-vision"):
        from .multimodal import ColVisionEmbedder
        return ColVisionEmbedder(backend=backend, **kw)
    return Embedder(**kw)


# ── corpus → encoder routing ────────────────────────────────────────────────────────────────
# The cross-dataset A/B (TEMPO English vs Russian nutrition) showed NO universal best encoder:
# cheap bge wins English, Nemotron-Embed wins Russian/domain. So the encoder is routed on the
# *corpus* (language/domain) — a STATIC per-index binding chosen at registration time — not per
# query (that's the retriever's bandit) and not per mission (that's model competence). Switching
# the encoder means re-embedding the whole corpus, which is exactly why it can't be an online arm.
#
# Rule: English text → bge (cheap, robust, DIVER-friendly); any other KNOWN language →
# Nemotron-Embed (multilingual/domain); page-image / doc-visual corpora → the colpali arm
# (text-dense scans where generic CLIP underperforms); unknown language + no domain signal →
# the cheap bge default (matches make_embedder). ENCODER_ROUTES lets a deployment pin specific
# (lang, domain) pairs, checked before the rule; keys use "*" as a wildcard.
ENCODER_ROUTES: dict[tuple[str, str], str] = {}
_MULTIMODAL_DOMAINS = {"multimodal", "visual", "doc-visual", "docvisual", "scanned", "pdf-image"}


def encoder_for(lang: str | None = None, domain: str | None = None) -> str:
    """Resolve the encoder backend for a corpus's ``lang`` (ISO-639-1, e.g. ``en``, ``ru``) and
    ``domain`` (free tag, e.g. ``nutrition``, ``multimodal``).

    Deployment overrides in ``ENCODER_ROUTES`` win first (exact (lang, domain) → (lang, ``*``) →
    (``*``, domain)). Otherwise: a doc-visual ``domain`` → the ``colpali`` arm regardless of
    language; English → ``bge``; any other known language → ``nemotron``; unknown language with no
    domain signal → the cheap ``bge`` default.

    This is the retrieval-side analog of answer-side competence routing: route the encoder on the
    corpus, the retriever on the regime, the model on measured competence."""
    lang = (lang or "").strip().lower()
    domain = (domain or "").strip().lower()
    for key in ((lang or "*", domain or "*"), (lang or "*", "*"), ("*", domain or "*")):
        if key in ENCODER_ROUTES:
            return ENCODER_ROUTES[key]
    if domain in _MULTIMODAL_DOMAINS:
        return "colpali"
    if lang == "en":
        return "bge"
    if lang:
        return "nemotron"
    return "bge"


def make_embedder_for(lang: str | None = None, domain: str | None = None, **kw):
    """Convenience: :func:`encoder_for` then :func:`make_embedder` — build the encoder a corpus's
    language/domain routes to. The returned embedder's ``.backend`` is what to persist in the index
    (see :meth:`redevops_rag.store.Store.set_meta`) so query-time reconstructs the same encoder."""
    return make_embedder(encoder_for(lang, domain), **kw)
