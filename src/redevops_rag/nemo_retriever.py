"""NeMo Retriever **extraction** capability + the thin ``extract → embed → rerank`` seam.

NVIDIA's NeMo Retriever is a suite of retrieval microservices (NIMs) behind OpenAI-compatible HTTP:
an embedding NIM (:class:`~redevops_rag.embed.NemoRetrieverEmbedder`), a reranking NIM
(:class:`~redevops_rag.rerank.NemoRetrieverReranker`), and an **extraction** NIM (nemoretriever-parse:
page-elements / tables / charts out of PDFs and office docs). This module wires the extraction NIM
and stitches the three into one capability surface.

The contract stays ReDevOps RAG's, not the provider's. Provider-specific extraction **cannot bypass
content hashing**: every extracted segment is minted into a canonical :class:`ArtifactHandle` through
:func:`make_artifact_handle`, the single content-hashing path — so identical ingestion yields
identical, stable chunk identifiers regardless of which NIM produced the text. Content the extraction
NIM surfaces but this pipeline cannot yet embed (charts, raw images with no extracted text) is
**typed**, not silently dropped: it comes back as an :class:`UnsupportedArtifact` the caller can see,
count and route — never a hole in the index.

All HTTP is stdlib ``urllib`` (monkeypatchable, no heavy deps); tests run model-free and offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable

#: media types this pipeline can turn into embeddable text. Everything else is typed, not dropped.
SUPPORTED_MEDIA = {"text/plain", "text/markdown", "text/html", "application/x-table"}


def content_hash(text: str) -> str:
    """SHA-256 hex of the chunk content — the identity primitive extraction cannot route around."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_identity(document_ref: str, content: str) -> str:
    """Canonical, position-independent chunk id: ``{document_ref}::{content_hash[:16]}``.

    Derived purely from the document reference and the content hash, so re-extracting the same
    document — in any order, through any NIM — reproduces the exact same identifier."""
    return f"{document_ref}::{content_hash(content)[:16]}"


@dataclass(frozen=True)
class ArtifactHandle:
    """A canonical extracted chunk: stable :attr:`chunk_id` (content-hash derived), its source
    :attr:`document_ref`, the text, its :attr:`content_hash` and ``media_type``. Frozen so a handle
    is a value keyed by content — two identical extractions are equal."""

    chunk_id: str
    document_ref: str
    content: str
    content_hash: str
    media_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class UnsupportedArtifact:
    """Typed marker for content the extraction NIM surfaced but this pipeline cannot embed (e.g. a
    chart or a raw image with no extracted text). Returned in place of a handle so unsupported
    multimodal content is **visible and typed**, never silently dropped from the index."""

    document_ref: str
    media_type: str
    reason: str
    metadata: dict = field(default_factory=dict)


def make_artifact_handle(document_ref: str, content: str, media_type: str = "text/plain",
                         metadata: dict | None = None) -> ArtifactHandle:
    """The single canonical content-hashing path. Every extracted segment MUST pass through here to
    become an :class:`ArtifactHandle`; nothing mints a chunk id any other way."""
    h = content_hash(content)
    return ArtifactHandle(
        chunk_id=chunk_identity(document_ref, content), document_ref=document_ref,
        content=content, content_hash=h, media_type=media_type, metadata=dict(metadata or {}),
    )


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class NemoRetrieverExtractor:
    """NeMo Retriever **extraction** NIM (nemoretriever-parse) over OpenAI-compatible HTTP.

    Serve it on GPU (``docker run … nvcr.io/nim/nvidia/nemoretriever-parse``); it exposes an
    extraction endpoint that takes a document reference and returns page elements
    (``{"elements": [{"type", "content", "media_type"}, …]}``). Env-configurable like the other
    NeMo NIMs: ``REDEVOPS_RAG_NEMO_EXTRACT_URL``, ``REDEVOPS_RAG_NEMO_EXTRACT_MODEL``,
    optional ``REDEVOPS_RAG_NEMO_API_KEY`` → ``Authorization`` header.
    """

    provider = "nvidia-nemo-retriever"
    service_kind = "nemo-retriever"

    def __init__(self, url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: float = 180.0):
        self.url = (url or os.environ.get("REDEVOPS_RAG_NEMO_EXTRACT_URL")
                    or "http://127.0.0.1:8016/v1/extraction")
        self.model = model or os.environ.get("REDEVOPS_RAG_NEMO_EXTRACT_MODEL", "nvidia/nemoretriever-parse")
        self.api_key = api_key if api_key is not None else os.environ.get("REDEVOPS_RAG_NEMO_API_KEY", "")
        self.timeout = timeout

    def _post(self, document_ref: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "document": {"ref": document_ref}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        return json.load(urllib.request.urlopen(req, timeout=self.timeout))

    def extract(self, document_ref: str) -> list[ArtifactHandle | UnsupportedArtifact]:
        """Extract ``document_ref`` into canonical handles. Text-bearing elements pass through
        :func:`make_artifact_handle` (content-hashed chunk identity); non-text / empty elements come
        back as :class:`UnsupportedArtifact` markers — typed, not dropped."""
        resp = self._post(document_ref)
        return _elements_to_handles(document_ref, resp.get("elements") or [])


def _elements_to_handles(document_ref: str,
                         elements: Iterable[dict]) -> list[ArtifactHandle | UnsupportedArtifact]:
    out: list[ArtifactHandle | UnsupportedArtifact] = []
    for el in elements:
        media_type = (el.get("media_type") or "text/plain").strip().lower()
        content = el.get("content") or el.get("text") or ""
        meta = {"element_type": el.get("type")} if el.get("type") else {}
        if media_type in SUPPORTED_MEDIA and content.strip():
            out.append(make_artifact_handle(document_ref, content, media_type, meta))
        else:
            reason = ("empty extracted text" if not content.strip()
                      else f"unsupported media type {media_type!r}")
            out.append(UnsupportedArtifact(document_ref, media_type, reason, meta))
    return out


def extract(document_ref: str, extractor: NemoRetrieverExtractor | None = None
            ) -> list[ArtifactHandle | UnsupportedArtifact]:
    """``extract(document_ref) -> extracted artifact handles`` (+ typed unsupported markers)."""
    return (extractor or NemoRetrieverExtractor()).extract(document_ref)


def embed(handles: Iterable[ArtifactHandle], embedder=None) -> list[dict]:
    """``embed(handles) -> embedding refs``. Embeds the text of each canonical handle through the
    NeMo Retriever embedding NIM and returns refs that tie every vector back to the content-hashed
    chunk identity — the vector references the chunk, it does not redefine it. Only
    :class:`ArtifactHandle` items are embedded; :class:`UnsupportedArtifact` markers are skipped
    here (the caller has already seen them, typed)."""
    handles = [h for h in handles if isinstance(h, ArtifactHandle)]
    if embedder is None:
        from .embed import NemoRetrieverEmbedder
        embedder = NemoRetrieverEmbedder()
    vectors = embedder.encode([h.content for h in handles]) if handles else []
    model_id = getattr(embedder, "model", None)
    return [
        {"chunk_id": h.chunk_id, "content_hash": h.content_hash, "document_ref": h.document_ref,
         "media_type": h.media_type, "embedding": v, "model_id": model_id}
        for h, v in zip(handles, vectors)
    ]


def rerank(query: str, candidates: list[dict], reranker=None) -> dict:
    """``rerank(query, candidates) -> ranked+evidence``: the ``provider-capability`` result envelope
    from the NeMo Retriever reranking NIM (``ranked`` list + ``evidence`` with pre/post-rank order)."""
    if reranker is None:
        from .rerank import NemoRetrieverReranker
        reranker = NemoRetrieverReranker()
    return reranker.rerank_with_evidence(query, candidates)
