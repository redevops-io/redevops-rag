"""Canonical evidence identity for retrieval — the source-revision → EvidenceRef → chunk chain.

The base ingest path (:mod:`redevops_rag.ingest`) addresses a chunk by its own bare content hash and
prunes superseded content, which is correct for a *current-state* notes vault but severs the canonical
identity that point-in-time replay depends on: once a source is edited, the evidence that a past
decision was bound to is gone, and a retrieval hit cannot say *which source revision* it came from.

This module closes that seam without disturbing the base path. It represents a **source revision**
explicitly, mints a canonical ``runtime_contracts.EvidenceRef`` (strict ``rcv1`` identity) for the
source and for each derived chunk, ingests with **version-aware retention** (a new revision supersedes
the prior one but does not delete it — historical evidence stays addressable), and lets a retrieval
hit be turned back into the exact ``EvidenceRef`` (ref + version + content hash) a consumer can pin.

runtime-contracts is imported lazily so the base package still installs and runs without it; only the
evidence-native path here requires it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .ingest import chunk_text

#: ref_type values (subset of runtime_contracts KNOWN_REF_TYPES) used by this bridge.
RAW_SOURCE = "raw_source"
CHUNK = "chunk"


def _rc():
    """Import runtime-contracts lazily; raise a clear error if the evidence path is used without it."""
    try:
        import runtime_contracts as rc  # noqa: F401
        from runtime_contracts.canonical import content_hash as rcv1_hash
        from runtime_contracts import EvidenceRef
        from runtime_contracts.protocol.lineage import derived_from
        return rcv1_hash, EvidenceRef, derived_from
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "redevops_rag.evidence requires runtime-contracts (>=0.3.0). "
            "Install it to use the canonical evidence path."
        ) from e


def rcv1(content: str) -> str:
    """Strict rcv1 content hash (``rcv1:<sha256>``) of an authoritative string representation.

    This is runtime-contracts' *strict* canonical hash (``canonical.content_hash``), NOT the bare
    sha256 the base RAG path uses — so a chunk minted here shares one identity space with Discovery,
    Mission and the seal/conformance layer.
    """
    rcv1_hash, _, _ = _rc()
    return rcv1_hash(content)


@dataclass(frozen=True)
class EvidenceRevision:
    """One authoritative revision of a source, the input to :func:`ingest_revision`.

    ``ref`` is the *stable* source identity (survives renames); ``version`` is the source's own
    revision id (e.g. a Wikimedia ``revid``, a git sha, a document version). ``content`` is the
    authoritative text whose ``rcv1`` becomes the source content hash. This type is domain-neutral:
    a caller maps its own source (wiki, repo, CMS) onto these fields.
    """

    ref: str
    version: str
    content: str
    observed_at: str = ""          # source timestamp (ISO-8601), distinct from ingest time
    source: str = ""               # provenance system name, e.g. "wikimedia"
    media_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)


def source_evidence_ref(rev: EvidenceRevision) -> Any:
    """The canonical ``EvidenceRef`` for the source revision itself (``ref_type="raw_source"``)."""
    _, EvidenceRef, _ = _rc()
    return EvidenceRef(
        ref=rev.ref, content_hash=rcv1(rev.content), version=rev.version,
        media_type=rev.media_type, source=rev.source, ref_type=RAW_SOURCE,
    )


def chunk_evidence_ref(rev: EvidenceRevision, chunk_content: str) -> Any:
    """The canonical ``EvidenceRef`` for one chunk derived from ``rev`` (``ref_type="chunk"``).

    The chunk's ref is ``{source_ref}#chunk`` and its version is the *source* revision, so a chunk's
    identity is pinned to the exact revision it was derived from — two revisions of the same source
    yield distinct, independently-addressable chunk refs even when the chunk text is unchanged.
    """
    _, EvidenceRef, _ = _rc()
    return EvidenceRef(
        ref=f"{rev.ref}#chunk", content_hash=rcv1(chunk_content), version=rev.version,
        media_type=rev.media_type, source=rev.source, ref_type=CHUNK,
    )


def chunk_lineage(rev: EvidenceRevision, chunk_content: str) -> Any:
    """A ``DERIVED_FROM`` relation tying the chunk EvidenceRef back to the source EvidenceRef.

    Uses the existing runtime-contracts lineage vocabulary rather than inventing a second scheme.
    """
    _, _, derived_from = _rc()
    src = source_evidence_ref(rev)
    chk = chunk_evidence_ref(rev, chunk_content)
    return derived_from(
        chk.pin(), src.pin(), subject_type=CHUNK, source_type=RAW_SOURCE,
    )


def versioned_chunk_id(rev: EvidenceRevision, chunk_content: str) -> str:
    """Row identity for a chunk: ``{source_ref}@{version}::{rcv1_short}``.

    Includes the source version so each (source, revision, chunk-content) is uniquely addressable and
    retained — unlike the base path's ``{doc}::{hash[:16]}``, which collapses revisions of the same
    content and is pruned on edit.
    """
    h = rcv1(chunk_content).split(":", 1)[-1]  # drop the "rcv1:" prefix for a compact id
    return f"{rev.ref}@{rev.version}::{h[:16]}"


def evidence_ref_from_hit(hit: dict) -> Any:
    """Reconstruct the source ``EvidenceRef`` a retrieval hit came from, ready to ``.pin()``.

    Returns ``None`` for a legacy hit that carries no source identity (base-path ingest), so callers
    can tell evidence-native hits from legacy ones.
    """
    if not hit.get("source_ref"):
        return None
    _, EvidenceRef, _ = _rc()
    return EvidenceRef(
        ref=hit["source_ref"], content_hash=hit.get("source_content_hash", ""),
        version=hit.get("source_version", ""), source=hit.get("source", ""),
        ref_type=RAW_SOURCE,
    )


def ingest_revision(store, embedder, rev: EvidenceRevision, *, size: int = 1000,
                    overlap: int = 150, batch: int = 64) -> dict:
    """Ingest one source revision with canonical identity + version-aware retention.

    Chunks ``rev.content``, mints a canonical chunk EvidenceRef per chunk, persists the source
    identity (ref/version/content_hash/observed_at) on every row, then marks any *prior* revision of
    the same ``rev.ref`` as superseded (retained, still addressable) rather than deleting it.

    Returns ``{"source_ref", "source_version", "chunks", "superseded"}``.
    """
    pieces = chunk_text(rev.content, size, overlap)
    src_hash = rcv1(rev.content)
    now = datetime.now(timezone.utc)
    total = 0
    for i in range(0, len(pieces), batch):
        part = pieces[i : i + batch]
        embs = embedder.encode(part)
        rows = []
        for j, (t, e) in enumerate(zip(part, embs)):
            rows.append({
                "id": versioned_chunk_id(rev, t),
                "document_id": rev.ref,
                "filename": rev.ref,
                "chunk_index": i + j,
                "content_hash": rcv1(t),
                "text": t,
                "embedding": e,
                "metadata": {**rev.metadata, "chunk_ref": f"{rev.ref}#chunk"},
                "created_at": now,
                # canonical evidence identity (new, version-aware):
                "source_ref": rev.ref,
                "source_version": rev.version,
                "source_content_hash": src_hash,
                "observed_at": rev.observed_at,
            })
        total += store.add_chunks(rows)
    superseded = 0
    supersede = getattr(store, "supersede_source", None)
    if callable(supersede):
        superseded = supersede(rev.ref, rev.version)
    return {
        "source_ref": rev.ref, "source_version": rev.version,
        "chunks": total, "superseded": superseded,
    }
