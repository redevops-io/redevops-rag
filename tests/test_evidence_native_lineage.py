"""Canonical evidence identity end-to-end in the RAG store (v0.2.x evidence-native stabilization).

Closes the audited seam where RAG severed canonical evidence identity: a retrieval hit now carries
the source ref, source revision (version) and strict ``rcv1`` content hash, and a superseded revision
stays addressable (point-in-time) instead of being pruned. These are the RAG-side integration tests
required before the v0.2.x freeze (plan tests 1 and 2).
"""
import hashlib

import pytest

from redevops_rag.store import Store
from redevops_rag import evidence
from redevops_rag.evidence import EvidenceRevision, ingest_revision, evidence_ref_from_hit

from runtime_contracts.canonical import content_hash as rcv1


class FakeEmbedder:
    backend = "fake"
    model_name = "fake"

    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in str(t).lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


# One long single-token paragraph → the packer keeps it as one chunk, so chunk text == content.
BODY_A = ("alpha " * 120).strip()
BODY_B = ("bravo " * 120).strip()

REF = "strategywiki/page/42"


def _store():
    return Store(FakeEmbedder(), ":memory:")


def test_1_ingest_revision_A_retrieve_with_canonical_identity():
    """Test 1: ingest evidence revision A → retrieve A carrying canonical source identity."""
    s = _store()
    revA = EvidenceRevision(ref=REF, version="1001", content=BODY_A,
                            observed_at="2009-01-01T00:00:00Z", source="wikimedia")
    res = ingest_revision(s, s.embedder, revA)
    s.reindex_fts()
    assert res["chunks"] >= 1

    hits = s.semantic_search(BODY_A, top_k=5, threshold=0.0)
    assert hits, "expected a hit for revision A"
    h = hits[0]
    # the hit carries source identity + version + strict rcv1 content hash
    assert h["source_ref"] == REF
    assert h["source_version"] == "1001"
    assert h["source_content_hash"] == rcv1(BODY_A)
    assert h["content_hash"] == rcv1(BODY_A)  # single-chunk doc: chunk hash == source hash
    assert h["observed_at"] == "2009-01-01T00:00:00Z"
    assert h["superseded_by"] is None  # A is current

    # and it round-trips to a canonical EvidenceRef a consumer can pin
    er = evidence_ref_from_hit(h)
    assert er is not None
    assert er.ref == REF and er.version == "1001"
    assert er.pin() == f"{REF}@1001#{rcv1(BODY_A)}"


def test_2_update_to_B_current_is_B_and_A_remains_resolvable():
    """Test 2: update to B → retrieve B as current while A remains resolvable (point-in-time)."""
    s = _store()
    s.embedder  # noqa
    ingest_revision(s, s.embedder, EvidenceRevision(ref=REF, version="1001", content=BODY_A,
                                                    observed_at="2009-01-01T00:00:00Z", source="wikimedia"))
    res_b = ingest_revision(s, s.embedder, EvidenceRevision(ref=REF, version="1002", content=BODY_B,
                                                    observed_at="2010-01-01T00:00:00Z", source="wikimedia"))
    s.reindex_fts()
    assert res_b["superseded"] >= 1, "ingesting B should supersede A's chunk(s)"

    # current projection → B only
    cur = s.semantic_search(BODY_B, top_k=10, threshold=0.0, current_only=True)
    assert cur and all(h["source_version"] == "1002" for h in cur)
    cur_a = s.semantic_search(BODY_A, top_k=10, threshold=0.0, current_only=True)
    assert all(h["source_version"] != "1001" for h in cur_a), "A must not appear in the current view"

    # A remains resolvable by exact version …
    pinned = s.semantic_search(BODY_A, top_k=10, threshold=0.0, source_version="1001")
    assert pinned and pinned[0]["source_version"] == "1001"
    assert pinned[0]["source_content_hash"] == rcv1(BODY_A)
    assert pinned[0]["superseded_by"] == "1002"  # retained, marked superseded — not deleted

    # … and by point-in-time as-of (before B existed)
    asof = s.semantic_search(BODY_A, top_k=10, threshold=0.0, as_of="2009-06-01T00:00:00Z",
                             source_version="1001")
    assert asof and asof[0]["source_version"] == "1001"


def test_wrong_version_substitution_is_zero():
    """Hard gate: pinning a version never returns a different revision's chunk."""
    s = _store()
    ingest_revision(s, s.embedder, EvidenceRevision(ref=REF, version="1001", content=BODY_A, source="w"))
    ingest_revision(s, s.embedder, EvidenceRevision(ref=REF, version="1002", content=BODY_B, source="w"))
    s.reindex_fts()
    for v in ("1001", "1002"):
        for hit in s.semantic_search("alpha bravo", top_k=20, threshold=0.0, source_version=v):
            assert hit["source_version"] == v


def test_lineage_is_derived_from_relation():
    """The chunk→source lineage uses the runtime-contracts DERIVED_FROM vocabulary."""
    from runtime_contracts.protocol.lineage import RelationKind
    rev = EvidenceRevision(ref=REF, version="1001", content=BODY_A, source="w")
    rel = evidence.chunk_lineage(rev, BODY_A)
    assert rel.kind == RelationKind.DERIVED_FROM
    src = evidence.source_evidence_ref(rev)
    chk = evidence.chunk_evidence_ref(rev, BODY_A)
    assert src.ref_type == "raw_source" and chk.ref_type == "chunk"
    assert chk.version == "1001"  # chunk identity pinned to the source revision
