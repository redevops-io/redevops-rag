"""Explicit retrieval outcomes (COMPLETE / EMPTY / DEGRADED / UNAVAILABLE / DENIED) and a deterministic evaluation clock."""
import logging
import re
from datetime import datetime, timedelta, timezone

import pytest

from redevops_rag.namespace import NamespaceMismatch
from redevops_rag.retrieval_result import BackendUnavailable, RetrievalStatus, RetrievalUnavailable, redact
from redevops_rag.retrieve import hybrid_search, hybrid_search_result
from redevops_rag.store import Store

DIM = 64


def _h(w: str) -> int:
    h = 5381
    for ch in w:
        h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return h


class HashEmbedder:
    dim = DIM
    sim_floor = 0.05

    def encode(self, texts):
        import math
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[_h(w) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class BrokenEmbedder(HashEmbedder):
    def __init__(self, message="embedding service unavailable"):
        self.message = message

    def encode(self, texts):
        raise RuntimeError(self.message)


CORPUS = {"A": "alpha budget forecast for the quarter alpha alpha", "B": "budget notes for the quarter, nothing else"}
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_store(namespace=None):
    e = HashEmbedder()
    s = Store(e, ":memory:", namespace=namespace)
    s.add_chunks([{"id": f"{d}::0", "document_id": d, "filename": f"{d}.md", "chunk_index": 0, "text": t,
                   "embedding": e.encode([t])[0], "metadata": {}, "created_at": T0} for d, t in CORPUS.items()])
    s.reindex_fts()
    return s


@pytest.fixture()
def store():
    s = make_store()
    if not s._fts:
        pytest.skip("DuckDB fts extension unavailable in this environment")
    return s


def test_healthy_search_is_complete(store):
    r = hybrid_search_result(store, "alpha budget forecast")
    assert r.status is RetrievalStatus.COMPLETE and r.hits and r.complete_coverage and r.usable
    assert [(l.name, l.ok, l.checked) for l in r.legs] == [("vector", True, True), ("bm25", True, True)]


def test_a_genuine_no_match_is_empty_not_degraded(store):
    r = hybrid_search_result(store, "zzzzqqqq")
    assert r.status is RetrievalStatus.EMPTY and r.hits == [] and r.complete_coverage
    assert hybrid_search(store, "zzzzqqqq") == []                       # legacy API: still a plain empty list


def test_blank_query_is_empty_with_a_note(store):
    r = hybrid_search_result(store, "   ")
    assert r.status is RetrievalStatus.EMPTY and "blank query" in r.notes


def test_vector_leg_down_is_degraded_and_names_the_leg(store, caplog):
    store.embedder = BrokenEmbedder()
    r = hybrid_search_result(store, "alpha budget forecast")
    assert r.status is RetrievalStatus.DEGRADED and r.usable and not r.complete_coverage
    assert r.hits and {h["source_type"] for h in r.hits} == {"bm25"}
    assert [l.name for l in r.failed_legs()] == ["vector"] and "RuntimeError" in r.failed_legs()[0].error
    with caplog.at_level(logging.WARNING, logger="redevops_rag.retrieve"):
        assert hybrid_search(store, "alpha budget forecast")                # legacy wrapper returns the BM25 hits...
    assert any("degraded" in m and "vector leg failed" in m for m in caplog.messages)   # ...and says so


def test_degraded_with_no_hits_is_not_reported_as_a_no_match(store):
    store.embedder = BrokenEmbedder()
    r = hybrid_search_result(store, "zzzzqqqq")
    assert r.status is RetrievalStatus.DEGRADED and r.hits == [] and not r.complete_coverage


def test_both_legs_down_is_unavailable_and_the_legacy_api_raises(store):
    store.embedder = BrokenEmbedder()
    store.con.execute("PRAGMA drop_fts_index('chunks')")                # the lexical leg now errors inside the database
    r = hybrid_search_result(store, "alpha budget forecast")
    assert r.status is RetrievalStatus.UNAVAILABLE and r.hits == [] and not r.usable
    assert [l.name for l in r.failed_legs()] == ["vector", "bm25"]
    with pytest.raises(RetrievalUnavailable) as ei:
        hybrid_search(store, "alpha budget forecast")
    assert ei.value.result.status is RetrievalStatus.UNAVAILABLE
    # and it is distinguishable from a genuine no-match on a healthy store
    assert hybrid_search_result(make_store(), "zzzzqqqq").status is RetrievalStatus.EMPTY


def test_missing_fts_extension_is_a_degraded_lexical_leg_not_an_empty_one():
    s = make_store()
    s._fts = False                                                      # e.g. the extension could not be installed offline
    r = hybrid_search_result(s, "alpha budget forecast")
    assert r.status is RetrievalStatus.DEGRADED and r.hits
    assert [l.name for l in r.failed_legs()] == ["bm25"] and "extension" in r.failed_legs()[0].error
    with pytest.raises(BackendUnavailable):
        s.bm25_search("alpha", strict=True)
    assert s.bm25_search("alpha") == []                                 # legacy store API unchanged


def test_namespace_refusal_is_denied_and_the_legacy_api_still_raises():
    s = make_store(namespace="tenant-A")
    r = hybrid_search_result(s, "budget", namespace="tenant-B")
    assert r.status is RetrievalStatus.DENIED and r.hits == [] and not r.usable
    assert hybrid_search_result(s, "budget").status is RetrievalStatus.DENIED       # no namespace given
    with pytest.raises(NamespaceMismatch):
        hybrid_search(s, "budget", namespace="tenant-B")


def test_error_text_never_carries_credentials(store):
    store.embedder = BrokenEmbedder("could not connect to postgresql://app:s3cr3t-pw@db.internal:5432/rag (password=hunter2)")
    r = hybrid_search_result(store, "alpha")
    err = r.failed_legs()[0].error
    assert "s3cr3t-pw" not in err and "hunter2" not in err and "***" in err
    assert "s3cr3t-pw" not in str(r.to_dict())
    assert redact("x" * 1000).endswith("…") and len(redact("x" * 1000)) <= 240


def test_a_store_that_cannot_report_failures_is_flagged_unchecked(store):
    class DuckTypedStore:
        namespace = None

        def semantic_search(self, text, top_k, threshold, document_ids, query_mode):
            return store.semantic_search(text, top_k=top_k, threshold=threshold, document_ids=document_ids, query_mode=query_mode)

        def bm25_search(self, text, limit=50, document_ids=None):       # legacy signature: no `strict`
            return []

    r = hybrid_search_result(DuckTypedStore(), "alpha budget forecast")
    assert r.status is RetrievalStatus.COMPLETE
    leg = {l.name: l for l in r.legs}["bm25"]
    assert leg.ok and leg.checked is False and any("cannot report failures" in n for n in r.notes)


# ── deterministic evaluation clock ─────────────────────────────────────────────────────────────────────────────
def _scores(store, as_of):
    return {h["chunk_id"]: h["boosted_score"] for h in hybrid_search_result(store, "budget quarter", as_of=as_of).hits}


def test_same_as_of_gives_identical_scores_regardless_of_the_wall_clock(store):
    t = T0 + timedelta(days=30)
    a = _scores(store, t)
    b = _scores(store, t)
    assert a == b and a
    r = hybrid_search_result(store, "budget quarter", as_of=t)
    assert r.as_of == t


def test_recency_boost_moves_only_when_as_of_moves(store):
    near, far = _scores(store, T0 + timedelta(days=1)), _scores(store, T0 + timedelta(days=400))
    assert all(far[k] < near[k] for k in near)                          # same chunks, older relative to a later 'now'
    # a half-life later the boost has halved (rrf unchanged, keyword prior unchanged)
    half = _scores(store, T0 + timedelta(days=90))
    quarter = _scores(store, T0 + timedelta(days=180))
    assert all(abs(quarter[k] / half[k] - 0.5) < 1e-6 for k in half)


def test_naive_as_of_is_treated_as_utc_and_default_records_the_wall_clock(store):
    assert _scores(store, datetime(2026, 1, 31)) == _scores(store, datetime(2026, 1, 31, tzinfo=timezone.utc))
    r = hybrid_search_result(store, "budget")
    assert r.as_of is not None and abs((datetime.now(timezone.utc) - r.as_of).total_seconds()) < 60
