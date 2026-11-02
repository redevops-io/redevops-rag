"""BM25 statistics must come from the authorised candidate corpus, not the whole index."""
import random
import re

import pytest

from redevops_rag.retrieval_result import BackendUnavailable, RetrievalStatus
from redevops_rag.retrieve import hybrid_search_result
from redevops_rag.store import Store

DIM = 32
VOCAB = ("alpha beta gamma delta budget forecast quarter running runs ran cafe café naïve results result resulting the and of "
         "to filler words nothing tests testing").split()
QUERIES = ["alpha budget", "running results", "café naïve tests", "forecast", "Quarter  BUDGET, forecast!", "alpha alpha alpha"]


class Emb:
    dim, sim_floor = DIM, 0.0

    def encode(self, texts):
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


def mk(tenant_aware=True):
    s = Store(Emb(), ":memory:", tenant_aware=tenant_aware)
    if not s._fts:
        pytest.skip("DuckDB fts extension unavailable in this environment")
    return s


def rows(tag, n, seed):
    r = random.Random(seed)
    return [{"id": f"{tag}{i}", "document_id": f"{tag}-d{i % 7}", "text": " ".join(r.choices(VOCAB, k=r.randint(1, 25))),
             "embedding": [1.0] + [0.0] * (DIM - 1)} for i in range(n)]


def load(s, tenant_rows):
    for tenant, rs in tenant_rows.items():
        s.add_chunks([dict(x) for x in rs], tenant_scope=tenant)
    s.reindex_fts()
    return s


def scores(hits, strip=True):
    return {(h["chunk_id"].split("::", 1)[1] if strip else h["chunk_id"]): h["bm25_score"] for h in hits}


def same(a, b, tol=1e-9):
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= tol for k in a)


TENANTS = {"tenant-A": rows("a", 60, 1), "tenant-B": rows("b", 25, 2), "tenant-C": rows("c", 90, 3)}


@pytest.mark.parametrize("q", QUERIES)
def test_scoped_scores_equal_a_standalone_index_of_just_that_tenants_rows(q):
    big = load(mk(), TENANTS)
    for tenant, rs in TENANTS.items():
        alone = load(mk(), {tenant: rs})                      # a store that only ever held this tenant
        assert same(scores(big.bm25_search(q, limit=1000, tenant_scope=tenant)),
                    scores(alone.bm25_search(q, limit=1000, tenant_scope=tenant))), (tenant, q)


@pytest.mark.parametrize("q", QUERIES)
def test_out_of_scope_ingestion_does_not_change_in_scope_scores(q):
    s = load(mk(), {"tenant-A": TENANTS["tenant-A"]})
    before = scores(s.bm25_search(q, limit=1000, tenant_scope="tenant-A"))
    s.add_chunks(rows("z", 400, 99), tenant_scope="tenant-Z")      # 400 unrelated rows, many sharing the query terms
    s.reindex_fts()
    assert same(before, scores(s.bm25_search(q, limit=1000, tenant_scope="tenant-A")))


def test_the_legacy_global_statistics_do_leak_which_is_why_scoped_stats_exist():
    s = load(mk(), {"tenant-A": TENANTS["tenant-A"]})
    before = scores(s.bm25_search("alpha budget", limit=1000, tenant_scope="tenant-A", scoped_stats=False))
    s.add_chunks(rows("z", 400, 99), tenant_scope="tenant-Z")
    s.reindex_fts()
    after = scores(s.bm25_search("alpha budget", limit=1000, tenant_scope="tenant-A", scoped_stats=False))
    assert not same(before, after), "global statistics should move when unrelated documents arrive"


@pytest.mark.parametrize("q", QUERIES + ["the and of", "nonexistentword"])
def test_with_everything_in_scope_the_query_reproduces_duckdbs_own_match_bm25(q):
    s = mk(tenant_aware=False)
    s.add_chunks(rows("u", 150, 5)); s.reindex_fts()
    engine = scores(s.bm25_search(q, limit=1000, scoped_stats=False), strip=False)
    mine = scores(s.bm25_search(q, limit=1000, scoped_stats=True), strip=False)
    assert same(engine, mine)


def test_document_ids_only_keeps_legacy_global_statistics_unless_asked():
    s = mk(tenant_aware=False)
    s.add_chunks(rows("u", 150, 5)); s.reindex_fts()
    docs = ["u-d1", "u-d2"]
    legacy = scores(s.bm25_search("alpha budget", limit=1000, document_ids=docs), strip=False)
    explicit_global = scores(s.bm25_search("alpha budget", limit=1000, document_ids=docs, scoped_stats=False), strip=False)
    scoped = scores(s.bm25_search("alpha budget", limit=1000, document_ids=docs, scoped_stats=True), strip=False)
    assert same(legacy, explicit_global) and not same(legacy, scoped) and set(legacy) == set(scoped)
    alone = mk(tenant_aware=False)
    alone.add_chunks([r for r in rows("u", 150, 5) if r["document_id"] in docs]); alone.reindex_fts()
    assert same(scoped, scores(alone.bm25_search("alpha budget", limit=1000), strip=False))


def test_results_are_deterministic_and_ties_break_on_id():
    s = mk(tenant_aware=False)
    s.add_chunks([{"id": f"t{i}", "document_id": "d", "text": "alpha beta", "embedding": [1.0] + [0.0] * (DIM - 1)}
                  for i in (3, 1, 2, 0)])
    s.reindex_fts()
    for scoped in (False, True):
        ids = [h["chunk_id"] for h in s.bm25_search("alpha", scoped_stats=scoped)]
        assert ids == ["t0", "t1", "t2", "t3"], (scoped, ids)


def test_limit_is_applied_after_scoping_and_scoring():
    s = load(mk(), TENANTS)
    full = s.bm25_search("alpha budget", limit=1000, tenant_scope="tenant-B")
    top3 = s.bm25_search("alpha budget", limit=3, tenant_scope="tenant-B")
    assert [h["chunk_id"] for h in top3] == [h["chunk_id"] for h in full[:3]] and len(top3) == 3


def test_an_unrecognised_fts_layout_fails_closed_instead_of_using_global_statistics():
    s = load(mk(), TENANTS)
    s._scoped_bm25_ok = False                                     # as if a future DuckDB changed the index layout
    with pytest.raises(BackendUnavailable):
        s.bm25_search("alpha", tenant_scope="tenant-A", strict=True)
    assert s.bm25_search("alpha", tenant_scope="tenant-A") == []   # legacy non-strict contract
    r = hybrid_search_result(s, "alpha budget", tenant_scope="tenant-A", vector_threshold=-1.0)
    assert r.status is RetrievalStatus.DEGRADED and [l.name for l in r.failed_legs()] == ["bm25"]
    assert "scoped BM25" in r.failed_legs()[0].error


def test_hybrid_on_a_tenant_aware_store_uses_scoped_statistics():
    s = load(mk(), TENANTS)
    alone = load(mk(), {"tenant-B": TENANTS["tenant-B"]})
    r = hybrid_search_result(s, "alpha budget", limit=1000, pool=1000, tenant_scope="tenant-B", vector_threshold=-1.0)
    assert r.status is RetrievalStatus.COMPLETE
    expect = scores(alone.bm25_search("alpha budget", limit=1000, tenant_scope="tenant-B"))
    got = {h["chunk_id"].split("::", 1)[1]: h["bm25_score"] for h in r.hits if "bm25_score" in h}
    assert got and all(abs(got[k] - expect[k]) < 1e-9 for k in got)
