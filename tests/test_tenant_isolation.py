"""Tenant-aware mode: isolation contract + fail-closed identity handling.

Unscoped stores keep the historical behaviour and give NO isolation (RetrievalResult.isolation == "unscoped").
A tenant-aware store must (a) refuse a missing/invalid scope, (b) filter before top-k in every leg, (c) never let one
tenant overwrite, prune or list another tenant's rows."""
import re

import pytest

from redevops_rag.namespace import NamespaceMismatch
from redevops_rag.retrieval_result import RetrievalStatus
from redevops_rag.retrieve import diver_search, hybrid_search, hybrid_search_result
from redevops_rag.store import Store
from redevops_rag.tenancy import TenantScopeError, qualify_id, validate_tenant_scope

DIM = 64


def _h(w):
    h = 5381
    for ch in w:
        h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return h


class HashEmbedder:
    dim, sim_floor = DIM, 0.0

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


E = HashEmbedder()


def chunk(cid, doc, text, **kw):
    return {"id": cid, "document_id": doc, "filename": f"{doc}.md", "chunk_index": 0, "text": text,
            "embedding": E.encode([text])[0], "metadata": {}, **kw}


def tenant_store(**kw):
    s = Store(E, ":memory:", tenant_aware=True, **kw)
    if not s._fts:
        pytest.skip("DuckDB fts extension unavailable in this environment")
    return s


def seed_two_tenants(s):
    """Colliding document id and id; A has many strong matches that outrank B's single one."""
    a = [chunk("c0", "doc1", "quarterly budget forecast shared text")]
    a += [chunk(f"a{i}", f"adoc{i}", "quarterly budget forecast alpha alpha alpha alpha") for i in range(1, 30)]
    b = [chunk("c0", "doc1", "quarterly budget forecast shared text"), chunk("b1", "bdoc1", "budget notes for beta only")]
    s.add_chunks(a, tenant_scope="tenant-A")
    s.add_chunks(b, tenant_scope="tenant-B")
    s.reindex_fts()
    return s


def owners(hits):
    return {h["chunk_id"].split("::", 1)[0] for h in hits}


# ── unscoped mode is unchanged and is never isolation ────────────────────────────────────────────────────────────
def test_unscoped_store_keeps_working_without_identity_and_reports_unscoped():
    s = Store(E, ":memory:")
    s.add_chunks([chunk("x", "d", "budget forecast")])
    s.reindex_fts()
    r = hybrid_search_result(s, "budget forecast")
    assert r.status in (RetrievalStatus.COMPLETE, RetrievalStatus.DEGRADED) and r.hits
    assert r.isolation == "unscoped" and r.to_dict()["isolation"] == "unscoped"
    assert s.count() == 1


def test_a_scope_given_to_an_unscoped_store_is_applied_but_does_not_make_it_isolated():
    s = Store(E, ":memory:")
    s.add_chunks([chunk("x", "d", "budget forecast")], tenant_scope="tenant-A")
    s.add_chunks([chunk("y", "d2", "budget forecast")], tenant_scope="tenant-B")
    s.reindex_fts()
    r = hybrid_search_result(s, "budget forecast", tenant_scope="tenant-A")
    assert {h["chunk_id"] for h in r.hits} == {"x"} and r.isolation == "unscoped"     # filtered, yet still "unscoped"


# ── fail closed ──────────────────────────────────────────────────────────────────────────────────────────────────
BAD_SCOPES = [None, "tenant::A", "", "   ", " tenant-A", "tenant-A ", "*", "%", "ALL", "public", "tenant\x00A", "tenant\nA", 123, b"tenant-A",
              ["tenant-A"], "x" * 257]


@pytest.mark.parametrize("scope", BAD_SCOPES)
def test_missing_or_invalid_identity_is_denied_before_any_retrieval(scope):
    s = seed_two_tenants(tenant_store())
    r = hybrid_search_result(s, "budget forecast", tenant_scope=scope)
    assert r.status is RetrievalStatus.DENIED and r.hits == [] and r.legs == () and r.isolation == "tenant_aware"
    with pytest.raises(TenantScopeError):
        hybrid_search(s, "budget forecast", tenant_scope=scope)
    for call in (lambda: s.semantic_search("budget", tenant_scope=scope), lambda: s.bm25_search("budget", tenant_scope=scope),
                 lambda: s.count(tenant_scope=scope)):
        with pytest.raises(TenantScopeError):
            call()


def test_a_denied_blank_query_is_still_denied_not_empty():
    s = seed_two_tenants(tenant_store())
    assert hybrid_search_result(s, "", tenant_scope=None).status is RetrievalStatus.DENIED
    with pytest.raises(TenantScopeError):
        s.bm25_search("   ")


def test_error_messages_do_not_echo_the_offending_scope():
    with pytest.raises(TenantScopeError) as ei:
        validate_tenant_scope("tenant\x00-secret-value")
    assert "secret-value" not in str(ei.value)


def test_a_duck_typed_store_that_cannot_enforce_a_scope_is_denied():
    class Plain:
        namespace = None
        tenant_aware = False

        def semantic_search(self, text, top_k, threshold, document_ids, query_mode):
            raise AssertionError("must not be reached")

        def bm25_search(self, text, limit=50, document_ids=None):
            raise AssertionError("must not be reached")

    r = hybrid_search_result(Plain(), "budget", tenant_scope="tenant-A")
    assert r.status is RetrievalStatus.DENIED and "cannot enforce" in r.notes[0]


# ── isolation across every leg, at top_k = 1 and 100 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 100])
def test_every_leg_returns_only_the_callers_rows(n):
    s = seed_two_tenants(tenant_store())
    for tenant, other in (("tenant-A", "tenant-B"), ("tenant-B", "tenant-A")):
        for hits in (s.semantic_search("budget forecast", top_k=n, threshold=-1.0, tenant_scope=tenant),
                     s.bm25_search("budget forecast", limit=n, tenant_scope=tenant),
                     hybrid_search(s, "budget forecast", limit=n, pool=n, tenant_scope=tenant, vector_threshold=-1.0),
                     diver_search(s, "budget forecast", None, limit=n, pool=n, tenant_scope=tenant)):
            assert hits, "the tenant has matching rows"
            assert owners(hits) == {tenant}, f"{other}'s rows leaked into {tenant}'s results"


def test_diver_with_an_llm_stays_in_scope_and_never_sends_a_denied_query():
    s = seed_two_tenants(tenant_store())
    seen = []

    def llm(system, user):
        seen.append(user)
        return "1. budget forecast\n2. beta notes"

    hits = diver_search(s, "budget forecast", llm, limit=5, pool=10, tenant_scope="tenant-B")
    assert hits and owners(hits) == {"tenant-B"}
    seen.clear()
    with pytest.raises(TenantScopeError):
        diver_search(s, "budget forecast", llm, tenant_scope=None)
    assert seen == [], "a refused request must not reach the LLM"


def test_scope_is_applied_before_top_k_not_after():
    # tenant-A has 29 rows that outrank tenant-B's best; B asking for 1 result must still get a B row, not nothing.
    s = seed_two_tenants(tenant_store())
    unscoped_top = Store.semantic_search(s, "budget forecast alpha", top_k=1, threshold=-1.0, tenant_scope="tenant-A")
    assert owners(unscoped_top) == {"tenant-A"}
    for hits in (s.semantic_search("budget forecast alpha", top_k=1, threshold=-1.0, tenant_scope="tenant-B"),
                 s.bm25_search("budget forecast", limit=1, tenant_scope="tenant-B"),
                 hybrid_search(s, "budget forecast alpha", limit=1, pool=1, tenant_scope="tenant-B", vector_threshold=-1.0)):
        assert len(hits) == 1 and owners(hits) == {"tenant-B"}


def test_document_ids_filter_and_tenant_scope_compose():
    s = seed_two_tenants(tenant_store())
    hits = hybrid_search(s, "budget", limit=10, tenant_scope="tenant-B", document_ids=["doc1"], vector_threshold=-1.0)
    assert {h["document_id"] for h in hits} == {"doc1"} and owners(hits) == {"tenant-B"}
    # a document id that belongs only to the other tenant yields nothing, not their row
    assert hybrid_search(s, "budget", limit=10, tenant_scope="tenant-B", document_ids=["adoc1"], vector_threshold=-1.0) == []


def test_namespace_and_tenant_scope_are_both_enforced():
    s = seed_two_tenants(tenant_store(namespace="ns1"))
    assert hybrid_search_result(s, "budget", namespace="ns1", tenant_scope="tenant-A").status is not RetrievalStatus.DENIED
    assert hybrid_search_result(s, "budget", namespace="ns2", tenant_scope="tenant-A").status is RetrievalStatus.DENIED
    assert hybrid_search_result(s, "budget", namespace="ns1", tenant_scope=None).status is RetrievalStatus.DENIED
    with pytest.raises(NamespaceMismatch):
        hybrid_search(s, "budget", namespace="ns2", tenant_scope="tenant-A")


# ── writes and admin calls ───────────────────────────────────────────────────────────────────────────────────────
def test_a_write_without_a_scope_is_refused_and_a_scope_mismatch_is_refused():
    s = tenant_store()
    with pytest.raises(TenantScopeError):
        s.add_chunks([chunk("x", "d", "t")])
    with pytest.raises(TenantScopeError):
        s.add_chunks([chunk("x", "d", "t", tenant_scope="tenant-B")], tenant_scope="tenant-A")
    s.add_chunks([chunk("x", "d", "t", tenant_scope="tenant-A")])         # the row's own scope is enough
    assert s.count(tenant_scope="tenant-A") == 1


def test_identical_ids_from_two_tenants_never_overwrite_each_other():
    s = tenant_store()
    s.add_chunks([chunk("c0", "doc1", "alpha original")], tenant_scope="tenant-A")
    s.add_chunks([chunk("c0", "doc1", "beta replacement")], tenant_scope="tenant-B")
    s.reindex_fts()
    a = s.semantic_search("alpha", threshold=-1.0, tenant_scope="tenant-A")
    b = s.semantic_search("beta", threshold=-1.0, tenant_scope="tenant-B")
    assert [h["text"] for h in a] == ["alpha original"] and [h["text"] for h in b] == ["beta replacement"]
    assert a[0]["chunk_id"] == qualify_id("tenant-A", "c0") != b[0]["chunk_id"]
    assert s.count(tenant_scope="tenant-A") == 1 and s.count(tenant_scope="tenant-B") == 1


def test_scopes_that_would_make_two_tenants_share_a_stored_id_are_rejected():
    # scope "t" + id "x::y" and scope "t::x" + id "y" would both be stored as "t::x::y" — so "::" is not a legal scope
    s = tenant_store()
    s.add_chunks([chunk("x::y", "doc1", "alpha original")], tenant_scope="t")
    with pytest.raises(TenantScopeError):
        s.add_chunks([chunk("y", "doc1", "poison")], tenant_scope="t::x")
    assert [h["text"] for h in s.semantic_search("alpha", threshold=-1.0, tenant_scope="t")] == ["alpha original"]


def test_the_store_refuses_to_replace_a_row_owned_by_another_scope_even_if_the_id_matches():
    # defence in depth: corrupt the state so a stored id belongs to another scope, then try to write that id
    s = tenant_store()
    s.add_chunks([chunk("c0", "doc1", "alpha original")], tenant_scope="tenant-A")
    s.con.execute("UPDATE chunks SET tenant_scope = 'tenant-X'")
    with pytest.raises(TenantScopeError):
        s.add_chunks([chunk("c0", "doc1", "re-ingest by A")], tenant_scope="tenant-A")
    assert s.con.execute("SELECT text FROM chunks").fetchone()[0] == "alpha original"


def test_a_tenant_can_update_its_own_rows():
    s = tenant_store()
    s.add_chunks([chunk("c0", "doc1", "alpha original")], tenant_scope="tenant-A")
    s.add_chunks([chunk("c0", "doc1", "alpha updated")], tenant_scope="tenant-A")
    assert [h["text"] for h in s.semantic_search("alpha", threshold=-1.0, tenant_scope="tenant-A")] == ["alpha updated"]


def test_prune_and_document_listing_are_confined_to_the_callers_tenant():
    s = tenant_store()
    s.add_chunks([chunk("k1", "/data/doc.md", "alpha one"), chunk("k2", "/data/doc.md", "alpha two")], tenant_scope="tenant-A")
    s.add_chunks([chunk("k1", "/data/doc.md", "beta one")], tenant_scope="tenant-B")
    assert s.document_ids("/data", tenant_scope="tenant-A") == ["/data/doc.md"]
    assert s.document_ids("/other", tenant_scope="tenant-A") == []
    removed = s.prune_document("/data/doc.md", ["k1"], tenant_scope="tenant-A")        # A keeps k1, drops k2
    assert removed == 1 and s.count(tenant_scope="tenant-A") == 1 and s.count(tenant_scope="tenant-B") == 1
    assert s.prune_document("/data/doc.md", [], tenant_scope="tenant-A") == 1          # A drops the rest...
    assert s.count(tenant_scope="tenant-B") == 1                                       # ...B untouched
    with pytest.raises(TenantScopeError):
        s.prune_document("/data/doc.md", [])
    with pytest.raises(TenantScopeError):
        s.document_ids("/data")


def test_ingest_requires_a_scope_and_one_tenants_reingest_never_prunes_anothers(tmp_path):
    from redevops_rag.ingest import ingest
    f = tmp_path / "shared.md"
    f.write_text("quarterly budget forecast text for both tenants")
    s = tenant_store()
    with pytest.raises(TenantScopeError):
        ingest(s, E, tmp_path)                                                          # refused before reading any file
    ingest(s, E, tmp_path, tenant_scope="tenant-A")
    ingest(s, E, tmp_path, tenant_scope="tenant-B")
    f.write_text("completely rewritten content about gardening")
    ingest(s, E, tmp_path, tenant_scope="tenant-A")                                     # A re-ingests; B's old chunk must stay
    s.reindex_fts()
    b = s.bm25_search("budget forecast", tenant_scope="tenant-B")
    assert b and owners(b) == {"tenant-B"}
    a = s.bm25_search("gardening", tenant_scope="tenant-A")
    assert a and owners(a) == {"tenant-A"}
    assert s.bm25_search("budget forecast", tenant_scope="tenant-A") == []


# ── migration of an existing database ───────────────────────────────────────────────────────────────────────────
def test_legacy_database_is_migrated_and_untagged_rows_are_invisible_to_tenants(tmp_path):
    db = str(tmp_path / "legacy.duckdb")
    import duckdb
    con = duckdb.connect(db)
    con.execute(f"""CREATE TABLE chunks (id VARCHAR PRIMARY KEY, document_id VARCHAR, filename VARCHAR, chunk_index INTEGER,
                    content_hash VARCHAR, text VARCHAR, embedding FLOAT[{DIM}], metadata VARCHAR, created_at TIMESTAMP)""")
    con.execute("INSERT INTO chunks VALUES ('old','d','d.md',0,NULL,'budget forecast legacy',?,'{}',now())", [E.encode(["budget forecast legacy"])[0]])
    con.close()
    legacy = Store(E, db)                                                               # unscoped reopen: still sees it
    assert legacy.count() == 1 and legacy.semantic_search("budget", threshold=-1.0)
    legacy.close()
    s = Store(E, db, tenant_aware=True)
    assert s.semantic_search("budget", threshold=-1.0, tenant_scope="tenant-A") == []   # fail closed: untagged ≠ shared
    s.add_chunks([chunk("n", "d2", "budget forecast new")], tenant_scope="tenant-A")
    assert owners(s.semantic_search("budget", threshold=-1.0, tenant_scope="tenant-A")) == {"tenant-A"}


# ── the suite must catch a leak (mutation check) ────────────────────────────────────────────────────────────────
def test_the_contract_detects_a_store_that_forgets_to_filter():
    class LeakyStore(Store):
        def semantic_search(self, text, top_k=50, threshold=None, document_ids=None, query_mode="auto", tenant_scope=None):
            self._scope(tenant_scope)                         # validates the scope... then (the bug) never filters on it
            q = self._encode_query(text, "auto")
            rows = self.con.execute(
                f"SELECT id, document_id, filename, chunk_index, content_hash, text, metadata, created_at, "
                f"array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS sim FROM chunks ORDER BY sim DESC LIMIT ?",
                [list(q), int(top_k)]).fetchall()
            return [self._row(r, "similarity", r[8], "vector") for r in rows]

    s = seed_two_tenants(LeakyStore(E, ":memory:", tenant_aware=True))
    leaked = s.semantic_search("budget forecast alpha", top_k=1, threshold=-1.0, tenant_scope="tenant-B")
    assert owners(leaked) == {"tenant-A"}, "the mutant must leak, otherwise this suite could not detect the defect"
