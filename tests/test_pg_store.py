"""Integration parity test for PgStore.

Verifies the pgvector-backed store honours the same public interface as
the DuckDB Store — same method names, same return shape, same ranking
semantics — so callers (RAG, context-runtime plugins) can swap backends
by URL scheme alone.

Skipped unless ``REDEVOPS_RAG_PG_TEST_URL`` is set to a live Postgres
that has (or can install) the ``vector`` extension. The test creates a
throwaway table and drops it at teardown; no persistent state.

Run locally against the startupblunders postgres::

    docker exec <pg-container> psql -U epicfails -d epicfails \\
        -c "CREATE DATABASE redevops_rag_test OWNER epicfails"
    export REDEVOPS_RAG_PG_TEST_URL="postgresql://epicfails:epicfails@localhost:5433/redevops_rag_test"
    pytest tests/test_pg_store.py -v
"""
from __future__ import annotations

import hashlib
import os
import uuid

import pytest

try:  # noqa: SIM105
    import psycopg  # noqa: F401
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


PG_URL = os.environ.get("REDEVOPS_RAG_PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL or psycopg is None,
    reason="set REDEVOPS_RAG_PG_TEST_URL + install [pg] extra to run",
)


class _FakeEmbedder:
    """Deterministic hash-based 'embedder' — no torch, no network.

    Maps text → a stable 32-dim unit vector by taking SHA-256 bytes,
    projecting to floats in [-1, 1], and normalising. Similar texts hash
    to different vectors, so this is only useful for exercising the
    plumbing (schema, cosine query returns rows, tsvector query returns
    rows) — not for measuring retrieval quality.
    """

    dim = 32

    def encode(self, texts):
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = [(b - 128) / 128.0 for b in h[: self.dim]]
            # unit-normalise so cosine similarity is stable
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs


@pytest.fixture
def embedder():
    return _FakeEmbedder()


@pytest.fixture
def store(embedder):
    from redevops_rag.pg_store import PgStore

    # Unique table per test run so parallel invocations don't collide.
    table = f"rag_test_{uuid.uuid4().hex[:12]}"
    s = PgStore(embedder, PG_URL, table=table)
    yield s
    # Cleanup
    with s.con.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
    s.con.commit()
    s.close()


def _chunks(embedder, *texts, **overrides):
    """Bundle text + pre-computed embedding, matching the shape ingest.py
    produces before it hands rows to store.add_chunks."""
    embs = embedder.encode(list(texts))
    out = []
    for i, (t, e) in enumerate(zip(texts, embs)):
        row = {"text": t, "embedding": e, "chunk_index": i}
        row.update(overrides)
        out.append(row)
    return out


def test_add_and_count(store, embedder):
    n = store.add_chunks(_chunks(embedder,
        "the gut microbiome regulates immune signalling",
        "vitamin d deficiency correlates with poor sleep",
        "creatine improves anaerobic exercise performance",
    ))
    assert n == 3
    assert store.count() == 3


def test_upsert_by_id(store, embedder):
    e1 = embedder.encode(["first version of the doc"])[0]
    e2 = embedder.encode(["second version overwrites"])[0]
    store.add_chunks([{"id": "a", "text": "first version of the doc", "embedding": e1}])
    store.add_chunks([{"id": "a", "text": "second version overwrites", "embedding": e2}])
    assert store.count() == 1


def test_semantic_search_returns_shape(store, embedder):
    store.add_chunks(_chunks(embedder, "vitamin d deficiency correlates with poor sleep"))
    hits = store.semantic_search("nutrition query", top_k=5, threshold=-1.0)
    assert len(hits) >= 1
    h = hits[0]
    # exact keys the DuckDB Store returns — parity contract
    for k in ("chunk_id", "document_id", "filename", "chunk_index", "text",
              "metadata", "created_at", "similarity", "source_type"):
        assert k in h, f"missing key {k!r} in {h}"
    assert h["source_type"] == "vector"
    assert isinstance(h["similarity"], float)


def test_bm25_search_returns_shape(store, embedder):
    store.add_chunks(_chunks(embedder,
        "the gut microbiome regulates immune signalling",
        "creatine improves anaerobic exercise performance",
    ))
    hits = store.bm25_search("microbiome immune", limit=5)
    assert len(hits) >= 1
    h = hits[0]
    for k in ("chunk_id", "document_id", "filename", "chunk_index", "text",
              "metadata", "created_at", "bm25_score", "source_type"):
        assert k in h, f"missing key {k!r} in {h}"
    assert h["source_type"] == "bm25"
    # ts_rank_cd > 0 when the query matched
    assert h["bm25_score"] > 0


def test_bm25_empty_query_returns_empty(store, embedder):
    store.add_chunks(_chunks(embedder, "any"))
    assert store.bm25_search("", limit=5) == []
    assert store.bm25_search("   ", limit=5) == []


def test_reindex_fts_is_idempotent(store, embedder):
    store.add_chunks(_chunks(embedder, "doc one", "doc two", "doc three"))
    store.reindex_fts()
    store.reindex_fts()  # second call must not raise


def test_metadata_json_roundtrip(store, embedder):
    store.add_chunks(_chunks(embedder, "hello",
                             metadata={"path": "/tmp/x.md", "tags": ["a", "b"]}))
    hits = store.semantic_search("hello", top_k=5, threshold=-1.0)
    assert hits[0]["metadata"] == {"path": "/tmp/x.md", "tags": ["a", "b"]}


def test_rag_facade_routes_on_pg_url(monkeypatch):
    """RAG(db_path='postgresql://...') must pick PgStore, not DuckDB.
    Uses monkeypatch to swap in the fake embedder so no torch download."""
    from redevops_rag import rag as rag_mod

    monkeypatch.setattr(rag_mod, "Embedder", lambda *a, **kw: _FakeEmbedder())
    table = f"rag_test_{uuid.uuid4().hex[:12]}"
    r = rag_mod.RAG(db_path=PG_URL, table=table)
    try:
        from redevops_rag.pg_store import PgStore
        assert isinstance(r.store, PgStore)
        # End-to-end: hybrid_search via RAG.search must not crash on empty store
        assert r.search("nothing indexed yet", k=3) == []
    finally:
        with r.store.con.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        r.store.con.commit()
        r.close()


def test_rag_facade_rejects_pg_kwargs_with_duckdb(tmp_path):
    """table/schema kwargs are pgvector-only — must raise when db_path is DuckDB."""
    from redevops_rag import rag as rag_mod

    with pytest.raises(ValueError, match="Postgres db_path"):
        rag_mod.RAG(db_path=str(tmp_path / "x.duckdb"), table="foo")


# ── parity with the DuckDB Store: metadata stamp, document_ids scoping, asymmetric query ──────
class _StampEmbedder(_FakeEmbedder):
    """A fake encoder that carries a backend tag + an asymmetric query side (like Nemotron)."""
    backend = "nemotron"
    model_name = "nemotron-embed"

    def __init__(self):
        self.q_calls = []

    def encode_queries(self, texts):
        self.q_calls += list(texts)
        return self.encode(texts)


def test_meta_stamp_and_get(store):
    # _FakeEmbedder has no .backend → nothing stamped automatically; set_meta/get_meta still work.
    store.set_meta(backend="bge", model="bge-small", dim=32, lang="en")
    assert store.get_meta("backend") == "bge"
    assert store.get_meta("dim") == 32
    assert store.get_meta()["lang"] == "en"


def test_stamp_encoder_records_backend_once():
    from redevops_rag.pg_store import PgStore
    table = f"rag_test_{uuid.uuid4().hex[:12]}"
    s = PgStore(_StampEmbedder(), PG_URL, table=table)
    try:
        assert s.get_meta("backend") == "nemotron"   # stamped from the embedder in __init__
        assert s.get_meta("dim") == 32
    finally:
        with s.con.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            cur.execute(f'DROP TABLE IF EXISTS "{table}_meta"')
        s.con.commit(); s.close()


def test_open_pg_store_reconstructs_encoder(monkeypatch):
    from redevops_rag import pg_store as pg_mod
    table = f"rag_test_{uuid.uuid4().hex[:12]}"
    s = pg_mod.PgStore(_StampEmbedder(), PG_URL, table=table)
    s.close()
    built = {}

    def fake_make(backend=None, **kw):
        built["backend"] = backend
        return _StampEmbedder()

    monkeypatch.setattr("redevops_rag.embed.make_embedder", fake_make)
    s2 = pg_mod.open_pg_store(PG_URL, table=table)
    try:
        assert built["backend"] == "nemotron"        # reopened with the stamped encoder
    finally:
        with s2.con.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            cur.execute(f'DROP TABLE IF EXISTS "{table}_meta"')
        s2.con.commit(); s2.close()


def test_document_ids_scope_semantic_and_bm25(store, embedder):
    e = embedder.encode(["alpha microbiome doc", "beta microbiome doc"])
    store.add_chunks([
        {"id": "a", "document_id": "DA", "text": "alpha microbiome doc", "embedding": e[0]},
        {"id": "b", "document_id": "DB", "text": "beta microbiome doc", "embedding": e[1]},
    ], reindex=True)
    sem = store.semantic_search("microbiome", top_k=10, threshold=-1.0, document_ids=["DA"])
    assert sem and all(h["document_id"] == "DA" for h in sem)
    bm = store.bm25_search("microbiome", limit=10, document_ids=["DB"])
    assert bm and all(h["document_id"] == "DB" for h in bm)


def test_query_mode_uses_encode_queries():
    from redevops_rag.pg_store import PgStore
    table = f"rag_test_{uuid.uuid4().hex[:12]}"
    emb = _StampEmbedder()
    s = PgStore(emb, PG_URL, table=table)
    try:
        s.add_chunks([{"id": "a", "document_id": "D", "text": "immune signalling",
                       "embedding": emb.encode(["immune signalling"])[0]}], reindex=True)
        emb.q_calls.clear()
        s.semantic_search("immune", top_k=5, threshold=-1.0, query_mode="instruct")
        assert emb.q_calls == ["immune"]             # asymmetric query side was used
        emb.q_calls.clear()
        s.semantic_search("immune", top_k=5, threshold=-1.0, query_mode="plain")
        assert emb.q_calls == []                     # plain never calls encode_queries
    finally:
        with s.con.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            cur.execute(f'DROP TABLE IF EXISTS "{table}_meta"')
        s.con.commit(); s.close()


# ── explicit retrieval outcomes on Postgres ──────────────────────────────────────────────────────────────────────
def test_pg_lexical_failure_is_reported_not_swallowed_and_does_not_poison_the_connection(store, embedder):
    from redevops_rag.retrieval_result import BackendUnavailable, RetrievalStatus
    from redevops_rag.retrieve import hybrid_search_result

    store.add_chunks(_chunks(embedder, "vitamin d deficiency correlates with poor sleep"))
    healthy = hybrid_search_result(store, "vitamin d deficiency", vector_threshold=-1.0)
    assert healthy.status is RetrievalStatus.COMPLETE

    with store.con.cursor() as cur:                         # the sparse leg's column disappears
        cur.execute(f"ALTER TABLE {store._qtable} DROP COLUMN text_tsv CASCADE")
    store.con.commit()
    with pytest.raises(BackendUnavailable):
        store.bm25_search("vitamin", strict=True)
    assert store.bm25_search("vitamin") == []               # legacy store API unchanged

    r = hybrid_search_result(store, "vitamin d deficiency", vector_threshold=-1.0)
    assert r.status is RetrievalStatus.DEGRADED and r.hits
    assert [l.name for l in r.failed_legs()] == ["bm25"] and "postgresql://" not in (r.failed_legs()[0].error or "")
    # a failed statement must not leave the transaction aborted for the next call
    again = hybrid_search_result(store, "vitamin d deficiency", vector_threshold=-1.0)
    assert again.status is RetrievalStatus.DEGRADED and again.hits


def test_pg_vector_failure_does_not_poison_the_lexical_leg(store, embedder):
    from redevops_rag.retrieval_result import RetrievalStatus
    from redevops_rag.retrieve import hybrid_search_result

    store.add_chunks(_chunks(embedder, "vitamin d deficiency correlates with poor sleep"))
    store.embedder = type("Bad", (), {"dim": 32, "encode": lambda self, t: [[0.0] * 31 for _ in t]})()   # wrong dimension
    r = hybrid_search_result(store, "vitamin d deficiency", vector_threshold=-1.0)
    assert r.status is RetrievalStatus.DEGRADED and r.hits and [l.name for l in r.failed_legs()] == ["vector"]


# ── tenant-aware mode on Postgres (same contract as the DuckDB store) ─────────────────────────────────────────────
@pytest.fixture
def tenant_store(embedder):
    from redevops_rag.pg_store import PgStore

    table = f"rag_tenant_{uuid.uuid4().hex[:12]}"
    s = PgStore(embedder, PG_URL, table=table, tenant_aware=True)
    yield s
    with s.con.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        cur.execute(f'DROP TABLE IF EXISTS "{table}_meta"')
    s.con.commit()
    s.close()


def _tenant_rows(embedder, prefix, n, text):
    return [dict(id=f"{prefix}{i}", document_id=f"{prefix}doc{i}", text=text, embedding=embedder.encode([text])[0], chunk_index=0)
            for i in range(n)]


def test_pg_tenant_aware_fails_closed_without_a_valid_scope(tenant_store, embedder):
    from redevops_rag.retrieval_result import RetrievalStatus
    from redevops_rag.retrieve import hybrid_search_result
    from redevops_rag.tenancy import TenantScopeError

    with pytest.raises(TenantScopeError):
        tenant_store.add_chunks(_tenant_rows(embedder, "x", 1, "budget forecast"))
    for bad in (None, "", "*", "a::b", 5):
        r = hybrid_search_result(tenant_store, "budget forecast", tenant_scope=bad)
        assert r.status is RetrievalStatus.DENIED and r.legs == () and r.isolation == "tenant_aware"
        with pytest.raises(TenantScopeError):
            tenant_store.bm25_search("budget", tenant_scope=bad)


def test_pg_tenant_filter_applies_before_limit_in_both_legs_and_ids_do_not_collide(tenant_store, embedder):
    from redevops_rag.retrieve import hybrid_search

    # tenant-A: 20 strong matches that outrank tenant-B's single weaker one; same ids/doc ids across tenants
    tenant_store.add_chunks(_tenant_rows(embedder, "r", 20, "quarterly budget forecast alpha alpha"), tenant_scope="tenant-A")
    tenant_store.add_chunks(_tenant_rows(embedder, "r", 1, "quarterly budget forecast alpha alpha"), tenant_scope="tenant-B")
    assert tenant_store.count(tenant_scope="tenant-A") == 20 and tenant_store.count(tenant_scope="tenant-B") == 1
    for tenant in ("tenant-A", "tenant-B"):
        for hits in (tenant_store.semantic_search("budget forecast", top_k=1, threshold=-1.0, tenant_scope=tenant),
                     tenant_store.bm25_search("budget forecast", limit=1, tenant_scope=tenant),
                     hybrid_search(tenant_store, "budget forecast", limit=1, pool=1, tenant_scope=tenant, vector_threshold=-1.0)):
            assert len(hits) == 1 and hits[0]["chunk_id"].startswith(tenant + "::")
    wide = hybrid_search(tenant_store, "budget", limit=100, pool=100, tenant_scope="tenant-B", vector_threshold=-1.0)
    assert {h["chunk_id"] for h in wide} == {"tenant-B::r0"}


def test_pg_prune_and_document_listing_are_confined_to_the_tenant(tenant_store, embedder):
    rows = lambda ids: [dict(id=i, document_id="/data/doc.md", text="alpha " + i, embedding=embedder.encode(["alpha " + i])[0],
                             chunk_index=0) for i in ids]
    tenant_store.add_chunks(rows(["k1", "k2"]), tenant_scope="tenant-A")
    tenant_store.add_chunks(rows(["k1"]), tenant_scope="tenant-B")
    assert tenant_store.document_ids("/data", tenant_scope="tenant-A") == ["/data/doc.md"]
    assert tenant_store.prune_document("/data/doc.md", ["k1"], tenant_scope="tenant-A") == 1
    assert tenant_store.prune_document("/data/doc.md", [], tenant_scope="tenant-A") == 1
    assert tenant_store.count(tenant_scope="tenant-B") == 1


def test_pg_a_row_owned_by_another_scope_is_never_replaced(tenant_store, embedder):
    from redevops_rag.tenancy import qualify_id

    tenant_store.add_chunks(_tenant_rows(embedder, "r", 1, "alpha original"), tenant_scope="tenant-A")
    with tenant_store.con.cursor() as cur:                    # corrupt state: the stored id now belongs to another scope
        cur.execute(f"UPDATE {tenant_store._qtable} SET tenant_scope = 'tenant-X'")
    tenant_store.con.commit()
    from redevops_rag.tenancy import TenantScopeError
    with pytest.raises(TenantScopeError):
        tenant_store.add_chunks(_tenant_rows(embedder, "r", 1, "re-ingest by A"), tenant_scope="tenant-A")
    with tenant_store.con.cursor() as cur:
        cur.execute(f"SELECT text, tenant_scope FROM {tenant_store._qtable} WHERE id = %s", [qualify_id("tenant-A", "r0")])
        assert cur.fetchone() == ("alpha original", "tenant-X")
