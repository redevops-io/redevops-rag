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
