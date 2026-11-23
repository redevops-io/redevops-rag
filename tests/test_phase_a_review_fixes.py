"""Fixes from the bounded independent review of the retrieval entry points Phase A did not touch."""
import logging
import re
import sys
import types

import pytest

from redevops_rag import cli
from redevops_rag.rag import RAG
from redevops_rag.retrieval_result import RetrievalStatus
from redevops_rag.retrieve import hybrid_search, hybrid_search_result
from redevops_rag.store import Store
from redevops_rag.temporal import TemporalReasoningRetriever
from redevops_rag.tenancy import TenantScopeError

DIM = 32


def _h(w):
    h = 5381
    for ch in w:
        h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return h


class Emb:
    dim, sim_floor = DIM, 0.0

    def __init__(self, *_a, **_k):
        pass

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


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr("redevops_rag.rag.Embedder", Emb)


def _corpus(tmp_path, name, text):
    d = tmp_path / name
    d.mkdir()
    (d / "note.md").write_text(text)
    return str(d)


def _fts_or_skip(db):
    s = Store(Emb(), db)
    ok = s._fts
    s.close()
    if not ok:
        pytest.skip("DuckDB fts extension unavailable")


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cli_tenant_aware_index_and_search_are_confined_and_fail_closed(tmp_path, capsys):
    db = str(tmp_path / "t.duckdb")
    _fts_or_skip(db)
    a, b = _corpus(tmp_path, "a", "quarterly budget forecast alpha"), _corpus(tmp_path, "b", "budget notes for beta only")
    assert cli.main(["--db", db, "--tenant-aware", "--tenant-scope", "tenant-A", "index", a]) == 0
    assert cli.main(["--db", db, "--tenant-aware", "--tenant-scope", "tenant-B", "index", b]) == 0
    capsys.readouterr()
    assert cli.main(["--db", db, "--tenant-aware", "--tenant-scope", "tenant-B", "search", "budget"]) == 0
    out = capsys.readouterr().out
    assert "note.md" in out and "alpha" not in out and "beta" in out
    # tenant-aware without an identity: refused, nothing printed to stdout
    assert cli.main(["--db", db, "--tenant-aware", "search", "budget"]) == 2
    assert cli.main(["--db", db, "--tenant-aware", "index", a]) == 2
    assert capsys.readouterr().out == ""


def test_cli_refuses_to_open_tenant_data_unscoped(tmp_path, capsys):
    db = str(tmp_path / "t.duckdb")
    _fts_or_skip(db)
    a = _corpus(tmp_path, "a", "quarterly budget forecast alpha")
    assert cli.main(["--db", db, "--tenant-aware", "--tenant-scope", "tenant-A", "index", a]) == 0
    capsys.readouterr()
    for argv in (["search", "budget"], ["index", a], ["ask", "budget"]):
        assert cli.main(["--db", db, *argv]) == 2                   # the historical unscoped CLI path used to return A's rows
    cap = capsys.readouterr()
    assert "tenant-scoped chunks" in cap.err and cap.out == ""


def test_cli_unscoped_use_of_an_untagged_corpus_is_unchanged(tmp_path, capsys):
    db = str(tmp_path / "plain.duckdb")
    _fts_or_skip(db)
    a = _corpus(tmp_path, "a", "quarterly budget forecast alpha")
    assert cli.main(["--db", db, "index", a]) == 0
    capsys.readouterr()
    assert cli.main(["--db", db, "search", "budget"]) == 0 and "note.md" in capsys.readouterr().out


def test_cli_does_not_report_no_matches_when_the_backend_is_down(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "plain.duckdb")
    _fts_or_skip(db)
    a = _corpus(tmp_path, "a", "quarterly budget forecast alpha")
    assert cli.main(["--db", db, "index", a]) == 0
    capsys.readouterr()
    def boom(*a, **k):
        raise RuntimeError("backend down")
    monkeypatch.setattr(Store, "semantic_search", boom)
    monkeypatch.setattr(Store, "bm25_search", boom)
    assert cli.main(["--db", db, "search", "budget"]) == 3
    cap = capsys.readouterr()
    assert "no matches" not in cap.out and "unavailable" in cap.err
    monkeypatch.undo()
    monkeypatch.setattr("redevops_rag.rag.Embedder", Emb)
    monkeypatch.setattr(Store, "semantic_search", boom)
    assert cli.main(["--db", db, "search", "zzzznomatch"]) == 0           # one leg down: degraded, warns, still exits 0
    assert "degraded" in capsys.readouterr().err


# ── RAG facade ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_rag_search_result_exposes_status_and_denies_without_scope(tmp_path):
    rag = RAG(db_path=str(tmp_path / "r.duckdb"), tenant_aware=True)
    if not rag.store._fts:
        pytest.skip("fts unavailable")
    d = _corpus(tmp_path, "a", "quarterly budget forecast alpha")
    rag.index(d, tenant_scope="tenant-A")
    assert rag.search_result("budget").status is RetrievalStatus.DENIED
    r = rag.search_result("budget", tenant_scope="tenant-A")
    assert r.status in (RetrievalStatus.COMPLETE, RetrievalStatus.DEGRADED) and r.hits and r.isolation == "tenant_aware"
    assert rag.search_result("zzzznomatch", tenant_scope="tenant-A", vector_threshold=0.99).status is RetrievalStatus.EMPTY
    with pytest.raises(TenantScopeError):
        rag.ask("budget")                                   # ask forwards scope too, and fails closed without one


def test_rag_ask_error_text_is_redacted(tmp_path, monkeypatch):
    rag = RAG(db_path=str(tmp_path / "r.duckdb"))
    class Boom:
        def __init__(self, **k): pass
        chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(
            RuntimeError("401 from https://svc:s3cr3t-pw@llm.internal/v1 (api_key=sk-abc123 password=hunter2)"))))
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Boom))
    out = rag.ask("anything", base_url="https://x", api_key="k")
    assert out["answer"] is None
    assert "s3cr3t-pw" not in out["error"] and "hunter2" not in out["error"] and "sk-abc123" not in out["error"]


# ── TemporalReasoningRetriever ───────────────────────────────────────────────────────────────────────────────────
def test_temporal_retriever_forwards_scope_to_search_and_insert():
    s = Store(Emb(), ":memory:", tenant_aware=True)
    if not s._fts:
        pytest.skip("fts unavailable")
    t = TemporalReasoningRetriever()
    with pytest.raises(TenantScopeError):
        t.insert(s, [{"text": "budget forecast alpha", "document_id": "d1"}])
    assert t.insert(s, [{"text": "budget forecast alpha", "document_id": "d1"}], tenant_scope="tenant-A") == 1
    assert t.insert(s, [{"text": "budget notes beta", "document_id": "d2"}], tenant_scope="tenant-B") == 1
    with pytest.raises(TenantScopeError):
        t.search(s, "budget")
    for use_llm in (False, True):
        tr = TemporalReasoningRetriever(reason_llm=(lambda system, user: "1. budget") if use_llm else None)
        hits = tr.search(s, "budget", tenant_scope="tenant-B")
        assert hits and all(h["chunk_id"].startswith("tenant-B::") for h in hits)


# ── reranker failure ─────────────────────────────────────────────────────────────────────────────────────────────
def test_a_failing_reranker_is_a_reported_degradation_not_a_crash_or_silence(caplog):
    s = Store(Emb(), ":memory:")
    if not s._fts:
        pytest.skip("fts unavailable")
    s.add_chunks([{"id": "x", "document_id": "d", "text": "budget forecast", "embedding": Emb().encode(["budget forecast"])[0]}])
    s.reindex_fts()
    class BadReranker:
        def rerank(self, q, hits):
            raise TimeoutError("rerank endpoint https://u:p4ss@nim.internal timed out")
    r = hybrid_search_result(s, "budget forecast", reranker=BadReranker(), vector_threshold=-1.0)
    assert r.status is RetrievalStatus.DEGRADED and r.hits
    leg = [l for l in r.failed_legs() if l.name == "rerank"][0]
    assert "p4ss" not in leg.error and "TimeoutError" in leg.error and any("un-reranked" in n for n in r.notes)
    with caplog.at_level(logging.WARNING, logger="redevops_rag.retrieve"):
        assert hybrid_search(s, "budget forecast", reranker=BadReranker(), vector_threshold=-1.0)
    assert any("rerank leg failed" in m for m in caplog.messages)
    ok = type("Ok", (), {"rerank": lambda self, q, hits: list(reversed(hits))})()
    assert hybrid_search_result(s, "budget forecast", reranker=ok, vector_threshold=-1.0).status is RetrievalStatus.COMPLETE


# ── FTS reindex failure ──────────────────────────────────────────────────────────────────────────────────────────
def test_a_failed_fts_reindex_is_logged_and_remembered(caplog):
    s = Store(Emb(), ":memory:")
    if not s._fts:
        pytest.skip("fts unavailable")
    real = s.con
    class Failing:
        def __getattr__(self, n): return getattr(real, n)
        def execute(self, sql, *a, **k):
            if "create_fts_index" in sql:
                raise RuntimeError("disk full at /var/lib/x?password=hunter2")
            return real.execute(sql, *a, **k)
    s.con = Failing()
    with caplog.at_level(logging.WARNING, logger="redevops_rag.store"):
        s.reindex_fts()
    assert s.fts_index_error and "hunter2" not in s.fts_index_error and any("FTS reindex failed" in m for m in caplog.messages)
    s.con = real
    s.reindex_fts()
    assert s.fts_index_error is None
