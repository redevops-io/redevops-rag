"""RAG retrieval fan-out (LLM-parallelization audit, P0/P1) — opt-in, order-preserving, result-identical.
Default (REDEVOPS_RAG_CONCURRENCY unset) is the historical serial path."""
from __future__ import annotations

import time

from redevops_rag import retrieve
from redevops_rag._parallel import run_parallel


def test_run_parallel_serial_by_default(monkeypatch):
    monkeypatch.delenv("REDEVOPS_RAG_CONCURRENCY", raising=False)
    assert run_parallel([lambda: 1, lambda: 2, lambda: 3]) == [1, 2, 3]


def test_run_parallel_preserves_order_and_overlaps(monkeypatch):
    monkeypatch.setenv("REDEVOPS_RAG_CONCURRENCY", "3")

    def slow(v):
        return lambda: (time.sleep(0.05), v)[1]

    t0 = time.perf_counter()
    out = run_parallel([slow(1), slow(2), slow(3)])
    assert out == [1, 2, 3]
    assert time.perf_counter() - t0 < 0.12          # 3×0.05=0.15s serial → overlapped to ~0.05s


class _FakeStore:
    """Two independent legs, each with a fixed latency, returning dict hits keyed by document_id."""

    def semantic_search(self, query, **kw):
        time.sleep(0.04)
        return [{"document_id": "d1", "id": "d1", "score": 0.9, "text": "alpha", "created_at": None},
                {"document_id": "d2", "id": "d2", "score": 0.7, "text": "beta", "created_at": None}]

    def bm25_search(self, query, **kw):
        time.sleep(0.04)
        return [{"document_id": "d2", "id": "d2", "score": 5.0, "text": "beta", "created_at": None},
                {"document_id": "d3", "id": "d3", "score": 4.0, "text": "gamma", "created_at": None}]


def _ids(hits):
    return [h["document_id"] for h in hits]


def test_hybrid_legs_concurrent_equal_serial(monkeypatch):
    st = _FakeStore()
    monkeypatch.delenv("REDEVOPS_RAG_CONCURRENCY", raising=False)
    serial = _ids(retrieve.hybrid_search(st, "query", limit=5, pool=5))

    monkeypatch.setenv("REDEVOPS_RAG_CONCURRENCY", "2")
    parallel = _ids(retrieve.hybrid_search(st, "query", limit=5, pool=5))

    assert serial == parallel and len(serial) > 0   # identical RRF-fused ranking, order and all


def test_hybrid_legs_overlap(monkeypatch):
    st = _FakeStore()
    monkeypatch.delenv("REDEVOPS_RAG_CONCURRENCY", raising=False)
    t0 = time.perf_counter(); retrieve.hybrid_search(st, "q", limit=5, pool=5); serial = time.perf_counter() - t0
    monkeypatch.setenv("REDEVOPS_RAG_CONCURRENCY", "2")
    t0 = time.perf_counter(); retrieve.hybrid_search(st, "q", limit=5, pool=5); parallel = time.perf_counter() - t0
    assert parallel < serial * 0.75, f"serial={serial:.3f}s parallel={parallel:.3f}s"  # 2 legs overlap
