"""Hybrid retrieval: vector + BM25 fused via Reciprocal Rank Fusion, then recency/keyword
priors, then an optional cross-encoder rerank.

Faithful to the pipeline in redevops-io/rag-saas-platform's `vector_database.py`
(RRF k=60, recency half-life 90d, keyword boost 0.05/term capped at 1.5), but decoupled
from the multi-tenant workspace/SaaS shell — it operates on a single :class:`Store`.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid importing duckdb just to type-hint
    from .store import Store


def rrf_fuse(rankings: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal-rank-fusion: ``score = Σ 1 / (k + rank)`` over each input ranking.

    Keyed by ``chunk_id`` (falls back to ``filename::chunk_index``). Rank is 0-based, so
    the top of each list contributes ``1/(k+0)``.
    """
    scores: dict[str, float] = {}
    cache: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            key = item.get("chunk_id") or f"{item.get('filename')}::{item.get('chunk_index')}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in cache:
                cache[key] = dict(item)
            else:
                for k2, v2 in item.items():
                    cache[key].setdefault(k2, v2)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[dict[str, Any]] = []
    for key, score in ordered:
        row = cache[key]
        row["rrf_score"] = score
        out.append(row)
    return out


def _apply_score_boosts(
    query: str,
    candidates: list[dict[str, Any]],
    recency_half_life_days: float,
    keyword_boost_per_term: float,
    keyword_boost_cap: float,
) -> list[dict[str, Any]]:
    """Recency (exponential decay by half-life) + keyword priors on RRF-scored candidates."""
    if not candidates:
        return candidates
    do_recency = recency_half_life_days > 0
    do_keyword = keyword_boost_per_term > 0
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2} if do_keyword else set()
    now = datetime.now(timezone.utc)

    for c in candidates:
        boost = 1.0
        if do_recency:
            ts = c.get("created_at")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    ts = None
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                boost *= 0.5 ** (age_days / recency_half_life_days)
        if do_keyword and terms:
            text = (c.get("text") or "").lower()
            hits = sum(1 for t in terms if t in text)
            boost *= min(1.0 + keyword_boost_per_term * hits, keyword_boost_cap)
        c["boosted_score"] = c.get("rrf_score", 0.0) * boost
    candidates.sort(key=lambda r: r.get("boosted_score", 0.0), reverse=True)
    return candidates


def hybrid_search(
    store: Store,
    query: str,
    limit: int = 8,
    pool: int = 50,
    vector_threshold: float = 0.4,
    recency_half_life_days: float = 90.0,
    keyword_boost_per_term: float = 0.05,
    keyword_boost_cap: float = 1.5,
    reranker: Optional["Reranker"] = None,  # noqa: F821
    document_ids: list | None = None,
) -> list[dict[str, Any]]:
    """Vector + BM25 → RRF → recency/keyword boosts → optional cross-encoder rerank.

    ``pool`` candidates are fused/boosted; the top ``pool`` then go to the reranker (if
    given) which returns the final ``limit``. ``document_ids`` (optional) scopes the
    search to a document subset — used to build graduated-pollution candidate pools.
    """
    if not query or not query.strip():
        return []
    try:
        vector_hits = store.semantic_search(query, top_k=pool, threshold=vector_threshold,
                                            document_ids=document_ids)
    except Exception:
        vector_hits = []
    try:
        bm25_hits = store.bm25_search(query, limit=pool, document_ids=document_ids)
    except Exception:
        bm25_hits = []
    if not vector_hits and not bm25_hits:
        return []
    fused = rrf_fuse([vector_hits, bm25_hits])
    fused = _apply_score_boosts(
        query, fused, recency_half_life_days, keyword_boost_per_term, keyword_boost_cap
    )
    fused = fused[:pool]
    if reranker is not None and fused:
        fused = reranker.rerank(query, fused)
    return fused[:limit]
