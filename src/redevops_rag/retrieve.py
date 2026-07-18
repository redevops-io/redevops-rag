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
    query_mode: str = "auto",
) -> list[dict[str, Any]]:
    """Vector + BM25 → RRF → recency/keyword boosts → optional cross-encoder rerank.

    ``pool`` candidates are fused/boosted; the top ``pool`` then go to the reranker (if
    given) which returns the final ``limit``. ``document_ids`` (optional) scopes the
    search to a document subset — used to build graduated-pollution candidate pools.
    ``query_mode`` controls the vector leg's query encoding for asymmetric encoders
    (``instruct``/``plain``/``auto``) — inert for a symmetric encoder like bge.
    """
    if not query or not query.strip():
        return []
    try:
        vector_hits = store.semantic_search(query, top_k=pool, threshold=vector_threshold,
                                            document_ids=document_ids, query_mode=query_mode)
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


# ─────────────────────────── DIVER-style reasoning retrieval ───────────────────────────
# arXiv:2508.07995 (DIVER): LLM query expansion → hybrid retrieve (union) → LLM listwise
# rerank. A drop-in, stronger replacement for single-query BM25/hybrid on reasoning-
# intensive / temporal queries. Measured on TEMPO/workplace (64.7k docs): NDCG@10 0.197
# (BM25) → 0.300 (hybrid) → 0.448 (DIVER); Recall@10 0.245 → 0.422 → 0.615.

_ID = lambda h: h.get("document_id") or h.get("id")  # noqa: E731


def _expand_query(reason_llm, query: str, n: int) -> list[str]:
    """Decompose a query into ≤n focused sub-queries (reasoning-aware, temporal-aware)."""
    out = reason_llm(
        f"Decompose the query into {n} focused sub-queries covering the entities and the "
        "relevant time periods needed to answer it. One per line, no numbering.",
        query[:1500],
    ) or ""
    subs = [ln.strip("-•* ").strip() for ln in out.splitlines() if ln.strip()]
    return subs[:n]


def _listwise_rerank(reason_llm, query: str, cands: list[dict], limit: int) -> list[dict]:
    """LLM listwise rerank: score the candidate list jointly, return the top `limit`."""
    if len(cands) <= limit:
        return cands
    snippets = "\n".join(f"[{i}] {(c.get('text') or '')[:280]}" for i, c in enumerate(cands))
    out = reason_llm(
        f"Rank the passages by usefulness for the query. Return the {limit} best passage "
        "numbers, comma-separated, best first. Numbers only.",
        f"Query: {query[:800]}\n\nPassages:\n{snippets}",
    ) or ""
    order = [int(x) for x in re.findall(r"\d+", out)]
    seen, ranked = set(), []
    for i in order:
        if 0 <= i < len(cands) and i not in seen:
            seen.add(i)
            ranked.append(cands[i])
    for i, c in enumerate(cands):  # stable fallback: keep any the model didn't list
        if i not in seen:
            ranked.append(c)
    return ranked[:limit]


def diver_search(
    store: Store,
    query: str,
    reason_llm,
    limit: int = 8,
    pool: int = 25,
    n_subqueries: int = 3,
    recency_half_life_days: float = 0.0,
    reranker: Optional["Reranker"] = None,  # noqa: F821
    document_ids: list | None = None,
) -> list[dict[str, Any]]:
    """DIVER-style reasoning-intensive retrieval.

    ``reason_llm(system, user) -> str`` supplies the reasoning model for query expansion
    and listwise reranking. Pipeline: expand → hybrid retrieve each sub-query → dedup union
    → (optional cross-encoder) → LLM listwise rerank → top ``limit``. With ``reason_llm``
    None it degrades to plain :func:`hybrid_search`, so it is a safe drop-in replacement.

    Encoder-aware query construction: the reasoning-heavy ORIGINAL query is embedded with the
    encoder's instruction side (``query_mode='instruct'``), while the expanded sub-query
    fragments go in ``plain`` — the instruction is tuned for the full reasoning query and *hurts*
    the fragments, which is why DIVER over a reasoning encoder (ReasonIR/Nemotron) regressed vs
    bge. On a symmetric encoder (bge) both modes are plain ``encode``, so DIVER-over-bge — the
    recommended/cheapest config — stays byte-identical.
    """
    if not query or not query.strip():
        return []
    if reason_llm is None:
        return hybrid_search(store, query, limit=limit, pool=pool, document_ids=document_ids,
                             recency_half_life_days=recency_half_life_days, reranker=reranker)
    cand: dict[Any, dict] = {}
    expanded = [(query, "instruct")] + [(s, "plain") for s in _expand_query(reason_llm, query, n_subqueries)]
    for q, mode in expanded:
        for h in hybrid_search(store, q, limit=pool, pool=pool, document_ids=document_ids,
                               recency_half_life_days=recency_half_life_days, query_mode=mode):
            cand.setdefault(_ID(h), h)
    candidates = list(cand.values())
    if reranker is not None and candidates:
        candidates = reranker.rerank(query, candidates)
    return _listwise_rerank(reason_llm, query, candidates, limit)
