"""Bounded, order-preserving parallel execution for independent RAG retrieval work (LLM-parallelization
audit, P0/P1). hybrid_search runs independent vector + BM25 legs; diver_search runs N independent sub-query
retrievals. Both evaluate serially today. ``run_parallel`` overlaps them when opted in via
``REDEVOPS_RAG_CONCURRENCY`` (default 1 = unchanged serial), preserving order so RRF fusion and the
first-occurrence candidate dedup are identical to the serial path."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def rag_concurrency() -> int:
    try:
        return max(1, int(os.getenv("REDEVOPS_RAG_CONCURRENCY", "1")))
    except ValueError:
        return 1


def run_parallel(thunks) -> list:
    """Run 0-arg callables, results in input order. Serial when concurrency<=1 (default) or one thunk;
    otherwise a bounded ThreadPool. Thunks must be independent."""
    thunks = list(thunks)
    c = rag_concurrency()
    if c <= 1 or len(thunks) <= 1:
        return [t() for t in thunks]
    with ThreadPoolExecutor(max_workers=min(c, len(thunks))) as pool:
        futures = [pool.submit(t) for t in thunks]
        return [f.result() for f in futures]        # input order preserved
