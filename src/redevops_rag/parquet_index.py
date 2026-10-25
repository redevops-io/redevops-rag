"""Parquet-backed retrieval: BM25 over a Polars ``scan_parquet`` plus a :class:`ChangeClosureGraph` loaded from an edge table.

For corpora that live as Parquet (locally or on S3) rather than in the DuckDB store. Everything here is a pure function of
LazyFrames, so projection and predicate pushdown apply and nothing is loaded that the query does not touch. ``polars`` is the
optional ``[parquet]`` extra; importing this module without it raises a clear error.

Tables (columns beyond these are ignored):
* chunks: ``doc`` (str), ``node`` (str, unique within ``doc``), ``text`` (str)
* edges:  ``doc`` (str), ``src`` (str), ``dst`` (str), ``kind`` (str; e.g. xref, term)   [src depends on dst]

Node ids handed to the graph are ``f"{doc}#{node}"`` so one graph can span documents.

This is a primitive, not a policy: how to compose content ranking and closure (content-led vs closure-first) is the caller's
decision, as in :mod:`redevops_rag.citation_graph`.
"""
from __future__ import annotations

from typing import Iterable, Optional

try:
    import polars as pl
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("redevops_rag.parquet_index needs polars: pip install 'redevops-rag[parquet]'") from e

from .citation_graph import ChangeClosureGraph, Node

_TOKEN = r"[a-z0-9]+"


def node_id(doc: str, node: str) -> str:
    return f"{doc}#{node}"


def scan(path: str, *, storage_options: Optional[dict] = None) -> pl.LazyFrame:
    """``pl.scan_parquet`` over a file, glob or ``s3://`` prefix (credentials via ``storage_options`` or the environment)."""
    return pl.scan_parquet(path, storage_options=storage_options)


def _tokens(col: str) -> pl.Expr:
    return pl.col(col).str.to_lowercase().str.extract_all(_TOKEN)


def bm25_search(chunks: pl.LazyFrame, query: str, k: int = 10, *, k1: float = 1.5, b: float = 0.75,
                docs: Optional[Iterable[str]] = None) -> list[str]:
    """Top-``k`` node ids (``doc#node``) by BM25, computed with Polars expressions. ``docs`` restricts the corpus first (a
    predicate that Parquet scans can push down), so statistics are those of the restricted corpus."""
    q = sorted(set(_tokens_py(query)))
    if not q:
        return []
    lf = chunks.select("doc", "node", "text")
    if docs is not None:
        lf = lf.filter(pl.col("doc").is_in(list(docs)))
    lf = lf.with_columns(_tokens("text").alias("tok")).with_columns(pl.col("tok").list.len().alias("dl"))
    n = lf.select(pl.len()).collect().item()
    if n == 0:
        return []
    avg = lf.select(pl.col("dl").mean()).collect().item() or 1.0
    exploded = lf.select("doc", "node", "dl", pl.col("tok")).explode("tok").filter(pl.col("tok").is_in(q))
    tf = exploded.group_by("doc", "node", "dl", "tok").agg(pl.len().alias("tf"))
    df = tf.group_by("tok").agg(pl.len().alias("df"))
    scored = (tf.join(df, on="tok")
              .with_columns((1 + (n - pl.col("df") + 0.5) / (pl.col("df") + 0.5)).log().alias("idf"))
              .with_columns((pl.col("idf") * pl.col("tf") * (k1 + 1)
                             / (pl.col("tf") + k1 * (1 - b + b * pl.col("dl") / avg))).alias("s"))
              .group_by("doc", "node").agg(pl.col("s").sum())
              .sort(["s", "doc", "node"], descending=[True, False, False]).head(k).collect())
    return [node_id(d, nd) for d, nd in scored.select("doc", "node").iter_rows()]


def _tokens_py(s: str) -> list[str]:
    import re
    return re.findall(_TOKEN, s.lower())


def load_graph(chunks: pl.LazyFrame, edges: pl.LazyFrame, *, kinds: Optional[Iterable[str]] = None,
               docs: Optional[Iterable[str]] = None) -> ChangeClosureGraph:
    """Materialise (only) the requested documents into a :class:`ChangeClosureGraph`. Edges to unknown nodes are dropped by the
    graph; ``kinds`` filters edge kinds (e.g. exclude noisy ``term`` edges)."""
    c = chunks.select("doc", "node", "text")
    e = edges.select("doc", "src", "dst", "kind")
    if docs is not None:
        ds = list(docs)
        c, e = c.filter(pl.col("doc").is_in(ds)), e.filter(pl.col("doc").is_in(ds))
    if kinds is not None:
        e = e.filter(pl.col("kind").is_in(list(kinds)))
    nodes = [Node(node_id(d, n), t or "") for d, n, t in c.collect().iter_rows()]
    cites: dict[str, set[str]] = {}
    for d, s, t in e.select("doc", "src", "dst").unique().collect().iter_rows():
        cites.setdefault(node_id(d, s), set()).add(node_id(d, t))
    return ChangeClosureGraph(nodes, cites)


def dependency_order(graph: ChangeClosureGraph, seed: str, hops: int = 2) -> list[str]:
    """Forward closure: what ``seed`` itself depends on (what governs it), hop-ordered, deterministic. The mirror image of
    :func:`redevops_rag.citation_graph.citation_closure`, which finds what is affected *by* editing the seed."""
    order, seen, frontier = [], {seed}, [seed]
    for _ in range(max(0, hops)):
        found = sorted({t for n in frontier for t in graph.cites.get(n, ()) if t not in seen},
                       key=lambda x: (-graph.in_degree(x), x))
        seen.update(found); order.extend(found); frontier = found
        if not frontier:
            break
    return order


def content_led(graph: ChangeClosureGraph, seed: str, content_order: list[str], budget: int, *, hops: int = 2,
                forward: bool = True, reverse: bool = True) -> list[str]:
    """Content ranking first, then the seed's closure (forward = what governs it, reverse = what it affects), filled under a
    token ``budget``. The seed is always first and counts against the budget."""
    from .citation_graph import _budget_fill, _closure_order
    rest = list(content_order)
    structural = []
    if forward:
        structural += dependency_order(graph, seed, hops)
    if reverse:
        structural += _closure_order(graph.cited_by, graph._in_degree, seed, hops)
    order = [x for x in dict.fromkeys([*rest, *structural]) if x in graph.tokens and x != seed]
    spent = graph.tokens.get(seed, 0)
    return [seed, *_budget_fill(order, graph.tokens, max(0, budget - spent), seed)]
