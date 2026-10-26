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


# ───────────────────────── embeddings: dense, hybrid, HippoRAG-style associative graph ─────────────────────────
# Embeddings live in the same Parquet as the text, as a ``embedding`` list<float32> column (L2-normalised, so cosine = dot).
# DuckDB writes them (``write_chunks_with_embeddings``, local or s3:// via httpfs) and can search them in place
# (``duckdb_dense_search``); Polars/numpy do the in-memory variants. Nothing here loads a model: callers pass an ``embedder`` with
# ``encode(list[str]) -> list[list[float]]`` (e.g. :class:`redevops_rag.embed.Embedder`) or precomputed vectors.

def _np():
    import numpy as np
    return np


def write_chunks_with_embeddings(chunks: "pl.DataFrame", uri: str, embedder, *, text_col: str = "text", batch: int = 256,
                                 s3_credential_chain: bool = True) -> str:
    """Embed ``chunks[text_col]`` and write ``chunks + embedding`` to ``uri`` with DuckDB (``COPY ... TO ... (FORMAT PARQUET)``).
    ``uri`` may be a local path or ``s3://bucket/key.parquet`` (httpfs + the AWS credential chain; nothing is read or written
    remotely unless the URI says so)."""
    import duckdb
    np = _np()
    texts = chunks[text_col].to_list()
    vecs = []
    for i in range(0, len(texts), batch):
        vecs += list(embedder.encode(texts[i:i + batch]))
    m = np.asarray(vecs, dtype="float32") if vecs else np.zeros((0, 1), dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.where(norms == 0, 1, norms)
    df = chunks.with_columns(pl.Series("embedding", m.tolist(), dtype=pl.List(pl.Float32)))
    con = duckdb.connect()
    import os
    if uri.startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        if s3_credential_chain:
            con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain)")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(uri)), exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as td:               # Polars -> Parquet -> DuckDB COPY: no pyarrow dependency
        stage = os.path.join(td, "stage.parquet")
        df.write_parquet(stage)
        con.execute(f"COPY (SELECT * FROM read_parquet('{stage}')) TO '{uri}' (FORMAT PARQUET)")
    return uri


def _embedding_matrix(chunks: "pl.LazyFrame", docs: Optional[Iterable[str]]):
    lf = chunks.select("doc", "node", "embedding")
    if docs is not None:
        lf = lf.filter(pl.col("doc").is_in(list(docs)))
    df = lf.collect()
    np = _np()
    if df.height == 0:
        return [], np.zeros((0, 1), dtype="float32")
    return [node_id(d, n) for d, n in df.select("doc", "node").iter_rows()], np.asarray(df["embedding"].to_list(), dtype="float32")


def dense_search(chunks: "pl.LazyFrame", qvec, k: int = 10, *, docs: Optional[Iterable[str]] = None, floor: float = 0.0) -> list[str]:
    """Top-``k`` ids by cosine to ``qvec`` (vectors are stored normalised). Ties break on id for determinism."""
    np = _np()
    ids, m = _embedding_matrix(chunks, docs)
    if not ids:
        return []
    q = np.asarray(qvec, dtype="float32"); q = q / (np.linalg.norm(q) or 1.0)
    sims = m @ q
    order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))
    return [ids[i] for i in order[:k] if sims[i] >= floor]


def duckdb_dense_search(path: str, qvec, k: int = 10, *, docs: Optional[Iterable[str]] = None) -> list[str]:
    """The same ranking computed by DuckDB over the Parquet in place (local, glob or s3://), using ``list_cosine_similarity``."""
    import duckdb
    con = duckdb.connect()
    if path.startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs; CREATE SECRET (TYPE s3, PROVIDER credential_chain)")
    where, params = "", [list(map(float, qvec)), k]
    if docs is not None:
        ds = list(docs)
        where = f"WHERE doc IN ({','.join('?' * len(ds))})"; params = [list(map(float, qvec)), *ds, k]
    rows = con.execute(f"SELECT doc, node FROM read_parquet('{path}') {where} "
                       f"ORDER BY list_cosine_similarity(embedding, ?::FLOAT[]) DESC, doc, node LIMIT ?",
                       [params[0], *params[1:]] if where == "" else params).fetchall()
    return [node_id(d, n) for d, n in rows]


def rrf_ids(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion over id rankings (the B3 'hybrid' arm: BM25 + dense)."""
    score: dict[str, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: (-score[i], i))


def hybrid_search(chunks: "pl.LazyFrame", query: str, qvec, k: int = 10, *, docs: Optional[Iterable[str]] = None, pool: int = 100) -> list[str]:
    return rrf_ids([bm25_search(chunks, query, k=pool, docs=docs), dense_search(chunks, qvec, k=pool, docs=docs)])[:k]


def simgraph_ppr_search(chunks: "pl.LazyFrame", qvec, k: int = 10, *, docs: Optional[Iterable[str]] = None, knn: int = 10,
                        alpha: float = 0.5, iters: int = 30, seeds: int = 5) -> list[str]:
    """HippoRAG-style associative retrieval: a similarity graph (each node linked to its ``knn`` nearest neighbours by embedding),
    personalised PageRank restarting on the ``seeds`` nodes closest to the query. This is the *similarity-derived* graph the
    redevops.io posts report does not recover dependencies (it cannot represent 'A cites B'); it is here so that claim can be
    re-measured on any corpus. It is NOT the full HippoRAG pipeline (no LLM OpenIE entity graph); see context_runtime's
    ``HippoRAGRetriever`` for that."""
    np = _np()
    ids, m = _embedding_matrix(chunks, docs)
    n = len(ids)
    if n == 0:
        return []
    q = np.asarray(qvec, dtype="float32"); q = q / (np.linalg.norm(q) or 1.0)
    qs = m @ q
    sim = m @ m.T
    np.fill_diagonal(sim, -1.0)
    a = np.zeros((n, n), dtype="float32")
    kk = min(knn, max(n - 1, 0))
    if kk:
        nbr = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        for i in range(n):
            a[i, nbr[i]] = np.clip(sim[i, nbr[i]], 0, None)
    a = np.maximum(a, a.T)
    deg = a.sum(axis=1, keepdims=True)
    p = np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)
    top = np.argsort(-qs)[:seeds]
    r0 = np.zeros(n, dtype="float32"); r0[top] = np.clip(qs[top], 1e-6, None); r0 /= r0.sum()
    r = r0.copy()
    for _ in range(iters):
        r = (1 - alpha) * r0 + alpha * (p.T @ r)
    order = sorted(range(n), key=lambda i: (-float(r[i]), ids[i]))
    return [ids[i] for i in order[:k]]
