"""Parquet-backed retrieval (needs the optional polars extra)."""
import pytest

pl = pytest.importorskip("polars")

from redevops_rag.parquet_index import (bm25_search, content_led, dependency_order, load_graph, node_id, scan)  # noqa: E402


@pytest.fixture()
def corpus(tmp_path):
    chunks = pl.DataFrame({
        "doc": ["d1"] * 5 + ["d2"],
        "node": ["4.2", "4.9", "sch4.2", "8.5", "x", "4.2"],
        "text": ["Seller represents compliance with applicable laws subject to Section 4.9",
                 "Notwithstanding Section 4.2 no environmental representation is made",
                 "Disclosed matters: a notice of violation remains unresolved",
                 "Aggregate liability shall not exceed ten percent",
                 "wholly unrelated trustee compensation schedule",
                 "Other document compliance with applicable laws"]})
    edges = pl.DataFrame({"doc": ["d1", "d1", "d1"], "src": ["4.2", "4.2", "4.9"], "dst": ["4.9", "sch4.2", "4.2"],
                          "kind": ["xref", "xref", "xref"]})
    cp, ep = tmp_path / "chunks.parquet", tmp_path / "edges.parquet"
    chunks.write_parquet(cp); edges.write_parquet(ep)
    return scan(str(cp)), scan(str(ep))


def test_bm25_ranks_lexical_match_and_is_deterministic(corpus):
    c, _ = corpus
    r = bm25_search(c, "compliance with applicable laws", k=3)
    assert r[0] in {node_id("d1", "4.2"), node_id("d2", "4.2")} and r == bm25_search(c, "compliance with applicable laws", k=3)
    assert bm25_search(c, "compliance with applicable laws", k=3, docs=["d2"]) == [node_id("d2", "4.2")]
    assert bm25_search(c, "zzzz") == [] and bm25_search(c, "") == []


def test_graph_spans_only_requested_docs_and_filters_kinds(corpus):
    c, e = corpus
    g = load_graph(c, e, docs=["d1"])
    assert g.num_edges() == 3 and node_id("d2", "4.2") not in g.by_id
    assert load_graph(c, e, kinds=["term"]).num_edges() == 0


def test_forward_and_reverse_closure(corpus):
    c, e = corpus
    g = load_graph(c, e, docs=["d1"])
    seed = node_id("d1", "4.2")
    assert set(dependency_order(g, seed, hops=1)) == {node_id("d1", "4.9"), node_id("d1", "sch4.2")}
    assert node_id("d1", "sch4.2") not in g.closure(seed)     # reverse: only 4.9 cites 4.2


def test_content_led_respects_budget_and_pulls_unlexical_members(corpus):
    c, e = corpus
    g = load_graph(c, e, docs=["d1"])
    seed = node_id("d1", "4.2")
    content = bm25_search(c, "compliance with applicable laws", k=1, docs=["d1"])
    out = content_led(g, seed, content, budget=10_000)
    assert out[0] == seed and node_id("d1", "sch4.2") in out and node_id("d1", "x") not in out
    tiny = content_led(g, seed, content, budget=g.tokens[seed])
    assert tiny == [seed]


# ───────────────────────── embeddings (deterministic bag-of-words stand-in; no model needed) ─────────────────────────
import hashlib  # noqa: E402

import numpy as np  # noqa: E402

from redevops_rag.parquet_index import (dense_search, duckdb_dense_search, hybrid_search, rrf_ids,  # noqa: E402
                                        simgraph_ppr_search, write_chunks_with_embeddings)


class BowEmbedder:
    """Hashing bag-of-words, 64-d. Stands in for a real encoder so the tests exercise the plumbing, not a model."""
    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64, dtype="float32")
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v.tolist())
        return out


@pytest.fixture()
def emb_corpus(tmp_path, corpus):
    c, _ = corpus
    df = c.collect()
    uri = str(tmp_path / "emb" / "chunks.parquet")
    write_chunks_with_embeddings(df, uri, BowEmbedder())
    return scan(uri), uri, BowEmbedder()


def test_duckdb_writes_embedding_column_normalised(emb_corpus):
    lf, uri, _ = emb_corpus
    df = lf.collect()
    assert df.schema["embedding"] == pl.List(pl.Float32) and df.height == 6
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-4 for v in df["embedding"].to_list())


def test_dense_matches_duckdb_in_place_ranking(emb_corpus):
    lf, uri, e = emb_corpus
    q = e.encode(["compliance with applicable laws"])[0]
    a = dense_search(lf, q, k=4)
    b = duckdb_dense_search(uri, q, k=4)
    assert a == b and a[0] in {node_id("d1", "4.2"), node_id("d2", "4.2")}
    assert dense_search(lf, q, k=4, docs=["d2"]) == [node_id("d2", "4.2")]


def test_hybrid_and_ppr_are_deterministic_and_bounded(emb_corpus):
    lf, uri, e = emb_corpus
    q = e.encode(["environmental representation compliance"])[0]
    h = hybrid_search(lf, "environmental representation compliance", q, k=3, docs=["d1"])
    assert h == hybrid_search(lf, "environmental representation compliance", q, k=3, docs=["d1"]) and len(h) == 3
    p = simgraph_ppr_search(lf, q, k=3, docs=["d1"], knn=2)
    assert p == simgraph_ppr_search(lf, q, k=3, docs=["d1"], knn=2) and len(set(p)) == 3
    assert rrf_ids([["a", "b"], ["b", "c"]])[0] == "b"
