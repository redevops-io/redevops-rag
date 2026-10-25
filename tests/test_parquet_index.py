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
