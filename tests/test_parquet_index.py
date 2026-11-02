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


# ── duckdb_dense_search: document filter, scope-before-LIMIT, and hostile paths ───────────────────────────────

def test_duckdb_dense_search_with_docs_matches_the_in_process_ranking(emb_corpus):
    lf, uri, e = emb_corpus
    q = e.encode(["compliance with applicable laws"])[0]
    all_docs = sorted(set(lf.collect()["doc"].to_list()))
    assert len(all_docs) >= 2
    for subset in ([all_docs[0]], [all_docs[-1]], all_docs, list(reversed(all_docs))):
        for k in (1, 3, 50):
            assert duckdb_dense_search(uri, q, k=k, docs=subset) == dense_search(lf, q, k=k, docs=subset), (subset, k)


def test_duckdb_dense_search_filters_before_the_limit(emb_corpus):
    lf, uri, e = emb_corpus
    q = e.encode(["compliance with applicable laws"])[0]
    best = duckdb_dense_search(uri, q, k=1)[0]
    best_doc = best.split("#")[0]
    other = sorted(d for d in set(lf.collect()["doc"].to_list()) if d != best_doc)[0]
    got = duckdb_dense_search(uri, q, k=1, docs=[other])
    assert len(got) == 1 and got[0] != best and got[0].startswith(other + "#")   # not truncated to the global top-1 then filtered


def test_duckdb_dense_search_empty_docs_selects_nothing(emb_corpus):
    _, uri, e = emb_corpus
    q = e.encode(["compliance"])[0]
    assert duckdb_dense_search(uri, q, k=5, docs=[]) == []
    assert duckdb_dense_search(uri, q, k=5, docs=iter(())) == []


@pytest.mark.parametrize("name", ["it's.parquet", "a (b); c.parquet", "x'); DROP TABLE t; --.parquet", "naïve-文書.parquet",
                                  'quote"d.parquet', "percent%20and space.parquet"])
def test_parquet_paths_with_sql_metacharacters_round_trip(tmp_path, corpus, name):
    c, _ = corpus
    uri = str(tmp_path / "dir with ' quote" / name)
    write_chunks_with_embeddings(c.collect(), uri, BowEmbedder())
    q = BowEmbedder().encode(["compliance with applicable laws"])[0]
    assert duckdb_dense_search(uri, q, k=3) == dense_search(scan(uri), q, k=3)
    assert not any(p.name.startswith("DROP") for p in tmp_path.iterdir())


def test_a_path_with_a_nul_byte_is_refused(tmp_path, corpus):
    c, _ = corpus
    with pytest.raises(ValueError):
        write_chunks_with_embeddings(c.collect(), str(tmp_path / "bad\x00name.parquet"), BowEmbedder())


# ── tenant scoping of a Parquet index ───────────────────────────────────────────────────────────────────────────
from redevops_rag.parquet_index import bm25_search as _bm25, scope_chunks  # noqa: E402
from redevops_rag.tenancy import TenantScopeError  # noqa: E402


@pytest.fixture()
def tenant_index(tmp_path):
    rows = []
    for t, n in (("tenant-A", 12), ("tenant-B", 2)):
        for i in range(n):
            rows.append((t, f"{t}-d{i}", "1", "quarterly budget forecast alpha alpha" if t == "tenant-A" else "budget notes beta"))
    df = pl.DataFrame({"tenant_scope": [r[0] for r in rows], "doc": [r[1] for r in rows], "node": [r[2] for r in rows],
                       "text": [r[3] for r in rows]})
    uri = str(tmp_path / "t" / "chunks.parquet")
    write_chunks_with_embeddings(df, uri, BowEmbedder())
    return scan(uri), uri


def test_scope_chunks_confines_every_ranking_and_its_statistics(tenant_index):
    lf, uri = tenant_index
    scoped = scope_chunks(lf, "tenant-B")
    for k in (1, 50):
        for ids in (_bm25(scoped, "budget forecast", k=k),
                    dense_search(scoped, BowEmbedder().encode(["budget forecast"])[0], k=k),
                    hybrid_search(scoped, "budget forecast", BowEmbedder().encode(["budget forecast"])[0], k=k)):
            assert ids and all(i.startswith("tenant-B-") for i in ids)
    # statistics are those of the tenant: same score ordering as a standalone index of only tenant-B's rows
    assert _bm25(scoped, "budget", k=5) == _bm25(lf.filter(pl.col("tenant_scope") == "tenant-B"), "budget", k=5)


def test_duckdb_dense_search_filters_on_tenant_before_the_limit(tenant_index):
    lf, uri = tenant_index
    q = BowEmbedder().encode(["quarterly budget forecast alpha"])[0]
    assert all(i.startswith("tenant-A-") for i in duckdb_dense_search(uri, q, k=1))          # A outranks B globally
    got = duckdb_dense_search(uri, q, k=1, tenant_scope="tenant-B")
    assert len(got) == 1 and got[0].startswith("tenant-B-")
    assert got == dense_search(scope_chunks(lf, "tenant-B"), q, k=1)
    with pytest.raises(TenantScopeError):
        duckdb_dense_search(uri, q, k=1, tenant_scope="*")


def test_an_untagged_index_cannot_be_scoped_and_bad_scopes_are_refused(corpus):
    lf, _ = corpus
    with pytest.raises(TenantScopeError):
        scope_chunks(lf, "tenant-A")                                   # no tenant_scope column ⇒ refuse, never return it whole
    for bad in ("", "*", "a::b", None):
        with pytest.raises(TenantScopeError):
            scope_chunks(lf, bad)
