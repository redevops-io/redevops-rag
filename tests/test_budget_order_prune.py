"""Budget, ordering, extraction-grounding, cache keys, ingest pruning and the optional namespace guard."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from redevops_rag import ChangeClosureGraph, Node, PrecisionEdgeExtractor
from redevops_rag.ingest import ingest, iter_files
from redevops_rag.namespace import NamespaceMismatch
from redevops_rag.retrieve import hybrid_search
from redevops_rag.store import Store

from test_content_addressed_ingest import FakeEmbedder


# ── budget ──
def test_retrieve_never_exceeds_budget_and_skips_oversize_nodes():
    g = ChangeClosureGraph(
        [Node("s", "x"), Node("big", "w " * 100), Node("small1", "a b"), Node("small2", "c d")],
        {"big": {"s"}, "small1": {"s"}, "small2": {"s"}})
    got = g.retrieve("s", 10)
    assert "big" not in got and set(got) == {"small1", "small2"}
    assert sum(g.tokens[i] for i in got) <= 10


def test_first_node_larger_than_budget_is_not_admitted():
    g = ChangeClosureGraph([Node("s", "x"), Node("big", "w " * 100)], {"big": {"s"}})
    assert g.retrieve("s", 10) == []


# ── determinism ──
_SCRIPT = """
from redevops_rag import ChangeClosureGraph, Node
nodes=[Node("seed","s")]+[Node(f"n{i}",f"t{i}") for i in range(12)]+[Node(f"m{i}","u") for i in range(5)]
edges={f"n{i}":{"seed"} for i in range(12)}
edges.update({f"m{i}":{f"n{i}"} for i in range(5)})
print(",".join(ChangeClosureGraph(nodes,edges).retrieve("seed",6,hops=2)))
"""


def test_closure_order_is_stable_across_hash_seeds():
    src = str(Path(__file__).resolve().parents[1] / "src")
    outs = set()
    for seed in ("0", "1", "2", "123", "999"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src}
        outs.add(subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    assert len(outs) == 1


def test_ties_break_by_in_degree_then_id():
    g = ChangeClosureGraph([Node(i, "t") for i in ("s", "b", "a", "c")],
                           {"b": {"s"}, "a": {"s"}, "c": {"s"}})
    order = g.retrieve("s", 100, hops=1)
    assert order == sorted(order)  # equal in-degree: id order
    g2 = ChangeClosureGraph([Node(i, "t") for i in ("s", "a", "b", "z")],
                            {"a": {"s"}, "b": {"s"}, "z": {"b"}})
    assert g2.retrieve("s", 100, hops=1) == ["b", "a"]  # b has higher in-degree


# ── union ──
def test_union_does_not_admit_unknown_ids():
    g = ChangeClosureGraph([Node("s", "x"), Node("a", "p q")], {"a": {"s"}})
    got = g.union_retrieve("s", 5, content_order=["ghost1", "ghost2", "a"])
    assert got == ["a"]


# ── extractor ──
def test_empty_trigger_is_not_accepted():
    ex = PrecisionEdgeExtractor(lambda s, u: '[{"target":"b","trigger":"","confidence":0.95}]')
    assert ex.extract([Node("a", "nothing related here"), Node("b", "y")]) == {}


def test_grounded_trigger_is_accepted():
    ex = PrecisionEdgeExtractor(lambda s, u: '[{"target":"b","trigger":"subject to b","confidence":0.95}]')
    assert ex.extract([Node("a", "this is subject to b above"), Node("b", "y")]) == {"a": {"b"}}


def test_cache_is_keyed_by_hash_and_ids_are_not_path_components(tmp_path):
    calls = []

    def chat(s, u):
        calls.append(1)
        return "[]"

    cache = tmp_path / "cache"
    ex = PrecisionEdgeExtractor(chat, cache_dir=str(cache))
    nodes = [Node("../../escape", "one"), Node("b", "two")]
    ex.extract(nodes)
    ex.extract(nodes)
    assert len(calls) == 2  # second run served from cache
    assert not (tmp_path / "escape.json").exists()
    assert all(p.parent == cache for p in cache.iterdir())
    ex.extract([Node("b", "changed text")])  # same id, new text: not a stale hit
    assert len(calls) == 3


# ── ingest ──
def _store():
    emb = FakeEmbedder()
    return emb, Store(emb, ":memory:")


def test_ingest_prunes_deleted_sources(tmp_path):
    emb, s = _store()
    (tmp_path / "a.md").write_text("alpha indemnity clause")
    (tmp_path / "b.md").write_text("beta unrelated")
    ingest(s, emb, tmp_path)
    s.reindex_fts()
    assert len(s.bm25_search("indemnity")) == 1
    (tmp_path / "a.md").unlink()
    r = ingest(s, emb, tmp_path)
    s.reindex_fts()
    assert r["removed_documents"] == 1
    assert s.bm25_search("indemnity") == []
    assert s.count() == 1


def test_ingest_prune_can_be_disabled_and_skips_missing_root(tmp_path):
    emb, s = _store()
    (tmp_path / "a.md").write_text("alpha text")
    ingest(s, emb, tmp_path)
    (tmp_path / "a.md").unlink()
    ingest(s, emb, tmp_path, prune=False)
    assert s.count() == 1
    missing = tmp_path / "gone"
    assert ingest(s, emb, missing)["removed_documents"] == 0
    assert s.count() == 1


def test_ingest_prune_does_not_touch_sibling_prefix(tmp_path):
    emb, s = _store()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs2").mkdir()
    (tmp_path / "docs" / "a.md").write_text("one")
    (tmp_path / "docs2" / "b.md").write_text("two")
    ingest(s, emb, tmp_path / "docs")
    ingest(s, emb, tmp_path / "docs2")
    ingest(s, emb, tmp_path / "docs")
    assert s.count() == 2


def test_ingest_emptied_source_is_removed_from_index(tmp_path):
    emb, s = _store()
    f = tmp_path / "a.md"
    f.write_text("alpha text")
    ingest(s, emb, tmp_path)
    f.write_text("")
    ingest(s, emb, tmp_path)
    assert s.count() == 0


def test_links_leaving_the_root_are_not_indexed(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("private text")
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.md").write_text("fine")
    (root / "link.md").symlink_to(outside)
    assert [p.name for p in iter_files(root)] == ["ok.md"]
    emb, s = _store()
    ingest(s, emb, root)
    assert s.count() == 1


# ── namespace guard ──
def test_graph_namespace_guard():
    g = ChangeClosureGraph([Node("s", "x"), Node("a", "y")], {"a": {"s"}}, namespace="t1")
    assert g.retrieve("s", 10, namespace="t1") == ["a"]
    with pytest.raises(NamespaceMismatch):
        g.retrieve("s", 10, namespace="t2")
    with pytest.raises(NamespaceMismatch):
        g.union_retrieve("s", 10)
    assert g.with_edges({}).namespace == "t1"
    unscoped = ChangeClosureGraph([Node("s", "x")], {})
    assert unscoped.retrieve("s", 10, namespace="anything") == []


def test_store_namespace_guard(tmp_path):
    emb = FakeEmbedder()
    s = Store(emb, ":memory:", namespace="t1")
    (tmp_path / "a.md").write_text("alpha text")
    ingest(s, emb, tmp_path)
    s.reindex_fts()
    assert hybrid_search(s, "alpha", namespace="t1", vector_threshold=-1)
    with pytest.raises(NamespaceMismatch):
        hybrid_search(s, "alpha", namespace="t2")
    with pytest.raises(NamespaceMismatch):
        hybrid_search(s, "alpha")
