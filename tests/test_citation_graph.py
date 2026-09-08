"""Self-contained tests for the structural change-closure primitive (no network, no heavy deps).

Mirrors the mechanism proven in redevops-benchmarks/workloads/{statutory,xdoc}_change_closure: the
citation graph recovers the reverse-dependency closure that similarity retrieval cannot, and the precision
controls (confidence + trigger grounding) are exactly what make LLM-extracted implicit edges help.
"""
import json

from redevops_rag import (ChangeClosureGraph, Node, PrecisionEdgeExtractor, citation_closure,
                          citation_graph_retrieve, graph_union_retrieve)


def _graph():
    # s362 (automatic stay) is the seed. s101 and s541 CITE s362; s105 cites s541 (a 2-hop transitive
    # dependant). The transitive node deliberately shares NO words with the seed's text — the case where
    # lexical/similarity retrieval fails and only edge-traversal recovers the affected provision.
    nodes = [
        Node("s362", "automatic stay operation of the petition"),
        Node("s101", "definitions incorporate the stay under section 362 for all purposes here"),
        Node("s541", "property of the estate is subject to section 362 limitations"),
        Node("s105", "the court may issue orders to enforce property of the estate rules"),
        Node("s999", "wholly unrelated provision about trustee compensation schedules"),
    ]
    edges = {"s101": {"s362"}, "s541": {"s362"}, "s105": {"s541"}}
    return ChangeClosureGraph(nodes, edges)


def test_closure_is_reverse_reachable_set():
    g = _graph()
    assert g.closure("s362", hops=1) == {"s101", "s541"}          # direct citers
    assert g.closure("s362", hops=2) == {"s101", "s541", "s105"}  # + the transitive citer
    assert "s362" not in g.closure("s362")                        # never returns the seed
    # pure function agrees with the method
    assert citation_closure(g.cited_by, "s362", 2) == g.closure("s362", 2)


def test_citation_graph_retrieve_recovers_closure_within_budget():
    g = _graph()
    got = citation_graph_retrieve(g, "s362", budget=10_000, hops=2)
    assert set(got) == {"s101", "s541", "s105"}
    assert "s362" not in got and "s999" not in got               # seed and unrelated node excluded
    # direct citers (hop 1) come before the transitive citer (hop 2)
    assert got.index("s105") == max(got.index("s101"), got.index("s541")) + 1


def test_budget_knapsack_is_respected():
    g = _graph()
    # budget below one node's cost still returns the single highest-priority closure member, never the seed
    small = citation_graph_retrieve(g, "s362", budget=1, hops=2)
    assert len(small) == 1 and small[0] in {"s101", "s541"}


def test_graph_union_fills_tail_with_content_order():
    g = _graph()
    # graph closure first, then a content ranking (as ids) fills the remaining budget; unrelated s999 can
    # only enter via content, and only after every closure member.
    union = graph_union_retrieve(g, "s362", budget=10_000, hops=2, content_order=["s999", "s101"])
    assert set(union) == {"s101", "s541", "s105", "s999"}
    assert union.index("s999") > max(union.index(x) for x in ("s101", "s541", "s105"))
    # with no content order it equals the plain citation arm
    assert graph_union_retrieve(g, "s362", 10_000, 2) == citation_graph_retrieve(g, "s362", 10_000, 2)


def test_precision_extractor_applies_confidence_and_grounding():
    g = _graph()
    # s999 has an implicit dependency on s105 ("the foregoing orders") with no number to parse. The stub
    # returns four candidate edges; only the first must survive all three precision controls.
    def stub(_system, _user):
        return json.dumps([
            {"target": "s105", "trigger": "trustee compensation", "confidence": 0.9},  # KEEP: grounded + confident
            {"target": "s101", "trigger": "trustee compensation", "confidence": 0.4},  # DROP: below threshold
            {"target": "s541", "trigger": "phrase that is not in the source", "confidence": 0.95},  # DROP: ungrounded
            {"target": "s000", "trigger": "trustee compensation", "confidence": 0.99},  # DROP: unknown target
        ]) if "UNIT s999" in _user else "[]"

    edges = PrecisionEdgeExtractor(stub, min_confidence=0.7, max_workers=2).extract(g.nodes)
    assert edges == {"s999": {"s105"}}


def test_extractor_fails_soft_without_a_chat_fn():
    g = _graph()
    assert PrecisionEdgeExtractor(None).extract(g.nodes) == {}   # deterministic graph stands


def test_with_edges_enlarges_the_closure_immutably():
    g = _graph()
    enriched = g.with_edges({"s999": {"s105"}})
    assert g.num_edges() == 3 and enriched.num_edges() == 4       # base graph unchanged
    # s999 now depends on s105, so editing s105 reaches s999 that the base graph could not
    assert "s999" not in g.closure("s105", 2)
    assert "s999" in enriched.closure("s105", 2)
