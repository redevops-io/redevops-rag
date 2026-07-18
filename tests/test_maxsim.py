"""MaxSim late-interaction store — numpy-only (no GPU / network). A tiny fake multi-vector encoder
stands in for ColPali/ColQwen."""
import math

import pytest

np = pytest.importorskip("numpy")

from redevops_rag.maxsim import MaxSimStore, maxsim_score


def test_maxsim_score_rewards_best_matching_vector():
    # query has two vectors; doc contains a near-duplicate of each → score ≈ 2 (cosine 1 each).
    q = [[1.0, 0.0], [0.0, 1.0]]
    d_match = [[1.0, 0.0], [0.0, 1.0], [0.3, 0.3]]
    d_orth = [[0.0, -1.0], [-1.0, 0.0]]           # anti-aligned → max cosines negative
    assert maxsim_score(q, d_match) == pytest.approx(2.0, abs=1e-5)
    assert maxsim_score(q, d_orth) < maxsim_score(q, d_match)


def test_maxsim_score_empty_sides_are_zero():
    assert maxsim_score([], [[1.0, 0.0]]) == 0.0
    assert maxsim_score([[1.0, 0.0]], []) == 0.0


def test_store_ranks_the_document_with_the_matching_patch_first():
    s = MaxSimStore()
    s.add([
        {"document_id": "pageA", "text": "A", "multivector": [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]},
        {"document_id": "pageB", "text": "B", "multivector": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]},
    ])
    assert s.count() == 2
    # query aligns with pageB's [0,0,1] patch
    hits = s.search([[0.0, 0.0, 1.0]], top_k=2)
    assert hits[0]["document_id"] == "pageB"
    assert hits[0]["source_type"] == "maxsim" and hits[0]["maxsim_score"] > hits[1]["maxsim_score"]


class FakeColVision:
    """Mimics ColVisionEmbedder.encode_multivector: image/text → a multi-vector."""
    backend = "colpali"

    def __init__(self):
        self.seen = []

    def encode_multivector(self, items, images=True):
        self.seen += [("img" if images else "txt", it) for it in items]
        # deterministic 2-patch multivector keyed by the last char, so a query can match a doc
        out = []
        for it in items:
            k = float(ord(str(it)[-1]) % 5)
            out.append([[k, 1.0], [k + 0.1, 0.9]])
        return out


def test_store_embeds_images_and_queries_via_encoder():
    e = FakeColVision()
    s = MaxSimStore(embedder=e)
    s.add([{"document_id": "d1", "image": "page1"}, {"document_id": "d2", "image": "page2"}])
    assert s.count() == 2
    assert ("img", "page1") in e.seen                     # documents embedded as images
    hits = s.search("page1", top_k=2)                     # query embedded as text
    assert ("txt", "page1") in e.seen
    assert hits[0]["document_id"] == "d1"                 # matches the doc built from the same key


def test_add_requires_embedder_or_precomputed():
    s = MaxSimStore()  # no embedder
    with pytest.raises(ValueError):
        s.add([{"document_id": "x", "image": "p"}])


def test_save_load_roundtrip(tmp_path):
    s = MaxSimStore()
    s.add([
        {"document_id": "d1", "text": "A", "multivector": [[1.0, 0.0], [0.0, 1.0]]},
        {"document_id": "d2", "text": "B", "multivector": [[0.0, 1.0]]},
    ])
    p = str(tmp_path / "mv")
    s.save(p)
    s2 = MaxSimStore.load(p)
    assert s2.count() == 2
    # identical ranking after reload
    a = s.search([[1.0, 0.0]], top_k=2)
    b = s2.search([[1.0, 0.0]], top_k=2)
    assert [h["document_id"] for h in a] == [h["document_id"] for h in b]
    assert a[0]["maxsim_score"] == pytest.approx(b[0]["maxsim_score"], abs=1e-6)


def test_exports_lazy():
    import redevops_rag
    assert "MaxSimStore" in redevops_rag.__all__
    assert redevops_rag.MaxSimStore is MaxSimStore    # __getattr__ lazy path resolves
