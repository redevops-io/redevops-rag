"""Incremental insert (LightRAG-style live-corpus update) for DIVER — no index rebuild, store-backed."""
import hashlib

from redevops_rag.store import Store
from redevops_rag.temporal import TemporalReasoningRetriever


class FakeEmbedder:
    """Bag-of-words hash embedder — lexical overlap ≈ cosine, no torch/network."""
    backend = "fake"
    model_name = "fake"

    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in str(t).lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def test_diver_insert_is_incremental_and_immediately_visible():
    s = Store(FakeEmbedder(), ":memory:")
    div = TemporalReasoningRetriever()  # reason_llm None → search degrades to plain hybrid (no LLM)

    n0 = div.insert(s, [{"document_id": "d1", "text": "alpha beta gamma"}])
    assert n0 == 1 and s.count() == 1

    # incremental add — a growing corpus, no rebuild
    n1 = div.insert(s, [{"document_id": "d2", "text": "delta epsilon zeta"},
                        {"document_id": "d3", "text": "eta theta iota"}])
    assert n1 == 2 and s.count() == 3

    # the very next search sees the newly inserted docs
    hits = div.search(s, "delta epsilon zeta", limit=3)
    assert any(h["document_id"] == "d2" for h in hits), "insert not visible to the next search"


def test_diver_insert_empty_is_noop():
    s = Store(FakeEmbedder(), ":memory:")
    assert TemporalReasoningRetriever().insert(s, []) == 0 and s.count() == 0
