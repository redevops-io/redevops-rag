"""Two correctness fixes: per-encoder vector sim_floor, and a Russian-capable FTS stemmer.

Both are real bugs the nutrition diagnosis surfaced (a bge-calibrated 0.4 threshold silently discards a
compressed-sim encoder's vector leg; the English Porter stemmer can't normalize Russian inflection).
Neither moves the (chunk-capped) nutrition answer-accuracy — they're correctness, not headline levers."""
import math

import pytest

from redevops_rag.store import Store


class SimEmbedder:
    """A 2-D embedder with a tunable sim_floor: query→[1,0], docs→a vector at cosine `sim` from it."""
    def __init__(self, sim=0.3, sim_floor=0.1, backend="nemotron"):
        self.dim = 2
        self._sim = sim
        self.sim_floor = sim_floor
        self.backend = backend

    def encode(self, texts):
        out = []
        for t in texts:
            if t == "__query__":
                out.append([1.0, 0.0])
            else:
                out.append([self._sim, math.sqrt(max(0.0, 1.0 - self._sim ** 2))])
        return out


def _seed(store, text="doc one"):
    store.add_chunks([{"id": "d1", "document_id": "d1", "text": text, "embedding": None}], reindex=False)
    # embed the doc through the store's embedder like ingest does
    store.con.execute("UPDATE chunks SET embedding = ? WHERE id='d1'", [store.embedder.encode([text])[0]])


# ── Fix 1: per-encoder vector threshold ──────────────────────────────────────────────────────
def test_threshold_none_uses_encoder_sim_floor():
    # compressed-sim encoder (sim 0.3): the bge-calibrated 0.4 would drop it; its own 0.1 floor keeps it.
    s = Store(SimEmbedder(sim=0.3, sim_floor=0.1), ":memory:")
    _seed(s)
    assert s.semantic_search("__query__", threshold=None), "sim_floor=0.1 should keep the 0.3 hit"
    assert s.semantic_search("__query__", threshold=0.4) == [], "the old global 0.4 discards it (the bug)"


def test_bge_like_floor_unchanged():
    # a bge-like encoder (sim_floor 0.4) resolves threshold=None to 0.4 → a 0.3 hit is filtered, as before.
    s = Store(SimEmbedder(sim=0.3, sim_floor=0.4, backend="bge"), ":memory:")
    _seed(s)
    assert s.semantic_search("__query__", threshold=None) == [], "bge behavior must be unchanged (0.4 floor)"
    # a stronger hit clears it
    s2 = Store(SimEmbedder(sim=0.9, sim_floor=0.4, backend="bge"), ":memory:")
    _seed(s2)
    assert s2.semantic_search("__query__", threshold=None), "0.9 sim clears the 0.4 floor"


# ── Fix 2: FTS stemmer / stopwords ───────────────────────────────────────────────────────────
def test_russian_stemmer_normalizes_inflection():
    s = Store(SimEmbedder(), ":memory:", fts_stemmer="russian", fts_stopwords="none")
    if not s._fts:
        pytest.skip("DuckDB fts extension unavailable")
    s.add_chunks([{"id": "r1", "document_id": "r1", "text": "фолиновая кислота помогает",
                   "embedding": [0.0, 0.0]}], reindex=True)
    # accusative query vs nominative doc — Porter (default) misses this; russian stemmer matches.
    hits = s.bm25_search("Фолиновую кислоту")
    assert any(h["chunk_id"] == "r1" for h in hits), "russian stemmer should match across RU inflection"


def test_default_stemmer_is_unchanged_english():
    s = Store(SimEmbedder(), ":memory:")   # defaults porter/english
    assert s.fts_stemmer == "porter" and s.fts_stopwords == "english"
    if not s._fts:
        pytest.skip("DuckDB fts extension unavailable")
    s.add_chunks([{"id": "e1", "document_id": "e1", "text": "the reconciliation jobs ran nightly",
                   "embedding": [0.0, 0.0]}], reindex=True)
    assert any(h["chunk_id"] == "e1" for h in s.bm25_search("reconciliation job")), "english stemming intact"


def test_invalid_stemmer_rejected():
    with pytest.raises(ValueError):
        Store(SimEmbedder(), ":memory:", fts_stemmer="russian'; DROP INDEX--")
