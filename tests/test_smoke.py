"""Model-free smoke tests: the fusion math + chunker (no embeddings/network needed)."""
from redevops_rag.ingest import chunk_text
from redevops_rag.retrieve import rrf_fuse


def test_rrf_rewards_agreement_and_rank():
    vec = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    bm25 = [{"chunk_id": "b"}, {"chunk_id": "a"}, {"chunk_id": "d"}]
    fused = rrf_fuse([vec, bm25], k=60)
    keys = [r["chunk_id"] for r in fused]
    # 'a' and 'b' appear in both lists near the top → outrank singletons 'c'/'d'.
    assert keys[0] in {"a", "b"} and keys[1] in {"a", "b"}
    assert keys.index("a") < keys.index("c")
    assert keys.index("b") < keys.index("d")
    # RRF score for 'a' = 1/(60+0) + 1/(60+1).
    a = next(r for r in fused if r["chunk_id"] == "a")
    assert abs(a["rrf_score"] - (1 / 60 + 1 / 61)) < 1e-9


def test_rrf_falls_back_to_filename_key():
    fused = rrf_fuse([[{"filename": "x.md", "chunk_index": 3}]])
    assert fused[0]["rrf_score"] == 1 / 60


def test_chunker_packs_and_splits():
    assert chunk_text("") == []
    small = chunk_text("para one\n\npara two", size=1000)
    assert len(small) == 1 and "para one" in small[0]
    big = chunk_text("x" * 2500, size=1000, overlap=150)
    assert len(big) >= 3
    assert all(len(c) <= 1000 for c in big)
