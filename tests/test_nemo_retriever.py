"""Model-free tests for the NeMo Retriever reranking NIM + the extraction capability (no torch, no
GPU, no network — every urllib call is monkeypatched). Mirrors tests/test_embed_factory.py."""
import json

import redevops_rag
from redevops_rag import nemo_retriever as nr
from redevops_rag.nemo_retriever import (ArtifactHandle, UnsupportedArtifact, chunk_identity,
                                         make_artifact_handle)
from redevops_rag.rerank import NemoRetrieverReranker


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── reranker ──────────────────────────────────────────────────────────────────────────────────

def test_reranker_reads_env_config(monkeypatch):
    monkeypatch.setenv("REDEVOPS_RAG_NEMO_RERANK_URL", "http://gpu:8015/v1/ranking")
    monkeypatch.setenv("REDEVOPS_RAG_NEMO_RERANK_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")
    r = NemoRetrieverReranker()
    assert r.url == "http://gpu:8015/v1/ranking"
    assert r.model == "nvidia/llama-3.2-nv-rerankqa-1b-v2"


def test_reranker_posts_ranking_body_and_records_pre_and_post_order(monkeypatch):
    """The NIM returns rankings best-first; evidence must record pre-rank AND post-rank order."""
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        sent["auth"] = req.headers.get("Authorization")
        # candidate 2 is most relevant, then 0, then 1 — deliberately not the input order.
        return _Resp({"rankings": [{"index": 2, "logit": 9.0},
                                   {"index": 0, "logit": 5.0},
                                   {"index": 1, "logit": 1.0}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cands = [{"chunk_id": "a", "text": "alpha"},
             {"chunk_id": "b", "text": "bravo"},
             {"chunk_id": "c", "text": "charlie"}]
    r = NemoRetrieverReranker(url="http://x/v1/ranking", model="m", api_key="secret")
    env = r.rerank_with_evidence("q", cands)

    assert sent["body"] == {"model": "m", "query": {"text": "q"},
                            "passages": [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]}
    assert sent["auth"] == "Bearer secret"
    # provider-capability/v1 envelope field names
    assert env["provider"] == "nvidia-nemo-retriever"
    assert env["service_kind"] == "nemo-retriever"
    assert env["model_id"] == "m"
    for f in ("request_digest", "response_digest", "cost_usd", "latency_ms", "evidence"):
        assert f in env
    # the acceptance criterion: pre-rank vs post-rank order is recorded as evidence
    assert env["evidence"]["pre_rank_order"] == ["a", "b", "c"]
    assert env["evidence"]["post_rank_order"] == ["c", "a", "b"]
    assert [c["chunk_id"] for c in env["ranked"]] == ["c", "a", "b"]
    assert env["ranked"][0]["rerank_score"] == 9.0


def test_reranker_rerank_is_interface_compatible_and_stashes_evidence(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _Resp({"rankings": [{"index": 1, "logit": 3.0}, {"index": 0, "logit": 2.0}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cands = [{"chunk_id": "x", "text": "one"}, {"chunk_id": "y", "text": "two"}]
    r = NemoRetrieverReranker(url="http://x/v1/ranking", model="m")
    ranked = r.rerank("q", cands)               # same contract as Reranker.rerank -> list[dict]
    assert [c["chunk_id"] for c in ranked] == ["y", "x"]
    assert r.last_evidence["pre_rank_order"] == ["x", "y"]
    assert r.last_evidence["post_rank_order"] == ["y", "x"]


def test_reranker_empty_candidates_short_circuits():
    r = NemoRetrieverReranker()
    assert r.rerank("q", []) == []
    assert r.last_evidence["pre_rank_order"] == []


def test_reranker_exported_lazily():
    assert redevops_rag.NemoRetrieverReranker is NemoRetrieverReranker
    assert "NemoRetrieverReranker" in redevops_rag.__all__


# ── extraction / content-hashing path ───────────────────────────────────────────────────────────

def test_content_hashing_gives_stable_chunk_ids():
    """Identical (document_ref, content) → identical chunk id (position-independent)."""
    h1 = make_artifact_handle("doc.pdf", "the paragraph text")
    h2 = make_artifact_handle("doc.pdf", "the paragraph text")
    assert h1.chunk_id == h2.chunk_id == chunk_identity("doc.pdf", "the paragraph text")
    assert h1 == h2                                    # frozen value equality
    # different content or ref → different id
    assert make_artifact_handle("doc.pdf", "other").chunk_id != h1.chunk_id
    assert make_artifact_handle("other.pdf", "the paragraph text").chunk_id != h1.chunk_id


def _extract_elements(monkeypatch, elements):
    def fake_urlopen(req, timeout=None):
        fake_urlopen.body = json.loads(req.data.decode())
        return _Resp({"elements": elements})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    fake_urlopen.body = None
    handles = nr.extract("report.pdf")
    return handles, fake_urlopen.body


def test_extract_routes_text_through_hashing_and_is_stable(monkeypatch):
    elements = [{"type": "text", "content": "intro paragraph", "media_type": "text/plain"},
                {"type": "table", "content": "col1,col2", "media_type": "application/x-table"}]
    handles, body = _extract_elements(monkeypatch, elements)
    assert body == {"model": "nvidia/nemoretriever-parse", "document": {"ref": "report.pdf"}}
    assert all(isinstance(h, ArtifactHandle) for h in handles)
    # every handle's id is exactly what the canonical hashing path produces (no bypass)
    for h in handles:
        assert h.chunk_id == chunk_identity("report.pdf", h.content)
    # identical ingestion → identical chunk identifiers
    handles2, _ = _extract_elements(monkeypatch, elements)
    assert [h.chunk_id for h in handles] == [h.chunk_id for h in handles2]


def test_extract_types_unsupported_multimodal_not_dropped(monkeypatch):
    elements = [{"type": "text", "content": "caption text", "media_type": "text/plain"},
                {"type": "chart", "media_type": "image/png"},                 # no text → typed
                {"type": "text", "content": "   ", "media_type": "text/plain"}]  # empty → typed
    handles, _ = _extract_elements(monkeypatch, elements)
    # nothing dropped: 3 in, 3 out
    assert len(handles) == 3
    supported = [h for h in handles if isinstance(h, ArtifactHandle)]
    unsupported = [h for h in handles if isinstance(h, UnsupportedArtifact)]
    assert len(supported) == 1 and len(unsupported) == 2
    assert unsupported[0].media_type == "image/png"
    assert unsupported[0].reason and unsupported[1].reason      # typed with a reason


def test_embed_capability_references_chunk_identity(monkeypatch):
    handles = [make_artifact_handle("d.pdf", "one"), make_artifact_handle("d.pdf", "two")]

    class _StubEmbedder:
        model = "nvidia/llama-3.2-nv-embedqa-1b-v2"

        def encode(self, texts):
            return [[0.1, 0.2] for _ in texts]

    refs = nr.embed(handles, embedder=_StubEmbedder())
    assert [r["chunk_id"] for r in refs] == [h.chunk_id for h in handles]
    assert all(r["content_hash"] == h.content_hash for r, h in zip(refs, handles))
    assert all(r["model_id"] == "nvidia/llama-3.2-nv-embedqa-1b-v2" for r in refs)


def test_embed_capability_skips_unsupported():
    refs = nr.embed([UnsupportedArtifact("d.pdf", "image/png", "no text")],
                    embedder=object())        # must not be called → no encode needed
    assert refs == []
