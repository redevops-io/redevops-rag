"""Model-free tests for the embedder backend factory + the Nemotron HTTP client (no torch, no GPU,
no network — the urllib call is monkeypatched)."""
import json

import redevops_rag
from redevops_rag.embed import NemoRetrieverEmbedder, NemotronEmbedder, make_embedder


def test_make_embedder_selects_backend_without_loading_bge(monkeypatch):
    # bge is the default and would import sentence-transformers; assert selection via a stub so the
    # test stays model-free. nemotron/reasonir select without any heavy import.
    monkeypatch.setenv("REDEVOPS_RAG_EMBED_BACKEND", "nemotron")
    assert isinstance(make_embedder(), NemotronEmbedder)
    assert isinstance(make_embedder("nemotron"), NemotronEmbedder)
    monkeypatch.delenv("REDEVOPS_RAG_EMBED_BACKEND", raising=False)
    # explicit backend wins over env default; reasonir routes to the ReasonIR embedder
    from redevops_rag.temporal import ReasonIREmbedder
    assert isinstance(make_embedder("reasonir"), ReasonIREmbedder)


def test_nemotron_reads_env_config(monkeypatch):
    monkeypatch.setenv("REDEVOPS_RAG_NEMOTRON_URL", "http://gpu:9999/v1/embeddings")
    monkeypatch.setenv("REDEVOPS_RAG_NEMOTRON_MODEL", "nemotron-embed-8b")
    monkeypatch.setenv("REDEVOPS_RAG_NEMOTRON_DIM", "4096")
    e = NemotronEmbedder()
    assert e.url == "http://gpu:9999/v1/embeddings"
    assert e.model == "nemotron-embed-8b"
    assert e.dim == 4096


def test_nemotron_encode_posts_openai_shape_and_orders_by_index(monkeypatch):
    """encode() must POST the OpenAI /v1/embeddings body and re-sort the response by `index` (servers
    may return out of order), so row i maps back to input i."""
    sent = {}

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def read(self):
            return json.dumps(self._p).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        sent["auth"] = req.headers.get("Authorization")
        # return embeddings deliberately out of order to prove we re-sort by index
        data = [{"index": 1, "embedding": [0.2] * 4}, {"index": 0, "embedding": [0.1] * 4}]
        return _Resp({"data": data})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    e = NemotronEmbedder(url="http://x/v1/embeddings", model="m", dim=4, api_key="secret")
    out = e.encode(["first", "second"])
    assert sent["body"] == {"model": "m", "input": ["first", "second"]}
    assert sent["auth"] == "Bearer secret"
    assert out[0] == [0.1] * 4 and out[1] == [0.2] * 4   # re-ordered to match inputs


def test_nemotron_encode_queries_prepends_the_instruction(monkeypatch):
    captured = {}

    def fake_post(inputs):
        captured["inputs"] = list(inputs)
        return [[0.0]] * len(inputs)

    e = NemotronEmbedder()
    monkeypatch.setattr(e, "_post", fake_post)
    e.encode_queries(["how did risk change?"])
    assert captured["inputs"][0].startswith("Instruct:")
    assert captured["inputs"][0].endswith("how did risk change?")
    # documents are embedded raw (no instruction)
    e.encode(["a passage"])
    assert captured["inputs"] == ["a passage"]


def test_make_embedder_selects_nemo_retriever_without_heavy_import(monkeypatch):
    # `nemo`/`nemo-retriever` route to the NeMo Retriever embedding NIM client — no torch/ST import.
    assert isinstance(make_embedder("nemo"), NemoRetrieverEmbedder)
    assert isinstance(make_embedder("nemo-retriever"), NemoRetrieverEmbedder)
    monkeypatch.setenv("REDEVOPS_RAG_EMBED_BACKEND", "nemo")
    assert isinstance(make_embedder(), NemoRetrieverEmbedder)
    # `nemotron` still routes to the Nemotron embedder (unchanged).
    assert isinstance(make_embedder("nemotron"), NemotronEmbedder)


def test_nemo_retriever_reads_env_config(monkeypatch):
    monkeypatch.setenv("REDEVOPS_RAG_NEMO_URL", "http://gpu:8014/v1/embeddings")
    monkeypatch.setenv("REDEVOPS_RAG_NEMO_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
    monkeypatch.setenv("REDEVOPS_RAG_NEMO_DIM", "2048")
    e = NemoRetrieverEmbedder()
    assert e.url == "http://gpu:8014/v1/embeddings"
    assert e.model == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert e.dim == 2048


def test_nemo_retriever_encode_posts_openai_shape_and_orders_by_index(monkeypatch):
    """encode() must POST the OpenAI /v1/embeddings body and re-sort the response by `index`."""
    sent = {}

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def read(self):
            return json.dumps(self._p).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        sent["auth"] = req.headers.get("Authorization")
        # deliberately out of order to prove we re-sort by index
        data = [{"index": 1, "embedding": [0.2] * 4}, {"index": 0, "embedding": [0.1] * 4}]
        return _Resp({"data": data})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    e = NemoRetrieverEmbedder(url="http://x/v1/embeddings", model="m", dim=4, api_key="secret")
    out = e.encode(["first", "second"])
    assert sent["body"]["model"] == "m" and sent["body"]["input"] == ["first", "second"]
    assert sent["body"]["input_type"] == "passage"          # documents embed as passages
    assert sent["auth"] == "Bearer secret"
    assert out[0] == [0.1] * 4 and out[1] == [0.2] * 4       # re-ordered to match inputs
    # queries flip input_type to the asymmetric query side
    e.encode_queries(["how did risk change?"])
    assert sent["body"]["input_type"] == "query"


def test_exports():
    for name in ("Embedder", "NemotronEmbedder", "NemoRetrieverEmbedder", "make_embedder"):
        assert hasattr(redevops_rag, name), name
        assert name in redevops_rag.__all__
