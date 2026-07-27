"""Model-free tests for corpus→encoder routing, index-metadata persistence, the asymmetric
query-encoding path, and encoder-aware DIVER. No torch / GPU / network — a tiny fake embedder
stands in for the real encoders and records how each query was encoded."""
import redevops_rag
from redevops_rag.embed import encoder_for, make_embedder_for, make_embedder
from redevops_rag.multimodal import ColVisionEmbedder
from redevops_rag.store import Store, open_store


# ── Task 1: corpus → encoder routing ────────────────────────────────────────────────────────
def test_encoder_for_routes_by_language_and_domain():
    assert encoder_for("en") == "bge"                 # English → cheap bge
    assert encoder_for("en", "nutrition") == "bge"    # English wins regardless of domain
    assert encoder_for("ru") == "nemotron"            # non-English → multilingual Nemotron
    assert encoder_for("ru", "nutrition") == "nemotron"
    assert encoder_for(None) == "bge"                 # unknown lang, unknown domain → default
    assert encoder_for("de", "multimodal") == "colpali"   # doc-visual domain → colpali arm


def test_make_embedder_for_builds_routed_backend(monkeypatch):
    # ru → nemotron without importing sentence-transformers
    from redevops_rag.embed import NemotronEmbedder
    assert isinstance(make_embedder_for("ru"), NemotronEmbedder)


def test_make_embedder_selects_colpali():
    assert isinstance(make_embedder("colpali"), ColVisionEmbedder)
    assert isinstance(make_embedder("colqwen"), ColVisionEmbedder)
    assert make_embedder("colqwen").backend == "colqwen"


# ── a tiny deterministic embedder that records query-side calls ──────────────────────────────
class FakeEmbedder:
    backend = "fake"
    model_name = "fake-model"

    def __init__(self, dim=4, asymmetric=False):
        self.dim = dim
        self.asymmetric = asymmetric
        self.calls = []  # (side, text)

    def encode(self, texts):
        for t in texts:
            self.calls.append(("plain", t))
        return [[float(len(t) % 7)] * self.dim for t in texts]

    # only present when asymmetric (mirrors Nemotron/ReasonIR)
    def encode_queries(self, texts):
        for t in texts:
            self.calls.append(("instruct", t))
        return [[float(len(t) % 7)] * self.dim for t in texts]


class SymmetricFake:
    """A symmetric encoder (bge-like): no ``encode_queries`` — instruct must fall back to encode."""
    backend = "bge"
    model_name = "bge-small"

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def encode(self, texts):
        for t in texts:
            self.calls.append(("plain", t))
        return [[1.0] * self.dim for _ in texts]


# ── Task 1: index metadata persists the building encoder ─────────────────────────────────────
def test_store_stamps_and_open_store_reconstructs(tmp_path, monkeypatch):
    db = str(tmp_path / "corpus.duckdb")
    e = FakeEmbedder()
    e.backend = "nemotron"
    s = Store(e, db)
    assert s.get_meta("backend") == "nemotron"
    assert s.get_meta("dim") == 4
    s.close()

    # open_store reads the stamp and rebuilds the matching encoder (monkeypatch make_embedder so
    # we don't need a live Nemotron endpoint).
    built = {}

    def fake_make(backend=None, **kw):
        built["backend"] = backend
        return FakeEmbedder()

    # open_store resolves `from .embed import make_embedder` at call time → patch it there.
    import redevops_rag.embed as _embed
    monkeypatch.setattr(_embed, "make_embedder", fake_make)
    s2 = open_store(db)
    assert built["backend"] == "nemotron"   # reopened with the encoder it was built with
    s2.close()


def test_stamp_never_clobbered_on_reopen(tmp_path):
    db = str(tmp_path / "c.duckdb")
    a = FakeEmbedder(); a.backend = "bge"
    Store(a, db).close()
    # reopen with a DIFFERENT backend — the original stamp must survive (mismatch guard)
    b = FakeEmbedder(); b.backend = "nemotron"
    s = Store(b, db)
    assert s.get_meta("backend") == "bge"
    s.close()


# ── Task 2: asymmetric query encoding + encoder-aware DIVER ──────────────────────────────────
def _seed(store):
    store.add_chunks([
        {"id": "1", "document_id": "d1", "filename": "f", "chunk_index": 0,
         "text": "alpha beta", "embedding": [1.0, 1.0, 1.0, 1.0]},
    ], reindex=True)


def test_semantic_search_instruct_uses_encode_queries():
    e = FakeEmbedder(asymmetric=True)
    s = Store(e, ":memory:")
    _seed(s)
    e.calls.clear()
    s.semantic_search("the reasoning-heavy question", query_mode="instruct", threshold=-1.0)
    assert ("instruct", "the reasoning-heavy question") in e.calls

    e.calls.clear()
    s.semantic_search("a plain fragment", query_mode="plain", threshold=-1.0)
    assert e.calls == [("plain", "a plain fragment")]


def test_diver_sends_original_instructed_and_subqueries_plain():
    e = FakeEmbedder(asymmetric=True)
    s = Store(e, ":memory:")
    _seed(s)
    e.calls.clear()

    def reason_llm(system, user):
        # query expansion → two sub-queries; listwise rerank → passthrough
        if "Decompose" in system:
            return "sub one\nsub two"
        return "0"

    from redevops_rag.retrieve import diver_search
    diver_search(s, "ORIGINAL reasoning query", reason_llm, limit=1, pool=3, n_subqueries=2)

    instructed = [t for side, t in e.calls if side == "instruct"]
    plain = [t for side, t in e.calls if side == "plain"]
    assert instructed == ["ORIGINAL reasoning query"]        # only the original gets the instruction
    assert "sub one" in plain and "sub two" in plain          # fragments go in plain


def test_symmetric_encoder_ignores_instruct_mode():
    """For a symmetric encoder (bge) instruct/plain/auto all collapse to plain encode — so
    DIVER-over-bge, the recommended config, stays byte-identical to before this change."""
    e = SymmetricFake()
    s = Store(e, ":memory:")
    _seed(s)
    e.calls.clear()
    s.semantic_search("q", query_mode="instruct", threshold=-1.0)
    s.semantic_search("q", query_mode="auto", threshold=-1.0)
    assert e.calls == [("plain", "q"), ("plain", "q")]   # never took an instruct path


def test_exports():
    for name in ("open_store", "make_embedder_for", "encoder_for", "ColVisionEmbedder"):
        assert name in redevops_rag.__all__ and hasattr(redevops_rag, name), name
