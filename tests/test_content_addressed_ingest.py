"""Content-addressed chunk identity in the live ingest→store path (v0.2.x evidence-native reconciliation).

The live store used to mint position-based ids (`{path}::{index}`) that silently went stale when a source
was edited. Ingest now mints the content-addressed `chunk_identity` (`{document_ref}::{content_hash[:16]}`)
from nemo_retriever, persists content_hash, and prunes orphaned chunks on re-ingest.
"""
import hashlib
from pathlib import Path

from redevops_rag.store import Store
from redevops_rag.ingest import ingest
from redevops_rag.nemo_retriever import chunk_identity, content_hash


class FakeEmbedder:
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


# Paragraphs are long single-token blocks (~720 chars) so the packer keeps one chunk per paragraph,
# and the stored chunk text equals the paragraph verbatim (so chunk_identity is predictable).
PARA_A = ("alpha " * 120).strip()
PARA_B = ("delta " * 120).strip()
PARA_C = ("eta " * 180).strip()


def _write(p: Path, paras) -> None:
    p.write_text("\n\n".join(paras), encoding="utf-8")


def test_ids_are_content_addressed_and_content_hash_persisted(tmp_path):
    emb = FakeEmbedder()
    s = Store(emb, ":memory:")
    f = tmp_path / "doc.md"
    _write(f, [PARA_A, PARA_B])
    ingest(s, emb, f)
    s.reindex_fts()

    # the id is `{path}::{content_hash[:16]}` — position-independent, derived from content
    hits = s.semantic_search(PARA_A, top_k=5, threshold=0.0)
    assert hits, "expected a hit"
    h = next(h for h in hits if h["chunk_id"] == chunk_identity(str(f), PARA_A))
    assert h["content_hash"] == content_hash(PARA_A)


def test_reingest_identical_is_idempotent(tmp_path):
    emb = FakeEmbedder()
    s = Store(emb, ":memory:")
    f = tmp_path / "doc.md"
    _write(f, [PARA_A, PARA_B])
    ingest(s, emb, f)
    assert s.count() == 2
    ingest(s, emb, f)                      # same content → same ids → no duplication
    assert s.count() == 2


def test_reingest_shrunk_source_prunes_orphaned_tail_chunks(tmp_path):
    emb = FakeEmbedder()
    s = Store(emb, ":memory:")
    f = tmp_path / "doc.md"
    _write(f, [PARA_A, PARA_B, PARA_C])
    ingest(s, emb, f)
    assert s.count() == 3

    # the source shrinks to one paragraph; the two now-missing chunks must be swept, not left stale
    _write(f, [PARA_A])
    ingest(s, emb, f)
    assert s.count() == 1
    hits = s.semantic_search(PARA_C, top_k=5, threshold=0.0)
    assert all("eta" not in h["text"] for h in hits), "orphaned chunk survived re-ingest"


def test_prune_document_removes_only_non_kept(tmp_path):
    emb = FakeEmbedder()
    s = Store(emb, ":memory:")
    f = tmp_path / "doc.md"
    _write(f, [PARA_A, PARA_B])
    ingest(s, emb, f)
    assert s.count() == 2
    keep = [chunk_identity(str(f), PARA_A)]
    removed = s.prune_document(str(f), keep)
    assert removed == 1 and s.count() == 1
