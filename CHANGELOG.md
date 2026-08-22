# Changelog

## 0.2.1 — evidence-native lineage (v0.2.x final stabilization)

Correctness/completeness fix to functionality v0.2.x already implied: retrieval now preserves
**canonical evidence identity** end-to-end, and historical evidence stays addressable for
point-in-time replay. Closes the audited seam where RAG severed evidence identity before a consumer
(Discovery/Mission) could pin it. Additive and backward-compatible — the base file-ingest path is
unchanged.

**What is now true**

- **Canonical `EvidenceRef` bridge** (`redevops_rag.evidence`): a source revision is represented
  explicitly (`EvidenceRevision{ref, version, content, observed_at, source}`) and minted into a
  strict-`rcv1` `runtime_contracts.EvidenceRef` for the source and for each derived chunk, with a
  `DERIVED_FROM` lineage relation. `runtime-contracts>=0.3.0` is now a dependency.
- **Retrieval hits carry source identity** — every hit exposes `source_ref`, `source_version`,
  `source_content_hash` (strict rcv1) and `observed_at`; `evidence_ref_from_hit(hit)` reconstructs the
  pinnable `EvidenceRef`.
- **Version-aware retention** — `ingest_revision(...)` marks a prior revision *superseded* rather than
  deleting it. Retrieval gained opt-in `current_only` / `source_version` / `as_of` filters: the current
  projection returns the live revision, while a superseded revision stays retrievable by version or
  point-in-time (`Store.supersede_source`).
- **Hard guarantee** — pinning a version never returns a different revision's chunk (wrong-version
  substitution = 0).

**Backward compatibility** — new store columns default `NULL`; the base `ingest`/`search` path and all
existing corpora behave exactly as before. New tests: `tests/test_evidence_native_lineage.py`.
