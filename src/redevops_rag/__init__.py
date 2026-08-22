"""redevops-rag — hybrid RAG (vector + BM25 + RRF + recency/keyword priors + optional
cross-encoder rerank) as an installable library + CLI.

Extracted from redevops-io/rag-saas-platform's retrieval pipeline, decoupled from its
multi-tenant workspace/SaaS shell.
"""
from .embed import (Embedder, NemotronEmbedder, NemoRetrieverEmbedder, make_embedder,
                    make_embedder_for, encoder_for)
from .retrieve import diver_search, hybrid_search, rrf_fuse
from .temporal import ReasonIREmbedder, TemporalReasoningRetriever
from .multimodal import ColVisionEmbedder
from .store import Store, open_store

__all__ = ["RAG", "Store", "open_store", "PgStore", "open_pg_store", "Embedder", "NemotronEmbedder",
           "NemoRetrieverEmbedder", "NemoRetrieverReranker", "make_embedder", "make_embedder_for",
           "encoder_for", "hybrid_search", "diver_search", "rrf_fuse", "TemporalReasoningRetriever",
           "ReasonIREmbedder", "ColVisionEmbedder", "MaxSimStore", "maxsim_score",
           "EvidenceRevision", "ingest_revision", "source_evidence_ref", "chunk_evidence_ref",
           "chunk_lineage", "evidence_ref_from_hit"]
__version__ = "0.2.1"


def __getattr__(name):
    # Lazy RAG so `import redevops_rag` stays light (no torch/sentence-transformers until used).
    if name == "RAG":
        from .rag import RAG
        return RAG
    if name == "PgStore":
        # Lazy: psycopg is an optional [pg] extra.
        from .pg_store import PgStore
        return PgStore
    if name == "open_pg_store":
        from .pg_store import open_pg_store
        return open_pg_store
    if name in ("MaxSimStore", "maxsim_score"):
        # Lazy: numpy import deferred so `import redevops_rag` stays light.
        from . import maxsim
        return getattr(maxsim, name)
    if name == "NemoRetrieverReranker":
        from .rerank import NemoRetrieverReranker
        return NemoRetrieverReranker
    if name in ("EvidenceRevision", "ingest_revision", "source_evidence_ref",
                "chunk_evidence_ref", "chunk_lineage", "evidence_ref_from_hit"):
        # Lazy: the evidence path imports runtime-contracts only when actually used.
        from . import evidence
        return getattr(evidence, name)
    raise AttributeError(name)
