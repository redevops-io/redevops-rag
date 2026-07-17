"""redevops-rag — hybrid RAG (vector + BM25 + RRF + recency/keyword priors + optional
cross-encoder rerank) as an installable library + CLI.

Extracted from redevops-io/rag-saas-platform's retrieval pipeline, decoupled from its
multi-tenant workspace/SaaS shell.
"""
from .embed import Embedder, NemotronEmbedder, make_embedder
from .retrieve import diver_search, hybrid_search, rrf_fuse
from .temporal import ReasonIREmbedder, TemporalReasoningRetriever
from .store import Store

__all__ = ["RAG", "Store", "PgStore", "Embedder", "NemotronEmbedder", "make_embedder", "hybrid_search", "diver_search", "rrf_fuse", "TemporalReasoningRetriever", "ReasonIREmbedder"]
__version__ = "0.2.0"


def __getattr__(name):
    # Lazy RAG so `import redevops_rag` stays light (no torch/sentence-transformers until used).
    if name == "RAG":
        from .rag import RAG
        return RAG
    if name == "PgStore":
        # Lazy: psycopg is an optional [pg] extra.
        from .pg_store import PgStore
        return PgStore
    raise AttributeError(name)
