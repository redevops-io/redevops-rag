"""redevops-rag — hybrid RAG (vector + BM25 + RRF + recency/keyword priors + optional
cross-encoder rerank) as an installable library + CLI.

Extracted from redevops-io/rag-saas-platform's retrieval pipeline, decoupled from its
multi-tenant workspace/SaaS shell.
"""
from .embed import Embedder
from .retrieve import hybrid_search, rrf_fuse
from .store import Store

__all__ = ["RAG", "Store", "Embedder", "hybrid_search", "rrf_fuse"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy RAG so `import redevops_rag` stays light (no torch/sentence-transformers until used).
    if name == "RAG":
        from .rag import RAG
        return RAG
    raise AttributeError(name)
