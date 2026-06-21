"""Minimal end-to-end example. Run:  python examples/quickstart.py <folder>"""
import sys

from redevops_rag import RAG

folder = sys.argv[1] if len(sys.argv) > 1 else "."
rag = RAG(db_path="example.duckdb")        # add use_reranker=True for the rerank stage
stats = rag.index(folder)
print(f"indexed {stats['files']} files → {stats['chunks']} chunks")

for q in ["how does retrieval work", "installation"]:
    print(f"\n# {q}")
    for i, h in enumerate(rag.search(q, k=5), 1):
        print(f"  [{i}] {h['filename']}  rrf={h.get('rrf_score', 0):.4f}")
