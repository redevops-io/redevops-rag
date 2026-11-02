"""`redevops-rag` CLI: index a folder, hybrid-search it, or ask it a question."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="redevops-rag", description="Hybrid RAG over a folder/vault.")
    p.add_argument("--db", default="./redevops_rag.duckdb", help="DuckDB path (default ./redevops_rag.duckdb)")
    p.add_argument("--embed-model", default=None, help="sentence-transformers model (default BAAI/bge-small-en-v1.5)")
    p.add_argument("--rerank", action="store_true", help="cross-encoder rerank (needs the [rerank] extra)")
    p.add_argument("--tenant-aware", action="store_true",
                   help="open the database in tenant-aware mode (every command then needs --tenant-scope; fails closed without it)")
    p.add_argument("--tenant-scope", default=None, help="tenant identity for index/search/ask (validated; applied as a filter)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ip = sub.add_parser("index", help="index a file or folder")
    ip.add_argument("path")
    ip.add_argument("--chunk-size", type=int, default=1000)
    ip.add_argument("--overlap", type=int, default=150)

    sp = sub.add_parser("search", help="hybrid search; print the top chunks")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=8)

    ap = sub.add_parser("ask", help="search + synthesize an answer (needs an OpenAI-compatible LLM env)")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)

    args = p.parse_args(argv)

    from .rag import RAG  # lazy: only load models when actually running a command
    from .tenancy import TenantScopeError
    rag = RAG(db_path=args.db, embed_model=args.embed_model, use_reranker=args.rerank, tenant_aware=args.tenant_aware)
    scope_kw = {"tenant_scope": args.tenant_scope} if args.tenant_scope is not None else {}
    try:
        tagged = getattr(rag.store, "tenant_tagged_rows", lambda: 0)()
        if tagged and args.tenant_scope is None:        # tenant data must not be read or extended without an identity
            print(f"refusing: {args.db} holds {tagged} tenant-scoped chunks; pass --tenant-scope (and --tenant-aware)", file=__import__("sys").stderr)
            return 2
    except Exception:
        pass

    if args.cmd == "index":
        try:
            r = rag.index(args.path, size=args.chunk_size, overlap=args.overlap, **scope_kw)
        except TenantScopeError as e:
            print(f"refused: {e}", file=__import__("sys").stderr)
            return 2
        print(f"indexed {r['files']} files → {r['chunks']} chunks into {args.db}")
    elif args.cmd == "search":
        res = rag.search_result(args.query, k=args.k, **scope_kw)
        hits = res.hits
        if res.status.value == "denied":
            print(f"refused: {'; '.join(res.notes)}", file=__import__("sys").stderr)
            return 2
        if res.status.value in ("degraded", "unavailable"):
            failed = "; ".join(f"{l.name}: {l.error}" for l in res.failed_legs())
            print(f"WARNING: retrieval {res.status.value} — an empty or short list does NOT mean no match ({failed})", file=__import__("sys").stderr)
            if res.status.value == "unavailable":
                return 3
        if not hits and res.complete_coverage:
            print("no matches (is the folder indexed?)")
        for i, h in enumerate(hits, 1):
            tags = f"rrf={h.get('rrf_score', 0):.4f}"
            if "rerank_score" in h:
                tags += f" rerank={h['rerank_score']:.3f}"
            print(f"[{i}] {h['filename']}  ({tags})")
            print("    " + h["text"][:200].replace("\n", " ") + "…\n")
    elif args.cmd == "ask":
        try:
            out = rag.ask(args.query, k=args.k, **scope_kw)
        except TenantScopeError as e:
            print(f"refused: {e}", file=__import__("sys").stderr)
            return 2
        if out.get("answer"):
            print(out["answer"])
            print("\n— sources:", ", ".join(sorted({h["filename"] for h in out["sources"]})))
        else:
            if out.get("error"):
                print("LLM call failed:", out["error"], "\n")
            print("(no LLM configured — set REDEVOPS_RAG_LLM_BASE_URL / _API_KEY / _MODEL to synthesize answers)\n")
            print("Top retrieved context:\n")
            print(out["context"][:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
