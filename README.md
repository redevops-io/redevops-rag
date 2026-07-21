# redevops-rag

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg) [![NVIDIA Inception](https://img.shields.io/badge/NVIDIA-Inception%20Program%20Member-76B900.svg)](https://www.nvidia.com/en-us/startups/)

> **🚀 NVIDIA Inception Program Member** — ReDevOps is a member of the NVIDIA Inception Program, supporting startups advancing AI and accelerated computing. Membership provides access to NVIDIA technology, technical resources, and the startup ecosystem. It does not imply product endorsement by NVIDIA.

Hybrid RAG as a small, installable library + CLI — **DuckDB vector + BM25 fused via
Reciprocal Rank Fusion, recency & keyword priors, and an optional cross-encoder rerank.**

It's the retrieval pipeline from [redevops-io/rag-saas-platform](https://github.com/redevops-io/rag-saas-platform)
carved out of its multi-tenant SaaS shell (no Auth0/Stripe/Kubernetes/workspace coupling) so
you can drop the *same* retrieval over any folder — a docs tree, a repo, an Obsidian vault —
in three lines.

> **Versioning — backwards-compatible across v1 · v2 · v3 · v4 · v5.** The library API, CLI, and on-disk index/store format from every prior version keep working — upgrade in place, no migration.

## Pipeline

```
query
  ├─ dense   : embedding → DuckDB array_cosine_similarity (threshold = the encoder's sim_floor; bge 0.4, Nemotron/ReasonIR 0.1)
  └─ sparse  : DuckDB FTS BM25
        └─ Reciprocal Rank Fusion   score = Σ 1 / (k + rank),  k = 60
              └─ recency prior       0.5 ** (age_days / 90)
              └─ keyword prior       ×(1 + 0.05·term_hits), capped 1.5
                    └─ (optional) cross-encoder rerank  BAAI/bge-reranker-v2-m3  → top-k
```

## Install

Not on PyPI yet — install from git:

```bash
pip install "git+https://github.com/redevops-io/redevops-rag.git"                          # core
pip install "redevops-rag[rerank] @ git+https://github.com/redevops-io/redevops-rag.git"   # + cross-encoder rerank
pip install "redevops-rag[llm]    @ git+https://github.com/redevops-io/redevops-rag.git"   # + answer synthesis
```

## CLI

```bash
redevops-rag index ~/obsidian-vault              # chunk + embed + index into ./redevops_rag.duckdb
redevops-rag search "how do we rotate API keys"  # hybrid search, top chunks
redevops-rag --rerank search "..."               # add the cross-encoder rerank stage
redevops-rag ask "what's our incident process?"  # search + synthesized answer (needs an LLM, below)
```

Answer synthesis (`ask`) talks to **any OpenAI-compatible endpoint** — a local MLX/llama.cpp
server, OpenAI, or Anthropic behind a gateway:

```bash
export REDEVOPS_RAG_LLM_BASE_URL=http://localhost:8080/v1   # e.g. a Mac running mlx_lm.server
export REDEVOPS_RAG_LLM_MODEL=DeepSeek-V4-Flash
export REDEVOPS_RAG_LLM_API_KEY=EMPTY                       # or sk-... for a cloud endpoint
redevops-rag ask "summarize our on-call runbook"
```

## Library (the 3 lines)

```python
from redevops_rag import RAG

rag = RAG(db_path="vault.duckdb")          # add use_reranker=True for the cross-encoder stage
rag.index("~/obsidian-vault")              # chunk + embed + index (incremental: re-run anytime)
hits = rag.search("zero-downtime deploys", k=8)
# answer = rag.ask("zero-downtime deploys")["answer"]   # if an LLM env is set
```

## Why a folder, not a sync

Point it at one **central** copy of the knowledge base and query it — you don't copy 200k
files to every machine, you index once and retrieve. For a team, run it on one box (or behind
a thin service) so everyone hits a single index. Configuration / skills / `CLAUDE.md` are
small — those belong in **git**, not in a RAG.

## Configuration

| env | default | meaning |
|-----|---------|---------|
| `REDEVOPS_RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers embedding model |
| `REDEVOPS_RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | cross-encoder for `--rerank` |
| `REDEVOPS_RAG_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | — | OpenAI-compatible endpoint for `ask` |

> Status: **runnable, benchmarked library.** The retrieval logic is a faithful port of the
> production pipeline; the packaging/CLI are new. The `benchmarks/` suite evaluates recall/NDCG
> across PopQA · MuSiQue · LongMemEval · TEMPO · Nutrition (bge / ReasonIR / Nemotron-3-Embed
> encoders, hybrid / DIVER / graph methods). Validate on your own data.

AGPL-3.0-or-later · redevops.io
