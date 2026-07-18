"""Embedder A/B — bge-small-en (cheap CPU default) vs Nemotron-3-Embed-8B (RTEB #1, GPU) — across
retrieval methods, on one dataset. Answers "does switching the encoder actually help, and on which
method?" before we consider making Nemotron a default.

For EACH embedder it builds a fresh store, embeds the corpus, and scores the same three retrieval
methods (SIM-RAG/BM25, hybrid, DIVER) over the same queries, then prints a side-by-side table with
per-method deltas. BM25 is embedder-independent (a sanity anchor: it should be ~identical in both
columns). Metrics: NDCG@10, Recall@10, MRR vs gold ids.

    # bge only (no GPU endpoint needed):
    EMBED_BACKENDS=bge  TEMPO_DOMAIN=workplace  python benchmarks/eval_embedders.py

    # A/B once Nemotron is served (NIM/vLLM on :8013, OpenAI-compatible /v1/embeddings):
    EMBED_BACKENDS=bge,nemotron  REDEVOPS_RAG_NEMOTRON_URL=http://192.168.40.105:8013/v1/embeddings \
      TEMPO_DOMAIN=workplace  TEMPO_NQ=36  python benchmarks/eval_embedders.py

Datasets: defaults to a TEMPO domain (tempo26/Tempo); this is deliberately the same loader as
eval_tempo_diver.py so the numbers line up with the DIVER ablation. Point it at other corpora by
adapting the loader block.
"""
import ast
import math
import os
import re
import statistics as st

import duckdb
from huggingface_hub import hf_hub_download
from openai import OpenAI

from redevops_rag.embed import make_embedder
from redevops_rag.retrieve import hybrid_search
from redevops_rag.store import Store

DOMAIN = os.environ.get("TEMPO_DOMAIN", "workplace")
N_Q = int(os.environ.get("TEMPO_NQ", "36"))
BACKENDS = [b.strip() for b in os.environ.get("EMBED_BACKENDS", "bge,nemotron").split(",") if b.strip()]
ANS_MODEL = os.environ.get("ANS_MODEL", "Qwen3.6-35B-A3B")
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
llm = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://192.168.40.105:30807/v1"), api_key="EMPTY")


def chat(system, user, max_tokens=120, temp=0.0):
    r = llm.chat.completions.create(model=ANS_MODEL, temperature=temp, max_tokens=max_tokens,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()


def expand_query(query):
    out = chat("Decompose the search query into 2-3 focused sub-queries covering the entities and the "
               "relevant time periods needed to answer it. One per line, no numbering.", query[:1500], temp=0.3)
    return [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:3]


def listwise_rerank(query, cands, top):
    if len(cands) <= top:
        return cands
    snip = "\n".join(f"[{i}] {c['text'][:280]}" for i, c in enumerate(cands))
    out = chat(f"Rank the passages by usefulness for the query. Return the {top} best passage numbers, "
               f"comma-separated, best first. Numbers only.", f"Query: {query[:800]}\n\nPassages:\n{snip}", max_tokens=80)
    order = [int(x) for x in re.findall(r"\d+", out)]
    seen, ranked = set(), []
    for i in order:
        if 0 <= i < len(cands) and i not in seen:
            seen.add(i); ranked.append(cands[i])
    for i, c in enumerate(cands):
        if i not in seen:
            ranked.append(c)
    return ranked[:top]


def diver(store, query, pool=25, top=10):
    cand = {}
    # Encoder-aware DIVER (I2): the reasoning-heavy ORIGINAL query is embedded with the encoder's
    # instruction side (query_mode='instruct'), the expanded sub-queries plain — the instruction is
    # tuned for the full query and hurts the fragments. Symmetric bge collapses both to plain encode.
    for i, q in enumerate([query] + expand_query(query)):
        mode = "instruct" if i == 0 else "plain"
        for h in hybrid_search(store, q, limit=pool, recency_half_life_days=0, query_mode=mode):
            cand.setdefault(h["document_id"], h)
    return listwise_rerank(query, list(cand.values()), top)


def ndcg_at_k(ranked_ids, gold, k=10):
    dcg = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(ranked_ids[:k]) if d in gold)
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_ids, gold, k=10):
    return len(set(ranked_ids[:k]) & gold) / len(gold) if gold else 0.0


def mrr(ranked_ids, gold):
    for i, d in enumerate(ranked_ids):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


# ---- load domain once (shared across embedders) ----
docs_p = hf_hub_download("tempo26/Tempo", f"documents/{DOMAIN}.parquet", repo_type="dataset")
ex_p = hf_hub_download("tempo26/Tempo", f"examples/{DOMAIN}.parquet", repo_type="dataset")
con = duckdb.connect()
docs = con.execute(f"SELECT id, content FROM '{docs_p}'").fetchall()
exs = con.execute(f"SELECT query, gold_ids FROM '{ex_p}' LIMIT {N_Q}").fetchall()
texts = [c for _, c in docs]
print(f"domain={DOMAIN}  docs={len(docs)}  queries={len(exs)}  backends={BACKENDS}", flush=True)

METHODS = ("sim_bm25", "hybrid", "diver")


def score_backend(backend):
    print(f"\n[{backend}] embedding {len(texts)} docs...", flush=True)
    emb = make_embedder(backend)
    print(f"[{backend}] dim={emb.dim}", flush=True)
    store = Store(emb, f"/tmp/embcmp_{backend}_{DOMAIN}.duckdb")
    vecs = emb.encode(texts)
    store.add_chunks([{"document_id": docs[i][0], "text": texts[i], "embedding": vecs[i], "metadata": {}}
                      for i in range(len(docs))], reindex=True)
    configs = {
        "sim_bm25": lambda q: store.bm25_search(q, limit=10),
        "hybrid":   lambda q: hybrid_search(store, q, limit=10, recency_half_life_days=0),
        "diver":    lambda q: diver(store, q, pool=25, top=10),
    }
    res = {m: {"ndcg": [], "recall": [], "mrr": []} for m in METHODS}
    for i, (query, gold_raw) in enumerate(exs):
        gold = set(ast.literal_eval(gold_raw) if isinstance(gold_raw, str) else list(gold_raw))
        for m in METHODS:
            ranked = [h["document_id"] for h in configs[m](query)]
            res[m]["ndcg"].append(ndcg_at_k(ranked, gold))
            res[m]["recall"].append(recall_at_k(ranked, gold))
            res[m]["mrr"].append(mrr(ranked, gold))
        if (i + 1) % 10 == 0:
            print(f"  [{backend}] {i+1}/{len(exs)}", flush=True)
    return {m: {k: st.mean(v) for k, v in res[m].items()} for m in METHODS}


scores = {b: score_backend(b) for b in BACKENDS}

print(f"\n=== embedder A/B — TEMPO/{DOMAIN} ({len(exs)} queries), NDCG@10 ===")
header = f"{'method':10}" + "".join(f"{b:>12}" for b in BACKENDS)
if len(BACKENDS) == 2:
    header += f"{'Δ':>10}"
print(header)
for m in METHODS:
    row = f"{m:10}" + "".join(f"{scores[b][m]['ndcg']:>12.3f}" for b in BACKENDS)
    if len(BACKENDS) == 2:
        row += f"{scores[BACKENDS[1]][m]['ndcg'] - scores[BACKENDS[0]][m]['ndcg']:>+10.3f}"
    print(row)
print("\n(Recall@10 / MRR per method also collected — extend the print block if you need them.)")
print("Note: queries now use the encoder's ASYMMETRIC side by default (hybrid_search query_mode='auto' → "
      "encode_queries for Nemotron/ReasonIR); DIVER instructs the original + keeps sub-queries plain. "
      "This measures I2 — expect Nemotron's hybrid/vector arms to lift vs the earlier symmetric baseline.")
