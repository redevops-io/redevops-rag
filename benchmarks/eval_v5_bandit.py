"""Context Runtime v5 — bandit learning curve for retrieval selection (deterministic).

Streams TEMPO queries through ContextRuntimeRetrieverTuner with an NDCG@10 *retrieval*
reward (gold_ids; no answer model, no judge). Shows the bandit LEARNING to select among
the redevops-rag arms — incl. the DIVER+ReasonIR temporal plugin — vs fixed SIM-RAG and a
fixed-DIVER oracle. Persists the ReasonIR store so re-runs are fast.

  DOMAIN=workplace EPOCHS=3 .venv/bin/python benchmarks/eval_v5_bandit.py
"""
import sys, os, math, ast, json, statistics as st, hashlib
import duckdb
from huggingface_hub import hf_hub_download
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
sys.path.insert(0, "/mnt/backup/projects/contextos")
from redevops_rag.store import Store
from redevops_rag.retrieve import diver_search
from redevops_rag.temporal import ReasonIREmbedder
from context_runtime.integrations.redevops_rag import (
    ContextRuntimeRetrieverTuner, reward_from_quality, RetrievalConfig)
from openai import OpenAI

DOMAIN = os.environ.get("DOMAIN", "workplace")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
STORE = f"/mnt/backup/projects/redevops-rag/benchmarks/results/v5_store_{DOMAIN}.duckdb"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
reason_cli = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")

def reason_llm(system, user):
    r = reason_cli.chat.completions.create(model="Qwen3.6-35B-A3B", temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def ndcg(hits, gold, k=10):
    ids = [h.get("document_id") or h.get("id") for h in hits]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ids[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0

emb = ReasonIREmbedder(url="http://192.168.40.105:8012/v1/embeddings")
docs_p = hf_hub_download("tempo26/Tempo", f"documents/{DOMAIN}.parquet", repo_type="dataset")
ex_p = hf_hub_download("tempo26/Tempo", f"examples/{DOMAIN}.parquet", repo_type="dataset")
con = duckdb.connect()
queries = [(q, set(ast.literal_eval(g) if isinstance(g, str) else list(g)))
           for q, g in con.execute(f"SELECT query, gold_ids FROM '{ex_p}'").fetchall()]

store = Store(emb, STORE)
if store.count() == 0:
    docs = con.execute(f"SELECT id, content FROM '{docs_p}'").fetchall()
    print(f"embedding {len(docs)} docs via ReasonIR (one-time, persisted)...", flush=True)
    for c in (chunks := [{"document_id": d[0], "text": d[1], "metadata": {}} for d in docs]):
        pass
    for c, e in zip(chunks, emb.encode([c["text"] for c in chunks])): c["embedding"] = e
    store.add_chunks(chunks, reindex=True)
print(f"store={store.count()} docs | queries={len(queries)} | epochs={EPOCHS}", flush=True)

class _RAG:
    def __init__(self, store): self.store, self.reranker = store, None
rag = _RAG(store)
tuner = ContextRuntimeRetrieverTuner(rag=rag, reason_llm=reason_llm)

# deterministic per-query cache of each arm's NDCG (so re-pulls are free + reproducible)
_cache: dict = {}
def arm_ndcg(cfg: RetrievalConfig, query: str, gold: set) -> float:
    ckey = (cfg.key, query)
    if ckey in _cache: return _cache[ckey]
    from redevops_rag.retrieve import hybrid_search
    if cfg.diver:
        hits = diver_search(store, query, reason_llm, limit=cfg.limit, pool=cfg.pool)
    else:
        hits = hybrid_search(store, query, reranker=None, **cfg.kwargs())
    v = ndcg(hits, gold); _cache[ckey] = v; return v

# fixed baselines (deterministic)
sim_cfg = RetrievalConfig(pool=50, limit=8)          # ~SIM/hybrid default
diver_cfg = RetrievalConfig(pool=25, limit=8, diver=True)
sim_q = st.mean(arm_ndcg(sim_cfg, q, g) for q, g in queries)
diver_q = st.mean(arm_ndcg(diver_cfg, q, g) for q, g in queries)

# stream queries through the bandit (deterministic order via hash → reproducible shuffle)
stream = sorted(queries * EPOCHS, key=lambda qg: hashlib.md5((qg[0]).encode()).hexdigest())
curve, arm_hist = [], []
for t, (q, gold) in enumerate(stream):
    cfg = tuner.choose(q)                    # bandit selects an arm for this query's intent bucket
    quality = arm_ndcg(cfg, q, gold)         # the arm's retrieval NDCG (deterministic, cached)
    tuner.record_outcome(q, quality=quality)  # feed reward (quality − cost) back to the bandit
    curve.append(quality); arm_hist.append("diver" if cfg.diver else ("rr" if cfg.rerank else "hy"))

def win(a, b): return f"{'+' if a>=b else ''}{a-b:+.3f}"
n = len(curve); h = n // 3
print(f"\n=== v5 bandit retrieval learning curve — TEMPO/{DOMAIN} (NDCG@10 reward) ===")
print(f"fixed SIM (hybrid default): {sim_q:.3f}   fixed DIVER (oracle): {diver_q:.3f}")
print(f"bandit NDCG — 1st third: {st.mean(curve[:h]):.3f}  →  last third: {st.mean(curve[-h:]):.3f}  "
      f"(learning Δ {st.mean(curve[-h:])-st.mean(curve[:h]):+.3f})")
print(f"bandit overall: {st.mean(curve):.3f}   vs SIM {win(st.mean(curve), sim_q)}   vs DIVER {win(st.mean(curve), diver_q)}")
from collections import Counter
print(f"arm mix — 1st third: {dict(Counter(arm_hist[:h]))}   last third: {dict(Counter(arm_hist[-h:]))}")
print(f"learned policy (bucket→arm): {tuner.policy()}")
