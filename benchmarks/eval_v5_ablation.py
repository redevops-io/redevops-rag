"""Clean ablation: SIM vs {bge, ReasonIR} × {hybrid, DIVER} on TEMPO/workplace, IDENTICAL
configs (limit=10, pool=25), so embedder and pipeline effects are isolated. Answers: does
solo DIVER (bge) beat the DIVER+ReasonIR combo? NDCG@10 / Recall@10 / MRR vs gold_ids."""
import sys, os, math, ast, statistics as st
import duckdb
from huggingface_hub import hf_hub_download
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
from redevops_rag.store import Store
from redevops_rag.embed import Embedder
from redevops_rag.retrieve import hybrid_search, diver_search
from redevops_rag.temporal import ReasonIREmbedder
from openai import OpenAI

DOMAIN = os.environ.get("DOMAIN", "workplace")
LIMIT, POOL = 10, 25
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
reason_cli = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")
def reason_llm(system, user):
    r = reason_cli.chat.completions.create(model="Qwen3.6-35B-A3B", temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def ndcg(ids, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ids[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0
def recall(ids, gold, k=10): return len(set(ids[:k]) & gold) / len(gold) if gold else 0.0
def mrr(ids, gold):
    for i, d in enumerate(ids):
        if d in gold: return 1.0 / (i + 1)
    return 0.0
def _ids(hits): return [h.get("document_id") or h.get("id") for h in hits]

docs_p = hf_hub_download("tempo26/Tempo", f"documents/{DOMAIN}.parquet", repo_type="dataset")
ex_p = hf_hub_download("tempo26/Tempo", f"examples/{DOMAIN}.parquet", repo_type="dataset")
con = duckdb.connect()
docs = con.execute(f"SELECT id, content FROM '{docs_p}'").fetchall()
queries = [(q, set(ast.literal_eval(g) if isinstance(g, str) else list(g)))
           for q, g in con.execute(f"SELECT query, gold_ids FROM '{ex_p}'").fetchall()]

def build(emb, path):
    s = Store(emb, path)
    if s.count() == 0:
        print(f"embedding {len(docs)} docs -> {os.path.basename(path)} ...", flush=True)
        ch = [{"document_id": d[0], "text": d[1], "metadata": {}} for d in docs]
        for c, e in zip(ch, emb.encode([c["text"] for c in ch])): c["embedding"] = e
        s.add_chunks(ch, reindex=True)
    return s

RES = "/mnt/backup/projects/redevops-rag/benchmarks/results"
store_bge = build(Embedder(), f"{RES}/v5_store_bge_{DOMAIN}.duckdb")
store_ri = build(ReasonIREmbedder(url="http://192.168.40.105:8012/v1/embeddings"), f"{RES}/v5_store_{DOMAIN}.duckdb")
print(f"stores ready | queries={len(queries)} | limit={LIMIT} pool={POOL}", flush=True)

METHODS = {
    "sim_bm25 (lexical)":     lambda q: store_ri.bm25_search(q, limit=LIMIT),
    "hybrid · bge":           lambda q: hybrid_search(store_bge, q, limit=LIMIT, pool=POOL, recency_half_life_days=0),
    "hybrid · ReasonIR":      lambda q: hybrid_search(store_ri, q, limit=LIMIT, pool=POOL, recency_half_life_days=0),
    "DIVER · bge (solo)":     lambda q: diver_search(store_bge, q, reason_llm, limit=LIMIT, pool=POOL),
    "DIVER · ReasonIR (combo)": lambda q: diver_search(store_ri, q, reason_llm, limit=LIMIT, pool=POOL),
}
res = {m: {"ndcg": [], "recall": [], "mrr": []} for m in METHODS}
for i, (q, gold) in enumerate(queries):
    for m, fn in METHODS.items():
        ids = _ids(fn(q))
        res[m]["ndcg"].append(ndcg(ids, gold)); res[m]["recall"].append(recall(ids, gold)); res[m]["mrr"].append(mrr(ids, gold))
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(queries)}", flush=True)

print(f"\n=== ablation — TEMPO/{DOMAIN} ({len(queries)} q, limit={LIMIT}, pool={POOL}) ===")
print(f"{'method':26}{'NDCG@10':>10}{'Recall@10':>12}{'MRR':>8}")
for m in METHODS:
    print(f"{m:26}{st.mean(res[m]['ndcg']):>10.3f}{st.mean(res[m]['recall']):>12.3f}{st.mean(res[m]['mrr']):>8.3f}")
d_solo, d_combo = st.mean(res["DIVER · bge (solo)"]["ndcg"]), st.mean(res["DIVER · ReasonIR (combo)"]["ndcg"])
print(f"\nsolo DIVER(bge) vs combo DIVER(ReasonIR): Δ NDCG@10 = {d_solo - d_combo:+.3f} "
      f"({'solo wins' if d_solo > d_combo else 'combo wins' if d_combo > d_solo else 'tie'})")
