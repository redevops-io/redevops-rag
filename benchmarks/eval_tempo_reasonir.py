"""TEMPO retrieval eval with ReasonIR-8B GPU embeddings (vLLM :8012, arXiv:2504.20595).
Compares SIM-RAG (BM25) / hybrid / DIVER, all on ReasonIR embeddings. NDCG@10/Recall@10/MRR.
Run: .venv/bin/python benchmarks/eval_tempo_reasonir.py   (TEMPO_DOMAIN, TEMPO_NQ env override)."""
import sys, os, math, ast, json, urllib.request, statistics as st
from openai import OpenAI
from huggingface_hub import hf_hub_download
import duckdb
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
from redevops_rag.store import Store
from redevops_rag.retrieve import hybrid_search, diver_search

DOMAIN = os.environ.get("TEMPO_DOMAIN", "workplace")
N_Q = int(os.environ.get("TEMPO_NQ", "36"))
REASONIR_URL = os.environ.get("REASONIR_URL", "http://192.168.40.105:8012/v1/embeddings")
LLM_URL = os.environ.get("REASON_LLM_URL", "http://192.168.40.105:30807/v1")
ANS_MODEL = os.environ.get("REASON_LLM_MODEL", "Qwen3.6-35B-A3B")
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
llm = OpenAI(base_url=LLM_URL, api_key="EMPTY")

def reason_llm(system, user):
    r = llm.chat.completions.create(model=ANS_MODEL, temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

class ReasonIREmbedder:
    dim = 4096
    def __init__(self, url=REASONIR_URL, model="reasonir", batch=48):
        self.url, self.model, self.batch = url, model, batch
    def encode(self, texts):
        texts, out = list(texts), []
        for i in range(0, len(texts), self.batch):
            chunk = [t[:6000] for t in texts[i:i + self.batch]]
            req = urllib.request.Request(self.url, data=json.dumps({"model": self.model, "input": chunk}).encode(),
                                         headers={"Content-Type": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=180))
            out += [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        return out

def ndcg(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0
def recall(ranked, gold, k=10): return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0
def mrr(ranked, gold):
    for i, d in enumerate(ranked):
        if d in gold: return 1.0 / (i + 1)
    return 0.0

docs_p = hf_hub_download("tempo26/Tempo", f"documents/{DOMAIN}.parquet", repo_type="dataset")
ex_p = hf_hub_download("tempo26/Tempo", f"examples/{DOMAIN}.parquet", repo_type="dataset")
con = duckdb.connect()
docs = con.execute(f"SELECT id, content FROM '{docs_p}'").fetchall()
exs = con.execute(f"SELECT query, gold_ids FROM '{ex_p}' LIMIT {N_Q}").fetchall()
print(f"domain={DOMAIN} docs={len(docs)} queries={len(exs)} | embedder=ReasonIR-8B(GPU)", flush=True)

emb = ReasonIREmbedder()
dbp = f"/tmp/tempo_ri_{DOMAIN}.duckdb"
if os.path.exists(dbp): os.remove(dbp)
store = Store(emb, dbp)
texts = [c for _, c in docs]
print("embedding corpus via ReasonIR (GPU)...", flush=True)
vecs = emb.encode(texts)
store.add_chunks([{"document_id": docs[i][0], "text": texts[i], "embedding": vecs[i], "metadata": {}}
                  for i in range(len(docs))], reindex=True)
print("indexed.", flush=True)

CONFIGS = {
    "sim_bm25": lambda q: store.bm25_search(q, limit=10),
    "hybrid":   lambda q: hybrid_search(store, q, limit=10, recency_half_life_days=0),
    "diver":    lambda q: diver_search(store, q, reason_llm, limit=10, pool=25),
}
res = {c: {"ndcg": [], "recall": [], "mrr": []} for c in CONFIGS}
for i, (query, gold_raw) in enumerate(exs):
    gold = set(ast.literal_eval(gold_raw) if isinstance(gold_raw, str) else list(gold_raw))
    for cname, fn in CONFIGS.items():
        ranked = [h.get("document_id") or h.get("id") for h in fn(query)]
        res[cname]["ndcg"].append(ndcg(ranked, gold))
        res[cname]["recall"].append(recall(ranked, gold))
        res[cname]["mrr"].append(mrr(ranked, gold))
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(exs)}", flush=True)

print(f"\n=== TEMPO / {DOMAIN} — ReasonIR-8B embeddings ({len(exs)} q) ===")
print(f"{'config':12}{'NDCG@10':>10}{'Recall@10':>12}{'MRR':>8}")
for c in CONFIGS:
    print(f"{c:12}{st.mean(res[c]['ndcg']):>10.3f}{st.mean(res[c]['recall']):>12.3f}{st.mean(res[c]['mrr']):>8.3f}")
