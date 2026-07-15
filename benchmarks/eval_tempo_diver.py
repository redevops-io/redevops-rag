"""TEMPO (tempo26/Tempo) retrieval eval, scoped to one domain.
Compares SIM-RAG (BM25) vs hybrid vs DIVER-style (query expansion + hybrid + listwise rerank).
Metrics: NDCG@10, Recall@10, MRR vs gold_ids (binary relevance)."""
import sys, os, re, math, ast, statistics as st
from openai import OpenAI
from huggingface_hub import hf_hub_download
import duckdb
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
from redevops_rag.rag import RAG
from redevops_rag.retrieve import hybrid_search

DOMAIN = os.environ.get("TEMPO_DOMAIN", "workplace")
N_Q = int(os.environ.get("TEMPO_NQ", "36"))
ANS_MODEL = "Qwen3.6-35B-A3B"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
llm = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")

def chat(system, user, max_tokens=120, temp=0.0):
    r = llm.chat.completions.create(model=ANS_MODEL, temperature=temp, max_tokens=max_tokens,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def expand_query(query):
    out = chat("Decompose the search query into 2-3 focused sub-queries covering the entities and the "
               "relevant time periods needed to answer it. One per line, no numbering.", query[:1500], temp=0.3)
    return [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:3]

def listwise_rerank(query, cands, top):
    if len(cands) <= top: return cands
    snip = "\n".join(f"[{i}] {c['text'][:280]}" for i, c in enumerate(cands))
    out = chat(f"Rank the passages by usefulness for the query. Return the {top} best passage numbers, "
               f"comma-separated, best first. Numbers only.", f"Query: {query[:800]}\n\nPassages:\n{snip}", max_tokens=80)
    order = [int(x) for x in re.findall(r"\d+", out)]
    seen, ranked = set(), []
    for i in order:
        if 0 <= i < len(cands) and i not in seen: seen.add(i); ranked.append(cands[i])
    for i, c in enumerate(cands):
        if i not in seen: ranked.append(c)
    return ranked[:top]

def diver(store, query, pool=25, top=10):
    cand = {}
    for q in [query] + expand_query(query):
        for h in hybrid_search(store, q, limit=pool, recency_half_life_days=0):
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
        if d in gold: return 1.0 / (i + 1)
    return 0.0

# ---- load domain ----
docs_p = hf_hub_download("tempo26/Tempo", f"documents/{DOMAIN}.parquet", repo_type="dataset")
ex_p = hf_hub_download("tempo26/Tempo", f"examples/{DOMAIN}.parquet", repo_type="dataset")
con = duckdb.connect()
docs = con.execute(f"SELECT id, content FROM '{docs_p}'").fetchall()
exs = con.execute(f"SELECT query, gold_ids FROM '{ex_p}' LIMIT {N_Q}").fetchall()
print(f"domain={DOMAIN}  docs={len(docs)}  queries={len(exs)}", flush=True)

rag = RAG(db_path=f"/tmp/tempo_{DOMAIN}.duckdb")
texts = [c for _, c in docs]
print("embedding corpus...", flush=True)
embs = rag.embedder.encode(texts)
chunks = [{"document_id": docs[i][0], "text": texts[i], "embedding": embs[i], "metadata": {}} for i in range(len(docs))]
print("adding to store...", flush=True)
rag.store.add_chunks(chunks, reindex=True)
print("indexed.", flush=True)

CONFIGS = {
    "sim_bm25": lambda q: rag.store.bm25_search(q, limit=10),
    "hybrid":   lambda q: hybrid_search(rag.store, q, limit=10, recency_half_life_days=0),
    "diver":    lambda q: diver(rag.store, q, pool=25, top=10),
}
res = {c: {"ndcg": [], "recall": [], "mrr": []} for c in CONFIGS}
for i, (query, gold_raw) in enumerate(exs):
    gold = set(ast.literal_eval(gold_raw) if isinstance(gold_raw, str) else list(gold_raw))
    for cname, fn in CONFIGS.items():
        ranked = [h["document_id"] for h in fn(query)]
        res[cname]["ndcg"].append(ndcg_at_k(ranked, gold))
        res[cname]["recall"].append(recall_at_k(ranked, gold))
        res[cname]["mrr"].append(mrr(ranked, gold))
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(exs)}", flush=True)

print(f"\n=== TEMPO / {DOMAIN} retrieval ({len(exs)} queries) ===")
print(f"{'config':12}{'NDCG@10':>10}{'Recall@10':>12}{'MRR':>8}")
for c in CONFIGS:
    print(f"{c:12}{st.mean(res[c]['ndcg']):>10.3f}{st.mean(res[c]['recall']):>12.3f}{st.mean(res[c]['mrr']):>8.3f}")
