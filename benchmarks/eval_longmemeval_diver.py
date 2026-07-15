"""DIVER-style reasoning-intensive retrieval on redevops-rag, evaluated on LongMemEval-S.
Pipeline: LLM query expansion → hybrid retrieve (union of sub-queries) → LLM listwise rerank.
Compared to SIM-RAG (BM25) and plain hybrid, grok-judge-graded (fair)."""
import json, sys, os, re, datetime as dt, statistics as st, collections
from openai import OpenAI
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
from redevops_rag.rag import RAG
from redevops_rag.retrieve import hybrid_search
from harness.grader import judge_grade

DATA = "/mnt/backup/projects/context-runtime-go/benchdata/longmemeval_s.jsonl"
N_PER_TYPE = 20
REF = dt.datetime(2026, 1, 1)
ANS_MODEL = "Qwen3.6-35B-A3B"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

ans_cli = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])

def chat(system, user, max_tokens=256, temp=0.0):
    r = ans_cli.chat.completions.create(model=ANS_MODEL, temperature=temp, max_tokens=max_tokens,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def judge_chat(system, user):
    r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

def answer(ctx, q):
    return chat("Answer the question using ONLY the provided context. Reply with just the answer in as "
                "few words as possible. If the answer is not in the context, reply exactly: NOT FOUND.",
                f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

# ---- DIVER stages ----
def expand_query(query):
    out = chat("Decompose the question into 2-3 focused search queries that would retrieve the evidence "
               "needed to answer it. Consider the relevant time periods and entities. One query per line, no numbering.",
               query, max_tokens=120, temp=0.3)
    subs = [l.strip("-• ").strip() for l in out.splitlines() if l.strip()]
    return subs[:3]

def listwise_rerank(query, cands, top):
    if len(cands) <= top:
        return cands
    snip = "\n".join(f"[{i}] {c['text'][:300]}" for i, c in enumerate(cands))
    out = chat(f"Rank the passages by how useful each is for answering the question. Return the {top} most "
               f"useful passage numbers, comma-separated, best first. Numbers only.",
               f"Question: {query}\n\nPassages:\n{snip}", max_tokens=60)
    order = [int(x) for x in re.findall(r"\d+", out)]
    seen, ranked = set(), []
    for i in order:
        if 0 <= i < len(cands) and i not in seen:
            seen.add(i); ranked.append(cands[i])
    for i, c in enumerate(cands):  # fallback: keep any not chosen
        if i not in seen:
            ranked.append(c)
    return ranked[:top]

def diver_retrieve(store, query, ids, pool=30, top=6):
    queries = [query] + expand_query(query)
    cand = {}
    for q in queries:
        for h in hybrid_search(store, q, limit=pool, recency_half_life_days=0, document_ids=ids):
            cand.setdefault(h["document_id"], h)
    return listwise_rerank(query, list(cand.values()), top)

def parse(s):
    try: return dt.datetime.strptime(s, "%Y/%m/%d (%a) %H:%M")
    except Exception: return None

def ctx(hits, budget_tok=22000):
    out, used = [], 0
    for h in hits:
        t = len(h["text"]) // 4
        if used + t > budget_tok and out: break
        out.append(h["text"]); used += t
    return "\n\n".join(out)

# ---- data + ingest ----
rows = [json.loads(l) for l in open(DATA)]
by = collections.defaultdict(list)
for r in rows: by[r["qtype"]].append(r)
items = []
for t in ["knowledge-update", "multi-session", "temporal-reasoning"]:
    items += by[t][:N_PER_TYPE]
print(f"items: {len(items)}", flush=True)

rag = RAG(db_path="/tmp/lme_diver.duckdb")
chunks, scope = [], {}
for it in items:
    qd = parse(it["question_date"]); ids = []
    for d in it["docs"]:
        sd = parse(d["created_at"])
        ca = (REF - (qd - sd)) if (qd and sd) else REF
        did = f"{it['qid']}::{d['chunk_id']}"
        chunks.append({"document_id": did, "text": d["text"], "created_at": ca, "metadata": {}})
        ids.append(did)
    scope[it["qid"]] = ids
print(f"ingesting {len(chunks)} sessions...", flush=True)
for c, e in zip(chunks, rag.embedder.encode([c["text"] for c in chunks])):
    c["embedding"] = e
rag.store.add_chunks(chunks, reindex=True)
print("ingested.", flush=True)

CONFIGS = {
    "sim_bm25": lambda q, ids: rag.store.bm25_search(q, limit=6, document_ids=ids),
    "hybrid":   lambda q, ids: hybrid_search(rag.store, q, limit=6, recency_half_life_days=0, document_ids=ids),
    "diver":    lambda q, ids: diver_retrieve(rag.store, q, ids, pool=30, top=6),
}

res = {c: collections.defaultdict(list) for c in CONFIGS}
for i, it in enumerate(items):
    ids = scope[it["qid"]]
    for cname, fn in CONFIGS.items():
        a = answer(ctx(fn(it["question"], ids)), it["question"])
        res[cname][it["qtype"]].append(bool(judge_grade(judge_chat, it["question"], it["answer"], a)))
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(items)}", flush=True)

print("\n=== DIVER vs hybrid vs SIM-RAG on LongMemEval-S (judge-graded, qwen-reasoning) ===")
print(f"{'config':12}{'knowledge-up':>14}{'multi-sess':>12}{'temporal-re':>12}{'ALL':>8}")
for c in CONFIGS:
    d = res[c]; allv = [x for v in d.values() for x in v]
    g = lambda t: f"{st.mean(d[t]):.2f}" if d[t] else "-"
    print(f"{c:12}{g('knowledge-update'):>14}{g('multi-session'):>12}{g('temporal-reasoning'):>12}{st.mean(allv):>8.3f}")
