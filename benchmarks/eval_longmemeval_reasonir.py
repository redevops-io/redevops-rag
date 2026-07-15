"""LongMemEval-S with ReasonIR-8B GPU embeddings (bidirectional, vLLM :8012).
SIM-RAG / hybrid / DIVER, grok-judge-graded. Compare to bge-small (eval_longmemeval_diver.py)."""
import sys, os, re, json, urllib.request, datetime as dt, statistics as st, collections
from openai import OpenAI
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
from redevops_rag.store import Store
from redevops_rag.retrieve import hybrid_search, diver_search
from harness.grader import judge_grade

DATA = "/mnt/backup/projects/context-runtime-go/benchdata/longmemeval_s.jsonl"
N_PER_TYPE = int(os.environ.get("N_PER_TYPE", "20"))
REF = dt.datetime(2026, 1, 1)
ANS_MODEL = "Qwen3.6-35B-A3B"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
llm = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])

def reason_llm(system, user):
    r = llm.chat.completions.create(model=ANS_MODEL, temperature=0, max_tokens=120, extra_body=NOTHINK,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()
def judge_chat(system, user):
    r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content
def answer(ctx, q):
    return reason_llm("Answer the question using ONLY the provided context. Reply with just the answer in as "
                      "few words as possible. If the answer is not in the context, reply exactly: NOT FOUND.",
                      f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

class ReasonIREmbedder:
    dim = 4096
    def __init__(self, url="http://192.168.40.105:8012/v1/embeddings", model="reasonir", batch=48):
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

def parse(s):
    try: return dt.datetime.strptime(s, "%Y/%m/%d (%a) %H:%M")
    except Exception: return None
def ctx(hits, budget=22000):
    out, used = [], 0
    for h in hits:
        t = len(h["text"]) // 4
        if used + t > budget and out: break
        out.append(h["text"]); used += t
    return "\n\n".join(out)

rows = [json.loads(l) for l in open(DATA)]
by = collections.defaultdict(list)
for r in rows: by[r["qtype"]].append(r)
items = []
for t in ["knowledge-update", "multi-session", "temporal-reasoning"]:
    items += by[t][:N_PER_TYPE]
print(f"items: {len(items)} | embedder=ReasonIR-8B(GPU,bidir)", flush=True)

emb = ReasonIREmbedder()
dbp = "/tmp/lme_ri.duckdb"
if os.path.exists(dbp): os.remove(dbp)
store = Store(emb, dbp)
chunks, scope = [], {}
for it in items:
    qd = parse(it["question_date"]); ids = []
    for d in it["docs"]:
        sd = parse(d["created_at"]); ca = (REF - (qd - sd)) if (qd and sd) else REF
        did = f"{it['qid']}::{d['chunk_id']}"
        chunks.append({"document_id": did, "text": d["text"], "created_at": ca, "metadata": {}}); ids.append(did)
    scope[it["qid"]] = ids
print(f"embedding {len(chunks)} sessions via ReasonIR...", flush=True)
for c, e in zip(chunks, emb.encode([c["text"] for c in chunks])): c["embedding"] = e
store.add_chunks(chunks, reindex=True)
print("ingested.", flush=True)

CONFIGS = {
    "sim_bm25": lambda q, ids: store.bm25_search(q, limit=6, document_ids=ids),
    "hybrid":   lambda q, ids: hybrid_search(store, q, limit=6, recency_half_life_days=0, document_ids=ids),
    "diver":    lambda q, ids: diver_search(store, q, reason_llm, limit=6, pool=25, document_ids=ids),
}
res = {c: collections.defaultdict(list) for c in CONFIGS}
for i, it in enumerate(items):
    ids = scope[it["qid"]]
    for cname, fn in CONFIGS.items():
        a = answer(ctx(fn(it["question"], ids)), it["question"])
        res[cname][it["qtype"]].append(bool(judge_grade(judge_chat, it["question"], it["answer"], a)))
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(items)}", flush=True)

print("\n=== LongMemEval-S — ReasonIR-8B embeddings (judge-graded) ===")
print(f"{'config':12}{'knowledge-up':>14}{'multi-sess':>12}{'temporal-re':>12}{'ALL':>8}")
for c in CONFIGS:
    d = res[c]; allv = [x for v in d.values() for x in v]
    g = lambda t: f"{st.mean(d[t]):.2f}" if d[t] else "-"
    print(f"{c:12}{g('knowledge-update'):>14}{g('multi-session'):>12}{g('temporal-reasoning'):>12}{st.mean(allv):>8.3f}")
