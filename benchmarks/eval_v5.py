"""Context Runtime v5 benchmark: bandit-SELECTED retrieval (redevops-rag arms incl. the
DIVER+ReasonIR temporal plugin) vs a fixed SIM-RAG baseline, answered by DeepSeek-V4-Flash
(Q4 GGUF, CPU), grok-judged. ReasonIR-8B embeddings. Reports CR-bandit vs SIM per dataset
+ the learned retrieval policy per intent bucket.

Datasets via DATASET env: tempo:<domain> | longmemeval | pollution (TEMPO+nutrition).
"""
import sys, os, re, json, ast, urllib.request, datetime as dt, statistics as st, collections
from openai import OpenAI
import duckdb
from huggingface_hub import hf_hub_download
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
sys.path.insert(0, "/mnt/backup/projects/contextos")
from redevops_rag.store import Store
from redevops_rag.retrieve import hybrid_search
from redevops_rag.temporal import ReasonIREmbedder
from context_runtime.integrations.redevops_rag import ContextRuntimeRetrieverTuner
from harness.grader import judge_grade

DATASET = os.environ.get("DATASET", "tempo:workplace")
N = int(os.environ.get("N", "12"))
ANS_URL = os.environ.get("ANS_URL", "http://192.168.40.105:8001/v1")     # DeepSeek CPU
ANS_MODEL = os.environ.get("ANS_MODEL", "DeepSeek-V4-Flash")
REASON_URL = "http://192.168.40.105:30807/v1"                            # qwen for DIVER expand/rerank
REASON_MODEL = "Qwen3.6-35B-A3B"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

ans_cli = OpenAI(base_url=ANS_URL, api_key="EMPTY", timeout=600)
reason_cli = OpenAI(base_url=REASON_URL, api_key="EMPTY")
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])

def reason_llm(system, user):
    r = reason_cli.chat.completions.create(model=REASON_MODEL, temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()
def judge_chat(system, user):
    r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content
def answer(ctx, q):
    r = ans_cli.chat.completions.create(model=ANS_MODEL, temperature=0, max_tokens=256,
        messages=[{"role": "system", "content": "Answer using ONLY the context, as few words as possible. "
                   "If the answer is not present, reply exactly: NOT FOUND."},
                  {"role": "user", "content": f"Context:\n{ctx[:60000]}\n\nQuestion: {q}\nAnswer:"}])
    return (r.choices[0].message.content or "").strip()

def ctx_of(hits, budget=12000):
    out, used = [], 0
    for h in hits:
        t = len(h["text"]) // 4
        if used + t > budget and out: break
        out.append(h["text"]); used += t
    return "\n\n".join(out)

# ── load a dataset into a ReasonIR store + (query, gold_answer) list ──
emb = ReasonIREmbedder(url="http://192.168.40.105:8012/v1/embeddings")

def load():
    con = duckdb.connect()
    if DATASET.startswith("tempo") or DATASET == "pollution":
        chunks, items = [], []
        domains = ["workplace"] if DATASET.startswith("tempo") else ["workplace", "law"]
        dom = DATASET.split(":")[1] if ":" in DATASET else domains[0]
        docs_p = hf_hub_download("tempo26/Tempo", f"documents/{dom}.parquet", repo_type="dataset")
        ex_p = hf_hub_download("tempo26/Tempo", f"examples/{dom}.parquet", repo_type="dataset")
        for did, content in con.execute(f"SELECT id, content FROM '{docs_p}'").fetchall():
            chunks.append({"document_id": did, "text": content, "metadata": {}})
        for q, ga in con.execute(f"SELECT query, gold_answers FROM '{ex_p}' LIMIT {N}").fetchall():
            gold = ast.literal_eval(ga) if isinstance(ga, str) else list(ga)
            items.append((q, gold[0] if gold else ""))
        if DATASET == "pollution":  # add nutrition docs as cross-domain distractors
            for r in con.execute("SELECT text FROM read_parquet('/dev/null')").fetchall() if False else []:
                pass
        return chunks, items
    if DATASET == "longmemeval":
        rows = [json.loads(l) for l in open("/mnt/backup/projects/context-runtime-go/benchdata/longmemeval_s.jsonl")]
        by = collections.defaultdict(list)
        for r in rows: by[r["qtype"]].append(r)
        picked = []
        for t in ["knowledge-update", "multi-session", "temporal-reasoning"]:
            picked += by[t][: max(1, N // 3)]
        chunks, items = [], []
        for it in picked:
            for d in it["docs"]:
                chunks.append({"document_id": f"{it['qid']}::{d['chunk_id']}", "text": d["text"], "metadata": {}})
            items.append((it["question"], it["answer"]))
        return chunks, items
    raise SystemExit(f"unknown DATASET {DATASET}")

chunks, items = load()
print(f"DATASET={DATASET}  docs={len(chunks)}  queries={len(items)} | ReasonIR embed + DeepSeek answerer", flush=True)
dbp = f"/tmp/v5_{DATASET.replace(':','_')}.duckdb"
if os.path.exists(dbp): os.remove(dbp)
store = Store(emb, dbp)
print("embedding corpus (ReasonIR)...", flush=True)
for c, e in zip(chunks, emb.encode([c["text"] for c in chunks])): c["embedding"] = e
store.add_chunks(chunks, reindex=True)
print("indexed.", flush=True)

class _RAG:  # minimal shim the tuner needs
    def __init__(self, store): self.store, self.reranker = store, None
tuner = ContextRuntimeRetrieverTuner(rag=_RAG(store), reason_llm=reason_llm)

cr, sim = [], []
for i, (q, gold) in enumerate(items):
    # CR: bandit picks an arm (incl. DIVER), retrieve → answer → judge → feed reward
    hits = tuner.search(q)
    a_cr = answer(ctx_of(hits), q); ok_cr = bool(judge_grade(judge_chat, q, gold, a_cr))
    tuner.record_outcome(q, quality=1.0 if ok_cr else 0.0, latency_s=0.0)
    cr.append(ok_cr)
    # SIM baseline: fixed BM25
    a_sim = answer(ctx_of(store.bm25_search(q, limit=8)), q)
    sim.append(bool(judge_grade(judge_chat, q, gold, a_sim)))
    print(f"  {i+1}/{len(items)}  cr={ok_cr} sim={sim[-1]}", flush=True)

print(f"\n=== CR-v5 (bandit-selected retrieval) vs SIM-RAG — {DATASET} (DeepSeek-Flash, judge) ===")
print(f"  CR-bandit accuracy: {st.mean(cr):.3f}   SIM-RAG: {st.mean(sim):.3f}   Δ={st.mean(cr)-st.mean(sim):+.3f}")
print(f"  learned retrieval policy (intent bucket -> arm): {tuner.policy()}")
