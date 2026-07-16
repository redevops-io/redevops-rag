"""Context-size effect on temporal task performance (a Context Runtime tunable).

Fixed retrieval (hybrid over the per-question sessions); vary how much of the ranked
context is passed to the answer model — the knob CR's sizer controls. Answer model set by
ANS: qwen (fast, 32k window) or deepseek (DeepSeek-V4-Flash Q4 CPU, the stronger long-ctx
model). grok-judged. Shows accuracy vs context budget — where "more context" stops helping.

  ANS=qwen N=20 .venv/bin/python benchmarks/eval_v5_context_size.py
"""
import sys, os, json, datetime as dt, statistics as st, collections
from openai import OpenAI
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
from redevops_rag.store import Store
from redevops_rag.embed import Embedder
from redevops_rag.retrieve import hybrid_search
from harness.grader import judge_grade

DATA = "/mnt/backup/projects/context-runtime-go/benchdata/longmemeval_s.jsonl"
N = int(os.environ.get("N", "20"))
ANS = os.environ.get("ANS", "qwen")
BUDGETS = [1000, 3000, 6000, 12000]  # tokens of retrieved context passed to the model
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
if ANS == "deepseek":
    ans_cli = OpenAI(base_url="http://192.168.40.105:8001/v1", api_key="EMPTY", timeout=600)
    ANS_MODEL, EXTRA = "DeepSeek-V4-Flash", {}
else:
    ans_cli = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")
    ANS_MODEL, EXTRA = "Qwen3.6-35B-A3B", NOTHINK
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])

def answer(ctx, q):
    r = ans_cli.chat.completions.create(model=ANS_MODEL, temperature=0, max_tokens=256, extra_body=EXTRA,
        messages=[{"role": "system", "content": "Answer using ONLY the context, as few words as possible. "
                   "If the answer is not present, reply exactly: NOT FOUND."},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"}])
    return (r.choices[0].message.content or "").strip()
def judge_chat(system, user):
    r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content
def parse(s):
    try: return dt.datetime.strptime(s, "%Y/%m/%d (%a) %H:%M")
    except Exception: return None
def ctx_of(hits, budget):
    out, used = [], 0
    for h in hits:
        t = len(h["text"]) // 4
        if used + t > budget and out: break
        out.append(h["text"]); used += t
    return "\n\n".join(out), used

rows = [json.loads(l) for l in open(DATA)]
by = collections.defaultdict(list)
for r in rows: by[r["qtype"]].append(r)
items = []
for t in ["knowledge-update", "multi-session", "temporal-reasoning"]:
    items += by[t][: max(1, N // 3)]
print(f"items={len(items)} | answerer={ANS_MODEL} | budgets={BUDGETS}", flush=True)

emb = Embedder()  # cheap bge — retrieval is fixed; only the passed size varies
res = {b: [] for b in BUDGETS}
toks = {b: [] for b in BUDGETS}
for i, it in enumerate(items):
    dbp = f"/tmp/cs_{it['qid']}.duckdb"
    if os.path.exists(dbp): os.remove(dbp)
    store = Store(emb, dbp)
    ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in it["docs"]]
    for c, e in zip(ch, emb.encode([c["text"] for c in ch])): c["embedding"] = e
    store.add_chunks(ch, reindex=True)
    ranked = hybrid_search(store, it["question"], limit=30, pool=40, recency_half_life_days=0)  # fixed retrieval
    for b in BUDGETS:
        ctx, used = ctx_of(ranked, b)
        ok = bool(judge_grade(judge_chat, it["question"], it["answer"], answer(ctx, it["question"])))
        res[b].append(ok); toks[b].append(used)
    store.close()
    if (i + 1) % 5 == 0: print(f"  {i+1}/{len(items)}", flush=True)

print(f"\n=== context-size effect — LongMemEval-S ({len(items)} q, {ANS_MODEL}, judge-graded) ===")
print(f"{'budget (tok)':>14}{'ctx passed':>12}{'accuracy':>10}")
for b in BUDGETS:
    print(f"{b:>14}{int(st.mean(toks[b])):>12}{st.mean(res[b]):>10.3f}")
