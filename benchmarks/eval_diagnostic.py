"""VALIDITY diagnostic: do the answer models actually USE the retrieved context, or answer
from parametric memory? For each (dataset, model) compare three conditions:

  closed   — NO context (pure parametric memory / contamination signal)
  oracle   — ONLY the gold docs as context (does the model use PERFECT context?)
  wrong    — a DIFFERENT question's gold docs (counterfactual: does bad context mislead it?)
  retrieved— the real hybrid pipeline

Reads:  closed ~= oracle on popqa            -> answered from memory (contamination)
        oracle >> closed on musique          -> model DOES use context; retrieval is the gap
        oracle ~= closed everywhere          -> deployment broken (context ignored)
        wrong  ~= closed (not oracle)         -> model follows memory, ignores given context
Dumps sample transcripts so the behavior is eyeballable.

  MODELS=qwen DATASETS=popqa,musique,longmemeval N=12 \
    .venv/bin/python benchmarks/eval_diagnostic.py
"""
import sys, os, json, hashlib, statistics as st
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench")
from openai import OpenAI
from redevops_rag.store import Store
from redevops_rag.embed import Embedder
from redevops_rag.retrieve import hybrid_search

DATADIR = os.environ.get("DATADIR", "/tmp/claude-1000/-mnt-backup-projects-ffmpeg-mcp-aws/4b091f87-0b28-4473-a19e-4caba574b251/scratchpad/datasets")
DATASETS = os.environ.get("DATASETS", "popqa,musique,longmemeval").split(",")
MODELS = os.environ.get("MODELS", "qwen").split(",")
N = int(os.environ.get("N", "12"))
K = int(os.environ.get("K", "6"))
BUDGET = int(os.environ.get("BUDGET", "3000"))
DUMP = int(os.environ.get("DUMP", "2"))
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
MODEL_CFG = {"qwen": {"url": "http://192.168.40.105:30807/v1", "model": "Qwen3.6-35B-A3B", "extra": NOTHINK},
             "deepseek": {"url": "http://192.168.40.105:8001/v1", "model": "DeepSeek-V4-Flash", "extra": {}}}
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
clients = {m: OpenAI(base_url=MODEL_CFG[m]["url"], api_key="EMPTY", timeout=600) for m in MODELS}
emb = Embedder()

def answer(model, ctx, q):
    c = MODEL_CFG[model]
    sys_p = ("Answer using ONLY the provided context, in as few words as possible. If the answer is "
             "not in the context, reply exactly: NOT FOUND.") if ctx else \
            ("Answer in as few words as possible. If you do not know, reply exactly: NOT FOUND.")
    user = (f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:") if ctx else f"Question: {q}\nAnswer:"
    r = clients[model].chat.completions.create(model=c["model"], temperature=0, max_tokens=80, extra_body=c["extra"],
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def judge(question, golds, cand):
    if not cand or cand.strip().upper() == "NOT FOUND": return False
    golds = [str(g) for g in golds if g is not None and str(g).strip()]
    if any(g.lower() in cand.lower() for g in golds): return True
    try:
        r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
            messages=[{"role": "system", "content": "Strict QA grader. Reply exactly CORRECT or INCORRECT: is the "
                       "MODEL answer correct given any GOLD variant (phrasing/language may differ)?"},
                      {"role": "user", "content": f"Q: {question}\nGOLD: {' / '.join(golds)}\nMODEL: {cand}\nVerdict:"}])
        v = (r.choices[0].message.content or "").strip().upper()
        return "CORRECT" in v and "INCORRECT" not in v
    except Exception: return False

def ctx_of(texts, budget):
    out, cap, used = [], budget * 4, 0
    for t in texts:
        if used >= cap: break
        b = t[:cap - used]; out.append(b); used += len(b)
    return "\n\n".join(out)

def load(name):
    rows = [json.loads(l) for l in open(f"{DATADIR}/{name}.jsonl")]
    rows.sort(key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())
    return [r for r in rows if str(r.get("answer", "")).strip()][:N]

for model in MODELS:
    print(f"\n########## model={model} ##########", flush=True)
    for name in DATASETS:
        items = load(name)
        goldtext = {it["qid"]: [d["text"] for d in it["docs"] if d["chunk_id"] in set(it["gold"])] for it in items}
        conds = {c: [] for c in ["closed", "oracle", "wrong", "retrieved"]}
        for i, it in enumerate(items):
            golds = [it["answer"], *it.get("aliases", [])]
            # retrieved
            dbp = f"/tmp/diag_{it['qid']}.duckdb"
            if os.path.exists(dbp): os.remove(dbp)
            s = Store(emb, dbp)
            ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in it["docs"]]
            for c, e in zip(ch, emb.encode([x["text"] for x in ch])): c["embedding"] = e
            s.add_chunks(ch, reindex=True)
            retr = ctx_of([h["text"] for h in hybrid_search(s, it["question"], limit=K, pool=25)], BUDGET); s.close()
            wrong_src = items[(i + 1) % len(items)]  # another question's gold = counterfactual context
            ctxs = {"closed": "", "oracle": ctx_of(goldtext[it["qid"]], BUDGET),
                    "wrong": ctx_of(goldtext[wrong_src["qid"]], BUDGET), "retrieved": retr}
            for cond, ctx in ctxs.items():
                a = answer(model, ctx, it["question"])
                conds[cond].append(judge(it["question"], golds, a))
                if i < DUMP:
                    print(f"  [{name}/{cond}] Q={it['question'][:60]!r} GOLD={str(it['answer'])[:40]!r} -> {a[:60]!r} {'OK' if conds[cond][-1] else 'x'}", flush=True)
        print(f"== {name:12} closed={st.mean(conds['closed']):.2f}  oracle={st.mean(conds['oracle']):.2f}  "
              f"wrong={st.mean(conds['wrong']):.2f}  retrieved={st.mean(conds['retrieved']):.2f}  (n={len(items)})", flush=True)
print("DIAGNOSTIC_DONE")
