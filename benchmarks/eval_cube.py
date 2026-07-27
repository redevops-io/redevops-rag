"""Answer-accuracy CUBE: datasets x retrieval-methods x answer-models -> end-to-end accuracy.

For each (dataset, method, model, question): retrieve context with the method, hand it to the
answer model, judge the answer (grok). This is the z-axis (models) on top of the retrieval plane.
Retrieval methods here are the Python-tractable ones (bm25/hybrid/reasonir/diver/cr-auto);
HippoRAG/Graphiti answer-cells come from routebench in a later pass.

  MODELS=qwen,deepseek METHODS=bm25,hybrid,reasonir,diver,cr-auto \
  DATASETS=popqa,musique,longmemeval,tempo,nutrition N=10 \
  <venv>/bin/python benchmarks/eval_cube.py
"""
import sys, os, json, hashlib, statistics as st, collections
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench")
from openai import OpenAI
from redevops_rag.store import Store
from redevops_rag.embed import Embedder
from redevops_rag.temporal import ReasonIREmbedder
from redevops_rag.retrieve import hybrid_search, diver_search
from context_runtime.planner.llm_intent import OpenAICompatModel

DATADIR = os.environ.get("DATADIR")
RES = os.environ.get("OUT", "/tmp/claude-1000/-mnt-backup-projects-ffmpeg-mcp-aws/4b091f87-0b28-4473-a19e-4caba574b251/scratchpad/cube_res")
DATASETS = os.environ.get("DATASETS", "popqa,musique,longmemeval,tempo,nutrition").split(",")
METHODS = os.environ.get("METHODS", "bm25,hybrid,reasonir,diver,cr-auto").split(",")
MODELS = os.environ.get("MODELS", "qwen,deepseek").split(",")
N = int(os.environ.get("N", "10"))
K = int(os.environ.get("K", "6"))
BUDGET = int(os.environ.get("BUDGET", "3000"))
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
os.makedirs(RES, exist_ok=True)

MODEL_CFG = {  # the 2 currently-served models; add more as they're stood up
    "qwen":     {"url": "http://192.168.40.105:30807/v1", "model": "Qwen3.6-35B-A3B", "extra": NOTHINK, "tier": "gpu"},
    "deepseek": {"url": "http://192.168.40.105:8001/v1",  "model": "DeepSeek-V4-Flash", "extra": {}, "tier": "cpu"},
}
QWEN = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")   # DIVER expand/rerank + router
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
router = OpenAICompatModel("http://192.168.40.105:30807/v1", "Qwen3.6-35B-A3B")
ans_clients = {m: OpenAI(base_url=c["url"], api_key="EMPTY", timeout=600) for m, c in MODEL_CFG.items() if m in MODELS}

def reason_llm(system, user):
    r = QWEN.chat.completions.create(model="Qwen3.6-35B-A3B", temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def answer(model, ctx, q):
    c = MODEL_CFG[model]
    r = ans_clients[model].chat.completions.create(model=c["model"], temperature=0, max_tokens=80, extra_body=c["extra"],
        messages=[{"role": "system", "content": "Answer using ONLY the context, in as few words as possible. "
                   "If the answer is not present, reply exactly: NOT FOUND."},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"}])
    return (r.choices[0].message.content or "").strip()

def judge(question, golds, cand):
    if not cand or cand.strip().upper() == "NOT FOUND": return False
    golds = [str(g) for g in golds if g is not None and str(g).strip()]
    low = cand.lower()
    if any(g.lower() in low for g in golds): return True
    try:
        r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
            messages=[{"role": "system", "content": "Strict QA grader. Reply exactly CORRECT or INCORRECT: is the "
                       "MODEL answer correct given any GOLD variant (phrasing/language/units may differ)?"},
                      {"role": "user", "content": f"Q: {question}\nGOLD: {' / '.join(golds)}\nMODEL: {cand}\nVerdict:"}])
        v = (r.choices[0].message.content or "").strip().upper()
        return "CORRECT" in v and "INCORRECT" not in v
    except Exception:
        return False

_bge = Embedder(); _ri = None
def store_for(item, embedder):
    dbp = f"/tmp/cube_{id(embedder)%9999}_{item['qid']}.duckdb"
    if os.path.exists(dbp): os.remove(dbp)
    s = Store(embedder, dbp)
    ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in item["docs"]]
    for c, e in zip(ch, embedder.encode([c["text"] for c in ch])): c["embedding"] = e
    s.add_chunks(ch, reindex=True); return s

def ctx_of(hits, budget):
    out, used, cap = [], 0, budget * 4  # budget is tokens; cap chars ~= 4*tokens
    for h in hits:
        remain = cap - used
        if remain <= 0: break
        block = h["text"][:remain]          # truncate every block (incl. the first) to fit the budget
        out.append(block); used += len(block)
    return "\n\n".join(out)

def retrieve_ctx(method, item):
    global _ri
    if method == "reasonir":
        if _ri is None: _ri = ReasonIREmbedder(url="http://192.168.40.105:8012/v1/embeddings")
        s = store_for(item, _ri); hits = hybrid_search(s, item["question"], limit=K, pool=25); s.close()
    elif method == "diver":
        s = store_for(item, _bge); hits = diver_search(s, item["question"], reason_llm, limit=K, pool=25); s.close()
    elif method == "cr-auto":
        rep = router.classify(item["question"]) or "document"
        s = store_for(item, _bge)
        if rep == "graph":    hits = diver_search(s, item["question"], reason_llm, limit=K, pool=25)
        elif rep == "temporal":
            # temporal → DIVER (the temporal_reasoning arm), now IN the routed set. The old
            # recency=1-day hybrid helped memory-temporal (longmemeval) but tanked temporal-REASONING
            # (tempo 0.84→0.68) by burying older-but-relevant evidence; DIVER's expand+rerank finds it
            # (tempo 0.91) and is strong on longmemeval too (0.92). Keep a MILD recency as a secondary
            # prior for the memory case — env-tunable (CR_TEMPORAL_RECENCY_HL) for the sweep; 0 disables.
            hl = float(os.environ.get("CR_TEMPORAL_RECENCY_HL", "30"))
            hits = diver_search(s, item["question"], reason_llm, limit=K, pool=25, recency_half_life_days=hl)
        else:                 hits = hybrid_search(s, item["question"], limit=K, pool=25)
        s.close()
    else:  # bm25 (baseline) or hybrid
        s = store_for(item, _bge)
        hits = s.bm25_search(item["question"], limit=K) if method == "bm25" else hybrid_search(s, item["question"], limit=K, pool=25)
        s.close()
    return ctx_of(hits, BUDGET)

def load(name):
    rows = [json.loads(l) for l in open(f"{DATADIR}/{name}.jsonl")]
    rows.sort(key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())
    return [r for r in rows if str(r.get("answer", "")).strip()][:N]

print(f"models={MODELS} methods={METHODS} datasets={DATASETS} N={N}", flush=True)
for name in DATASETS:
    items = load(name)
    for method in METHODS:
        ctxs = {it["qid"]: retrieve_ctx(method, it) for it in items}   # retrieve once, reuse across models
        for model in MODELS:
            correct = []
            for it in items:
                golds = [it["answer"], *it.get("aliases", [])]
                correct.append(judge(it["question"], golds, answer(model, ctxs[it["qid"]], it["question"])))
            acc = round(st.mean(correct), 3) if correct else None
            json.dump({"dataset": name, "method": method, "model": model, "n": len(correct), "acc": acc},
                      open(f"{RES}/{name}__{method}__{model}.json", "w"))
            print(f"  {name:12} {method:9} {model:9} acc={acc} (n={len(correct)})", flush=True)
print("CUBE_DONE", flush=True)
