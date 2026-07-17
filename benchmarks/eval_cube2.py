"""Answer cube v2 — datasets x methods x models, with per-cell CONDITIONS that expose the
bottleneck (a+b), Graphiti as a retrieval method, the musique both-hop fix (d), a 7-model
registry, and a bottleneck-aware CR router (c).

Per (dataset, method, model) we record accuracy under conditions:
  closed    — NO context (parametric-memory / contamination baseline)          [a]
  oracle    — ONLY the gold docs (is the ceiling model-bound or retrieval-bound?) [b]
  retrieved — the method's real retrieval

Reading a cell:  oracle >> retrieved -> RETRIEVAL-bound (better retriever helps)
                 oracle ~= closed (both low) -> MODEL-bound (better model / decompose helps)
                 closed high -> contamination (subtract it out)

Methods: bm25, hybrid, reasonir, diver, graphiti, cr-auto(bottleneck-aware).
Musique fix (d): retrieval K raised + union of sub-query hits so BOTH bridge hops are covered.

  MODELS=qwen METHODS=bm25,hybrid,reasonir,diver,cr-auto CONDS=closed,oracle,retrieved \
  DATASETS=popqa,musique,longmemeval,tempo N=12 <venv>/bin/python benchmarks/eval_cube2.py
"""
import sys, os, json, hashlib, statistics as st
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench")
from openai import OpenAI
from redevops_rag.store import Store
from redevops_rag.embed import Embedder
from redevops_rag.temporal import ReasonIREmbedder
from redevops_rag.retrieve import hybrid_search, diver_search
from context_runtime.planner.llm_intent import OpenAICompatModel

DATADIR = os.environ.get("DATADIR", "/tmp/claude-1000/-mnt-backup-projects-ffmpeg-mcp-aws/4b091f87-0b28-4473-a19e-4caba574b251/scratchpad/datasets")
RES = os.environ.get("OUT", "/tmp/claude-1000/-mnt-backup-projects-ffmpeg-mcp-aws/4b091f87-0b28-4473-a19e-4caba574b251/scratchpad/cube2_res")
DATASETS = os.environ.get("DATASETS", "popqa,musique,longmemeval,tempo").split(",")
METHODS = os.environ.get("METHODS", "bm25,hybrid,reasonir,diver,cr-auto").split(",")
MODELS = os.environ.get("MODELS", "qwen").split(",")
CONDS = os.environ.get("CONDS", "closed,oracle,retrieved").split(",")
N = int(os.environ.get("N", "12"))
K = int(os.environ.get("K", "8"))          # raised from 6 (d): more room for both multi-hop bridges
BUDGET = int(os.environ.get("BUDGET", "3000"))
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
os.makedirs(RES, exist_ok=True)

# ---- 7-model registry (4 GPU NVFP4 + 3 CPU GGUF). url=None => not currently served; the
# serve-and-swap driver fills the port it brings each model up on. tier drives default N. ----
MODEL_CFG = {
    "qwen":       {"url": "http://192.168.40.105:30807/v1", "model": "Qwen3.6-35B-A3B",   "extra": NOTHINK, "tier": "gpu", "rank": 2},
    "mistral":    {"url": os.environ.get("MISTRAL_URL"),    "model": "mistral-small-24b", "extra": {},       "tier": "gpu", "rank": 1},
    "gemma":      {"url": os.environ.get("GEMMA_URL"),      "model": "gemma4-26b-a4b",    "extra": {},       "tier": "gpu", "rank": 1},
    "nemotron":   {"url": os.environ.get("NEMOTRON_URL"),   "model": "nemotron3-nano-30b","extra": NOTHINK,  "tier": "gpu", "rank": 2},
    "coder":      {"url": os.environ.get("CODER_URL"),      "model": "Qwen3.6-Coder-Next","extra": NOTHINK,  "tier": "cpu", "rank": 3},
    "nemosuper":  {"url": os.environ.get("NEMOSUPER_URL"),  "model": "Nemotron-3-Super",  "extra": NOTHINK,  "tier": "cpu", "rank": 4},
    "deepseek":   {"url": "http://192.168.40.105:8001/v1",  "model": "DeepSeek-V4-Flash", "extra": {},       "tier": "cpu", "rank": 5},
}
STRONGEST = "deepseek"   # for CR-auto model escalation on model-bound queries
QWEN = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
router = OpenAICompatModel("http://192.168.40.105:30807/v1", "Qwen3.6-35B-A3B")
clients = {m: OpenAI(base_url=MODEL_CFG[m]["url"], api_key="EMPTY", timeout=900)
           for m in MODELS if MODEL_CFG[m]["url"]}

def reason_llm(system, user):
    r = QWEN.chat.completions.create(model="Qwen3.6-35B-A3B", temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def answer(model, ctx, q):
    c = MODEL_CFG[model]
    sys_p = ("Answer using ONLY the provided context, in as few words as possible. If the answer is "
             "not in the context, reply exactly: NOT FOUND.") if ctx else \
            "Answer in as few words as possible. If you do not know, reply exactly: NOT FOUND."
    user = (f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:") if ctx else f"Question: {q}\nAnswer:"
    r = clients[model].chat.completions.create(model=c["model"], temperature=0, max_tokens=96, extra_body=c["extra"],
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

_bge = Embedder(); _ri = None
def _store(item, embedder):
    dbp = f"/tmp/c2_{id(embedder)%9999}_{item['qid']}.duckdb"
    if os.path.exists(dbp): os.remove(dbp)
    s = Store(embedder, dbp)
    ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in item["docs"]]
    for c, e in zip(ch, embedder.encode([x["text"] for x in ch])): c["embedding"] = e
    s.add_chunks(ch, reindex=True); return s

def multihop_union(s, q):                                  # (d) cover BOTH bridge hops
    subs = [x.strip("-• 0123456789.") for x in reason_llm(
        "Break this question into the 2-3 atomic facts needed to answer it, one per line.", q).splitlines() if x.strip()][:3]
    seen, hits = set(), []
    for sq in [q, *subs]:
        for h in hybrid_search(s, sq, limit=4, pool=15):
            if h["document_id"] not in seen: seen.add(h["document_id"]); hits.append(h)
    return hits[:K]

def retrieve_texts(method, item):
    global _ri
    if method == "reasonir":
        if _ri is None: _ri = ReasonIREmbedder(url="http://192.168.40.105:8012/v1/embeddings")
        s = _store(item, _ri); hits = hybrid_search(s, item["question"], limit=K, pool=25); s.close()
        return [h["text"] for h in hits]
    if method == "graphiti":
        return graphiti_texts(item)
    s = _store(item, _bge)
    if method == "diver":     hits = diver_search(s, item["question"], reason_llm, limit=K, pool=25)
    elif method == "bm25":    hits = s.bm25_search(item["question"], limit=K)
    else:                     hits = multihop_union(s, item["question"]) if item["regime"] == "graph" else hybrid_search(s, item["question"], limit=K, pool=25)
    s.close()
    return [h["text"] for h in hits]

_graphiti_cache = {}
def graphiti_texts(item):
    import graphiti_core.llm_client.openai_generic_client as _ogc, datetime as dt
    _OC = _ogc.OpenAIGenericClient
    if not getattr(_OC, "_capped", False):
        class _C(_OC):
            _capped = True
            def __init__(self, *a, **k): k.setdefault("max_tokens", 1024); super().__init__(*a, **k)
        _ogc.OpenAIGenericClient = _C
    from context_runtime.adapters.store_temporal import GraphitiTemporalRetriever
    gid = "c2_" + hashlib.md5(item["qid"].encode()).hexdigest()[:12]
    g = GraphitiTemporalRetriever(neo4j_uri="bolt://192.168.40.105:7687",
                                  llm_base_url="http://192.168.40.105:30807/v1", llm_model="Qwen3.6-35B-A3B", group_id=gid)
    base = dt.datetime(2023, 1, 1)
    g.index([{"name": d["chunk_id"], "body": d["text"][:4000],
              "reference_time": (base - dt.timedelta(hours=i)).isoformat()} for i, d in enumerate(item["docs"][:20])])
    hits = g.search(item["question"], k=K)
    try: g.close()
    except Exception: pass
    return [h.text for h in hits]

def cr_route(item, model):
    """(c) bottleneck-aware: retrieval-bound regimes -> best retriever + given model;
    model-bound (temporal aggregation) -> escalate to the strongest served model."""
    rep = router.classify(item["question"]) or "document"
    if rep == "temporal":                      # aggregation/counting = MODEL-bound
        use_model = STRONGEST if (STRONGEST in clients and MODEL_CFG[STRONGEST]["rank"] > MODEL_CFG[model]["rank"]) else model
        texts = retrieve_texts("hybrid", item)
    elif rep == "graph":                        # multi-hop = RETRIEVAL-bound
        use_model, texts = model, retrieve_texts("diver", item)
    else:
        use_model, texts = model, retrieve_texts("hybrid", item)
    return use_model, texts

def load(name):
    rows = [json.loads(l) for l in open(f"{DATADIR}/{name}.jsonl")]
    rows.sort(key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())
    return [r for r in rows if str(r.get("answer", "")).strip()][:N]

print(f"models={MODELS} methods={METHODS} conds={CONDS} datasets={DATASETS} N={N} K={K}", flush=True)
for name in DATASETS:
    items = load(name)
    goldtext = {it["qid"]: [d["text"] for d in it["docs"] if d["chunk_id"] in set(it["gold"])] for it in items}
    for method in METHODS:
        # precompute retrieved context per item (reused across models & conds)
        retr = {}
        if "retrieved" in CONDS and method != "cr-auto":
            for it in items: retr[it["qid"]] = ctx_of(retrieve_texts(method, it), BUDGET)
        for model in MODELS:
            if model not in clients: continue
            res = {c: [] for c in CONDS}
            for it in items:
                golds = [it["answer"], *it.get("aliases", [])]
                for cond in CONDS:
                    if cond == "closed":   ctx = ""
                    elif cond == "oracle": ctx = ctx_of(goldtext[it["qid"]], BUDGET)
                    else:  # retrieved
                        if method == "cr-auto":
                            m, texts = cr_route(it, model); ctx = ctx_of(texts, BUDGET)
                            res[cond].append(judge(it["question"], golds, answer(m, ctx, it["question"]))); continue
                        ctx = retr[it["qid"]]
                    res[cond].append(judge(it["question"], golds, answer(model, ctx, it["question"])))
            cell = {"dataset": name, "method": method, "model": model, "n": len(items),
                    **{f"acc_{c}": round(st.mean(res[c]), 3) if res[c] else None for c in CONDS}}
            json.dump(cell, open(f"{RES}/{name}__{method}__{model}.json", "w"))
            print(f"  {name:11} {method:8} {model:9} " + " ".join(f"{c}={cell.get('acc_'+c)}" for c in CONDS), flush=True)
print("CUBE2_DONE", flush=True)
