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
# ---- Phase-0 generation-strategy ablation (the answer-plane axis) --------------------------------
# STRATEGIES sweeps the generation strategy per cell, isolating GENERATION from retrieval (run with
# CONDS=oracle to hold retrieval at gold). `direct` reproduces the prior terse/no-think baseline
# EXACTLY; reason/decompose/mapreduce turn thinking on, widen the budget, and recalibrate abstention.
#   direct     — terse extractive, no-think, 96 tok            (lookup)
#   reason     — think + CoT + short final answer              (synthesis / single-hop reasoning)
#   decompose  — list intermediate facts → answer → compose    (multi_hop)
#   mapreduce  — extract (date,entity,value) per source → agg  (temporal aggregation / counting)
# The resulting acc per (dataset, strategy, model) at oracle is the warm-start prior for the CR
# generation bandit (Phase 1).  e.g.:
#   CONDS=oracle STRATEGIES=direct,reason,decompose,mapreduce DATASETS=musique,longmemeval \
#     MODELS=qwen METHODS=hybrid N=12 <venv>/bin/python benchmarks/eval_cube2.py
STRATEGIES = os.environ.get("STRATEGIES", "direct").split(",")
GEN_BUDGET = int(os.environ.get("GEN_BUDGET", "768"))       # token budget for reasoning strategies
# Recalibrated abstention (Step 4): don't bail when the pieces are present — the cure for over-abstention.
_ABSTAIN = ("If the pieces needed to answer are present in the context, reason across them and answer; "
            "reply exactly NOT FOUND only if the context truly lacks the answer.")

# ---- (A) y-axis: best EMBEDDER per dataset (opt-in EMBED_ROUTING=1) --------------------------------
# "model coupled with best embedder for the dataset": route the encoder on the corpus (redevops-rag
# encoder_for) — nutrition is Russian/domain → Nemotron-Embed; the English sets → cheap bge. A Nemotron
# store auto-applies the asymmetric query instruction (semantic_search query_mode + DIVER instruct/plain),
# so no call-site change is needed here. Default OFF → hardcoded bge, byte-identical to prior cube2 runs.
# Needs the Nemotron endpoint served (REDEVOPS_RAG_NEMOTRON_URL). The 'reasonir' method keeps its own
# ReasonIR embedder regardless (a fixed English arm, a legitimate cell to measure).
EMBED_ROUTING = os.environ.get("EMBED_ROUTING", "").lower() in ("1", "true", "yes", "on")
DATASET_CORPUS = {"nutrition": ("ru", "nutrition")}   # dataset -> (lang, domain); default ('en','')

# ---- (B) z-axis: couple CR-auto retrieval mode with proper INFERENCE (opt-in CR_COUPLE=1) ----------
# cr-auto already picks retriever+model per regime; coupling also picks the reasoning strategy per
# regime (temporal→mapreduce aggregation, multi-hop→decompose, lookup/document→direct). When on, the
# cr-auto cell ignores the swept STRATEGIES and labels its strategy 'auto'. Default OFF → prior cr-auto.
CR_COUPLE = os.environ.get("CR_COUPLE", "").lower() in ("1", "true", "yes", "on")
REGIME_STRATEGY = {"temporal": "mapreduce", "graph": "decompose", "document": "direct", "low_graph": "direct"}
os.makedirs(RES, exist_ok=True)


def _extra(model, think):
    """Thinking-capable models were registered with the NOTHINK sentinel; flip enable_thinking per
    strategy. Models without a thinking switch pass their configured extra through unchanged."""
    if MODEL_CFG[model].get("extra") == NOTHINK:
        return {"chat_template_kwargs": {"enable_thinking": bool(think)}}
    return MODEL_CFG[model].get("extra") or {}


def _final(out):
    """Pull the final answer from a reasoning response: strip <think> blocks, then take the 'Answer:'
    line if present (the reasoning strategies end with one), else the last non-empty line."""
    import re
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    return out.splitlines()[-1].strip() if out.strip() else out

# ---- 7-model registry (4 GPU NVFP4 + 3 CPU GGUF). url=None => not currently served; the
# serve-and-swap driver fills the port it brings each model up on. tier drives default N. ----
MODEL_CFG = {
    "qwen":       {"url": "http://192.168.40.105:30807/v1", "model": "Qwen3.6-35B-A3B",   "extra": NOTHINK, "tier": "gpu", "rank": 2},
    "qwen35":     {"url": os.environ.get("QWEN35_URL"),     "model": "Qwen3.5-122B",      "extra": {},       "tier": "cpu", "rank": 3},
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

def _gen(model, sys_p, user, *, think, budget):
    c = MODEL_CFG[model]
    r = clients[model].chat.completions.create(model=c["model"], temperature=0, max_tokens=budget,
        extra_body=_extra(model, think), messages=[{"role": "system", "content": sys_p},
                                                    {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def answer(model, ctx, q, strategy="direct"):
    # closed-book (no context): unchanged, terse.
    if not ctx:
        return _gen(model, "Answer in as few words as possible. If you do not know, reply exactly: "
                    "NOT FOUND.", f"Question: {q}\nAnswer:", think=False, budget=96)
    # `direct` reproduces the prior terse/no-think baseline EXACTLY (verbatim prompt + 96 tok).
    if strategy in ("direct", "terse"):
        return _gen(model, "Answer using ONLY the provided context, in as few words as possible. If the "
                    "answer is not in the context, reply exactly: NOT FOUND.",
                    f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:", think=False, budget=96)
    # reasoning strategies: thinking on, wider budget, recalibrated abstention, 'Answer:' final line.
    if strategy == "reason":
        sys_p = ("Answer the question using ONLY the provided context. Think step by step, then give a "
                 "short final answer on a line beginning 'Answer:'. " + _ABSTAIN)
    elif strategy == "decompose":
        sys_p = ("Answer the multi-hop question using ONLY the provided context. First list the "
                 "intermediate facts needed and answer each from the context, then compose the final "
                 "answer on a line beginning 'Answer:'. " + _ABSTAIN)
    elif strategy == "mapreduce":
        sys_p = ("You aggregate across sources. From the context, extract every relevant fact as a "
                 "bullet '- (when, who/what, value)', then compute the answer over those facts and give "
                 "it on a line beginning 'Answer:'. " + _ABSTAIN)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return _final(_gen(model, sys_p, f"Context:\n{ctx}\n\nQuestion: {q}", think=True, budget=GEN_BUDGET))

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
_emb_cache = {"bge": _bge}
ACTIVE_EMB = _bge   # the dataset's routed default embedder (set per dataset in the main loop)

def dataset_embedder(name):
    """(A) The best embedder for a dataset's corpus. EMBED_ROUTING off → always bge (prior behavior)."""
    if not EMBED_ROUTING:
        return _bge
    from redevops_rag.embed import encoder_for, make_embedder
    lang, domain = DATASET_CORPUS.get(name, ("en", ""))
    backend = encoder_for(lang, domain)
    if backend == "colpali":     # doc-visual arm is out of scope for this text cube — fall back to bge
        backend = "bge"
    if backend not in _emb_cache:
        _emb_cache[backend] = make_embedder(backend)
    return _emb_cache[backend]

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
    s = _store(item, ACTIVE_EMB)
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
    model-bound (temporal aggregation) -> escalate to the strongest served model. Also returns the
    regime-coupled reasoning strategy (B); the caller applies it only when CR_COUPLE is on."""
    rep = router.classify(item["question"]) or "document"
    if rep == "temporal":                      # temporal-reasoning = RETRIEVAL- AND MODEL-sensitive
        # DIVER is now in the temporal routed set: on TEMPO it scores 0.91 vs hybrid 0.84 (and vs the
        # graph engines' 0.24-0.28), so temporal retrieves with DIVER, not plain hybrid — the fix for
        # CR-auto's TEMPO gap. Still escalate the model (temporal aggregation is also model-bound).
        use_model = STRONGEST if (STRONGEST in clients and MODEL_CFG[STRONGEST]["rank"] > MODEL_CFG[model]["rank"]) else model
        texts = retrieve_texts("diver", item)
    elif rep == "graph":                        # multi-hop = RETRIEVAL-bound
        use_model, texts = model, retrieve_texts("diver", item)
    else:
        use_model, texts = model, retrieve_texts("hybrid", item)
    return use_model, texts, REGIME_STRATEGY.get(rep, "direct")

def load(name):
    rows = [json.loads(l) for l in open(f"{DATADIR}/{name}.jsonl")]
    rows.sort(key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())
    return [r for r in rows if str(r.get("answer", "")).strip()][:N]

print(f"models={MODELS} methods={METHODS} conds={CONDS} datasets={DATASETS} N={N} K={K}", flush=True)
for name in DATASETS:
    items = load(name)
    ACTIVE_EMB = dataset_embedder(name)   # (A) route the dataset's default embedder (bge unless EMBED_ROUTING)
    if EMBED_ROUTING:
        print(f"  [embed-routing] {name} -> {getattr(ACTIVE_EMB, 'backend', 'bge')} (dim={ACTIVE_EMB.dim})", flush=True)
    goldtext = {it["qid"]: [d["text"] for d in it["docs"] if d["chunk_id"] in set(it["gold"])] for it in items}
    for method in METHODS:
        # precompute retrieved context per item (reused across models & conds)
        retr = {}
        if "retrieved" in CONDS and method != "cr-auto":
            for it in items: retr[it["qid"]] = ctx_of(retrieve_texts(method, it), BUDGET)
        for model in MODELS:
            if model not in clients: continue
            for strat in STRATEGIES:            # generation-strategy axis (Phase-0 answer-plane ablation)
                res = {c: [] for c in CONDS}
                for it in items:
                    golds = [it["answer"], *it.get("aliases", [])]
                    for cond in CONDS:
                        if cond == "closed":   ctx = ""
                        elif cond == "oracle": ctx = ctx_of(goldtext[it["qid"]], BUDGET)
                        else:  # retrieved
                            if method == "cr-auto":
                                m, texts, cr_strat = cr_route(it, model); ctx = ctx_of(texts, BUDGET)
                                use_strat = cr_strat if CR_COUPLE else strat   # (B) couple inference to regime
                                res[cond].append(judge(it["question"], golds, answer(m, ctx, it["question"], use_strat))); continue
                            ctx = retr[it["qid"]]
                        res[cond].append(judge(it["question"], golds, answer(model, ctx, it["question"], strat)))
                cell = {"dataset": name, "method": method, "model": model, "strategy": strat, "n": len(items),
                        **{f"acc_{c}": round(st.mean(res[c]), 3) if res[c] else None for c in CONDS}}
                json.dump(cell, open(f"{RES}/{name}__{method}__{model}__{strat}.json", "w"))
                print(f"  {name:11} {method:8} {model:9} {strat:9} " + " ".join(f"{c}={cell.get('acc_'+c)}" for c in CONDS), flush=True)
print("CUBE2_DONE", flush=True)
