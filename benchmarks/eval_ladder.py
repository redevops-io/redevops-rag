"""Unified capability ladder — one stream, one metric, cumulative capabilities.

The per-version benchmarks each measure a different thing on a different dataset
(v2 precision on a seeded sim, v4 routing accuracy on MuSiQue, v5 NDCG on TEMPO...),
so their numbers can't be read against each other. This harness fixes that: it runs
ONE mixed multi-regime query stream through the SAME answer model and the SAME judge,
turning capabilities on CUMULATIVELY, and reports end-to-end answer accuracy + the
tokens of context passed. Every rung answers the same questions, graded the same way.

Stream: PopQA (lookup) + MuSiQue (multi-hop) + LongMemEval (temporal), which share a
per-question schema {question, answer, aliases, docs:[{chunk_id,text}], gold, regime}.
Each question's corpus is polluted with K distractor docs sampled from the rest of the
stream, so context management actually matters (on clean oracle docs, recall saturates).

Rungs (each adds one mechanism; answer model fixed = Qwen so only CONTEXT differs):
  v1_base   naive: large-k hybrid, dump a big context, no gating/routing/abstention
  v2_sizer  + calibrated relevance gating + load-aware sizer + grounded abstention
  v3_online + bandit picks the retrieval arm per intent bucket (stationary ~= v2;
              adaptation is a drift property — see the v3 drift drill-down)
  v4_route  + LLM knowledge routing: temporal->recency, multi-hop->sub-query fan-out,
              lookup->document
  v5_diver  + reasoning-intensive retrieval (DIVER: expansion -> union -> listwise
              rerank) on the multi-hop/reasoning path, context sized at the sweet spot

  N=10 .venv/bin/python benchmarks/eval_ladder.py     # N = items per regime
"""
import sys, os, json, hashlib, statistics as st, collections
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench/benchmarks/context-vs-model")
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench")
from openai import OpenAI
from redevops_rag.store import Store
from redevops_rag.embed import Embedder
from redevops_rag.retrieve import hybrid_search, diver_search
from context_runtime.planner.llm_intent import OpenAICompatModel

BENCH = "/mnt/backup/projects/context-runtime-go/benchdata"
REGIMES = {"popqa": "lookup", "musique": "multi-hop", "longmemeval": "temporal"}
N = int(os.environ.get("N", "10"))          # items per regime
POLLUTE = int(os.environ.get("POLLUTE", "30"))  # distractor docs added per question
SWEET = int(os.environ.get("SWEET", "4500"))    # v5 context sweet-spot (tokens), from the context-size run
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

ans_cli = OpenAI(base_url="http://192.168.40.105:30807/v1", api_key="EMPTY")
judge_cli = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
router = OpenAICompatModel("http://192.168.40.105:30807/v1", "Qwen3.6-35B-A3B")
ANS_MODEL = "Qwen3.6-35B-A3B"

def reason_llm(system, user):
    r = ans_cli.chat.completions.create(model=ANS_MODEL, temperature=0, max_tokens=120,
        extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def answer(ctx, q):
    r = ans_cli.chat.completions.create(model=ANS_MODEL, temperature=0, max_tokens=80, extra_body=NOTHINK,
        messages=[{"role": "system", "content": "Answer using ONLY the context, in as few words as possible. "
                   "If the answer is not present, reply exactly: NOT FOUND."},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"}])
    return (r.choices[0].message.content or "").strip()

def judge(question, gold_variants, cand):
    if not cand or cand.strip().upper() == "NOT FOUND":
        return False
    gold_variants = [str(g) for g in gold_variants if g is not None and str(g).strip()]
    low = cand.lower()
    if any(g.lower() in low for g in gold_variants):   # cheap alias hit first
        return True
    gold = " / ".join(dict.fromkeys(g for g in gold_variants if g))
    try:
        r = judge_cli.chat.completions.create(model="grok-4.5", temperature=0, max_tokens=8,
            messages=[{"role": "system", "content": "Strict QA grader. Reply exactly CORRECT or INCORRECT: "
                       "is the MODEL answer correct given any GOLD variant (phrasing/units may differ)?"},
                      {"role": "user", "content": f"Q: {question}\nGOLD: {gold}\nMODEL: {cand}\nVerdict:"}])
        v = (r.choices[0].message.content or "").strip().upper()
        return "CORRECT" in v and "INCORRECT" not in v
    except Exception:
        return False

def toks(s): return len(s) // 4
def score_of(h): return h.get("similarity", 0.0)  # cosine relevance (0..1)
def ctx_of(hits, budget, gate=None):
    out, used = [], 0
    for h in hits:
        if gate is not None and score_of(h) < gate:  # calibrated relevance gating
            continue
        t = toks(h["text"])
        if used + t > budget and out: break
        out.append(f"[{h.get('document_id','')}] {h['text']}"); used += t
    return "\n\n".join(out), used

# ---- load a fixed, deterministic mixed stream + a global distractor pool ----
def load(name):
    rows = [json.loads(l) for l in open(f"{BENCH}/{name}.jsonl")]
    rows.sort(key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())  # deterministic sample
    return rows[:N]

stream, pool = [], []
for name in REGIMES:
    rows = load(name)
    for r in rows:
        r["_regime"] = REGIMES[name]
        stream.append(r)
    for r in rows:
        pool += [d["text"] for d in r["docs"]]
print(f"stream={len(stream)} ({', '.join(REGIMES.values())}) | pollute={POLLUTE} | answerer={ANS_MODEL}", flush=True)

emb = Embedder()
def build_store(item):
    dbp = f"/tmp/ladder_{item['qid']}.duckdb"
    if os.path.exists(dbp): os.remove(dbp)
    own = [d["text"] for d in item["docs"]]
    ownset = set(own)
    distract = [t for t in sorted(pool, key=lambda t: hashlib.md5((item["qid"]+t).encode()).hexdigest())
                if t not in ownset][:POLLUTE]
    docs = item["docs"] + [{"chunk_id": f"noise{i}", "text": t} for i, t in enumerate(distract)]
    store = Store(emb, dbp)
    ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in docs]
    for c, e in zip(ch, emb.encode([c["text"] for c in ch])): c["embedding"] = e
    store.add_chunks(ch, reindex=True)
    return store

def route(q):
    rep = router.classify(q) or "document"
    return rep if rep in ("graph", "temporal") else "document"

# ---- the five cumulative rungs: (name, retrieve->hits, budget) ----
def retr_v1(store, q, rep):   return hybrid_search(store, q, limit=20, pool=40, recency_half_life_days=0)
def retr_v2(store, q, rep):   return hybrid_search(store, q, limit=8, pool=25, recency_half_life_days=0)
def retr_v4(store, q, rep):
    if rep == "temporal": return hybrid_search(store, q, limit=8, pool=25, recency_half_life_days=1.0)
    if rep == "graph":                                   # sub-query fan-out (union), no listwise rerank
        subs = [s.strip("-• ") for s in reason_llm(
            "Decompose the question into 2-3 atomic sub-questions, one per line, no numbering.", q).splitlines() if s.strip()][:3] or [q]
        seen, hits = set(), []
        for sq in [q, *subs]:
            for h in hybrid_search(store, sq, limit=6, pool=20):
                if h["document_id"] not in seen:
                    seen.add(h["document_id"]); hits.append(h)
        return hits[:10]
    return hybrid_search(store, q, limit=8, pool=25)
def retr_v5(store, q, rep):
    if rep == "temporal": return hybrid_search(store, q, limit=8, pool=25, recency_half_life_days=1.0)
    if rep == "graph":    return diver_search(store, q, reason_llm, limit=10, pool=25, n_subqueries=3)
    return hybrid_search(store, q, limit=8, pool=25)

RUNGS = [
    ("v1_base",   "naive dump",                 retr_v1, 8000, None,  False, False),
    ("v2_sizer",  "+ gating/sizer/abstain",     retr_v2, 2500, 0.30,  True,  False),
    ("v3_online", "+ online arm bandit",        retr_v2, 2500, 0.30,  True,  True),
    ("v4_route",  "+ knowledge routing",        retr_v4, 3000, 0.25,  True,  True),
    ("v5_diver",  "+ DIVER reasoning retrieval", retr_v5, SWEET, 0.25, True,  True),
]

# v3 bandit: pick the sizer budget arm per bucket by observed reward (epsilon-greedy, discounted).
arms = {"lean": 1800, "mid": 2500, "rich": 4000}
qvals = collections.defaultdict(lambda: {a: (0.0, 0) for a in arms})  # bucket -> arm -> (avg, n)
def bandit_budget(bucket, i):
    order = sorted(arms, key=lambda a: -qvals[bucket][a][0])
    return arms[order[0] if (i % 5) else order[-1]]  # mostly exploit best, periodically explore

def record(bucket, arm_name, reward):
    avg, n = qvals[bucket][arm_name]
    qvals[bucket][arm_name] = (avg + 0.5 * (reward - avg), n + 1)  # discounted (recency-weighted)

results = {r[0]: {"correct": [], "toks": [], "by": collections.defaultdict(list)} for r in RUNGS}
for i, it in enumerate(stream):
    store = build_store(it)
    q, regime = it["question"], it["_regime"]
    golds = [it["answer"], *it.get("aliases", [])]
    rep = route(q)                                  # one router call, reused by v4/v5
    for name, _desc, retr, budget, gate, abstain, use_route in RUNGS:
        r = rep if use_route else "document"
        hits = retr(store, q, r)
        if name == "v3_online":
            bucket = rep
            budget = bandit_budget(bucket, i)
        ctx, used = ctx_of(hits, budget, gate=gate)
        best = max((score_of(h) for h in hits), default=0.0)
        if abstain and (not ctx or best < (gate or 0)):
            cand, used = "NOT FOUND", 0
        else:
            cand = answer(ctx, q)
        ok = judge(q, golds, cand)
        if name == "v3_online":
            record(rep, next(a for a, v in arms.items() if v == budget), 1.0 if ok else 0.0)
        results[name]["correct"].append(ok); results[name]["toks"].append(used)
        results[name]["by"][regime].append(ok)
    store.close()
    if (i + 1) % 5 == 0: print(f"  {i+1}/{len(stream)}", flush=True)

print(f"\n=== capability ladder — mixed stream ({len(stream)} q: {N}/regime, {ANS_MODEL}, grok-judged) ===")
print(f"{'config':12}{'mechanism':30}{'acc':>7}{'tok/q':>8}   per-regime acc")
for name, desc, *_ in RUNGS:
    R = results[name]
    acc = st.mean(R["correct"]); tk = int(st.mean(R["toks"]))
    per = "  ".join(f"{rg[:4]}={st.mean(R['by'][rg]):.2f}" for rg in ["lookup", "multi-hop", "temporal"])
    print(f"{name:12}{desc:30}{acc:>7.3f}{tk:>8}   {per}")
print("DONE")
