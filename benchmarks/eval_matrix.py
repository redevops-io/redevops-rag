"""Method x regime retrieval matrix — each retrieval method evaluated on EACH dataset.

The point the earlier "ladder" got wrong: a single method applied everywhere makes it look
like only lookups work. The right question is *which method wins which regime* — so this
scores every method on all three regimes and lets the winner-per-column speak. Metric is
retrieval quality against gold chunk_ids (recall@k + NDCG@k), which is answer-model-independent
(no small-model answer bottleneck).

Datasets (shared per-question schema {question, answer, docs:[{chunk_id,text,created_at?}],
gold, regime}): PopQA (lookup) · MuSiQue (multi-hop) · LongMemEval (temporal). Each question's
corpus is its own docs + POLLUTE distractors from the rest of the stream, so retrieval is
non-trivial (on the raw oracle docs recall saturates).

Methods (select via METHODS env; each runs in the venv that has its deps):
  hybrid    redevops-rag hybrid dense(bge)+BM25 -> RRF          (baseline)
  reasonir  same hybrid but dense vectors from ReasonIR-8B      (reasoning embedder)
  diver     LLM query-expansion -> union -> LLM listwise rerank (reasoning retrieval)
  simgraph  dependency-free 2-hop term-spreading graph          (cheap graph)
  hipporag  real HippoRAG LLM-OpenIE entity graph + PPR         (heavy graph)
  graphiti  real Graphiti bi-temporal KG over Neo4j             (heavy temporal)
  cr-auto   competence-routed: per item picks the cube-best retriever, measures ITS recall

  METHODS=hybrid,diver,simgraph DATASETS=popqa,musique,longmemeval N=15 POLLUTE=15 \
    <venv>/bin/python benchmarks/eval_matrix.py

cr-auto (closes I6 — the routed-recall row the retrieval page is missing): routes each question to the
competence-best retriever from CR_ROUTES (build_routes.py's routes.json — the SAME routing eval_cube2's
cr_route uses), then reports that retriever's recall@k. by_dataset[name] takes precedence (deterministic
per regime); else by_rep[classify(question)] routes per question via the Qwen intent classifier. This is
the recall-side proof that routing selects the right retriever per regime — reasonir on nutrition,
diver on temporal — rather than one method applied everywhere.
  METHODS=cr-auto CR_ROUTES=/path/routes.json DATASETS=nutrition,musique,longmemeval,tempo \
    <venv>/bin/python benchmarks/eval_matrix.py
"""
import sys, os, json, math, hashlib, datetime as dt, statistics as st
sys.path.insert(0, "/mnt/backup/projects/context-runtime-bench")

BENCH = os.environ.get("DATADIR", "/mnt/backup/projects/context-runtime-go/benchdata")
REGIME = {"popqa": "lookup", "musique": "multi-hop", "longmemeval": "temporal",
          "tempo": "temporal-reasoning", "nutrition": "domain-lookup"}
DATASETS = os.environ.get("DATASETS", "popqa,musique,longmemeval").split(",")
METHODS = os.environ.get("METHODS", "hybrid,diver,simgraph").split(",")
N = int(os.environ.get("N", "15"))
POLLUTE = int(os.environ.get("POLLUTE", "15"))
K = int(os.environ.get("K", "10"))
OUT = os.environ.get("OUT", "/mnt/backup/projects/redevops-rag/benchmarks/results/matrix")
QWEN = "http://192.168.40.105:30807/v1"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

def load(name):
    rows = [json.loads(l) for l in open(f"{BENCH}/{name}.jsonl")]
    rows.sort(key=lambda r: hashlib.md5(r["qid"].encode()).hexdigest())
    return rows[:N]

def corpus_for(item, pool):
    own = {d["chunk_id"]: d for d in item["docs"]}
    owntext = {d["text"] for d in item["docs"]}
    distract = [d for d in sorted(pool, key=lambda d: hashlib.md5((item["qid"]+d["chunk_id"]).encode()).hexdigest())
                if d["text"] not in owntext][:POLLUTE]
    docs = list(item["docs"]) + [{"chunk_id": f"noise_{i}_{d['chunk_id']}", "text": d["text"],
                                  "created_at": d.get("created_at")} for i, d in enumerate(distract)]
    return docs

def recall_at_k(ranked_ids, gold, k):
    return len(set(ranked_ids[:k]) & set(gold)) / len(gold) if gold else 0.0
def ndcg_at_k(ranked_ids, gold, k):
    g = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(ranked_ids[:k]) if x in g)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), k)))
    return dcg / idcg if idcg else 0.0

# ---------- engine factories (built once, reused across questions) ----------
def make_engine(method):
    if method == "bm25":
        # lexical baseline — the page's "vs BM25" reference point, on the current (rebuilt) datasets.
        from redevops_rag.store import Store
        from redevops_rag.embed import Embedder
        emb = Embedder()   # embeddings unused by bm25_search, but Store needs an encoder for its schema
        def run(item, docs):
            dbp = f"/tmp/mx_bm25_{item['qid']}.duckdb"
            if os.path.exists(dbp): os.remove(dbp)
            s = Store(emb, dbp)
            ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in docs]
            for c in ch: c["embedding"] = [0.0] * emb.dim   # placeholder; bm25 leg ignores vectors
            s.add_chunks(ch, reindex=True)
            hits = s.bm25_search(item["question"], limit=K)
            s.close()
            return [h["document_id"] for h in hits]
        return run
    if method in ("hybrid", "reasonir", "diver"):
        from redevops_rag.store import Store
        from redevops_rag.embed import Embedder
        from redevops_rag.retrieve import hybrid_search, diver_search
        from openai import OpenAI
        emb = Embedder() if method != "reasonir" else None
        if method == "reasonir":
            from redevops_rag.temporal import ReasonIREmbedder
            emb = ReasonIREmbedder(url="http://192.168.40.105:8012/v1/embeddings")
        cli = OpenAI(base_url=QWEN, api_key="EMPTY")
        def reason_llm(system, user):
            r = cli.chat.completions.create(model="Qwen3.6-35B-A3B", temperature=0, max_tokens=120,
                extra_body=NOTHINK, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return (r.choices[0].message.content or "").strip()
        def run(item, docs):
            dbp = f"/tmp/mx_{method}_{item['qid']}.duckdb"
            if os.path.exists(dbp): os.remove(dbp)
            s = Store(emb, dbp)
            ch = [{"document_id": d["chunk_id"], "text": d["text"], "metadata": {}} for d in docs]
            for c, e in zip(ch, emb.encode([c["text"] for c in ch])): c["embedding"] = e
            s.add_chunks(ch, reindex=True)
            if method == "diver":
                hits = diver_search(s, item["question"], reason_llm, limit=K, pool=max(25, K*2))
            else:
                hits = hybrid_search(s, item["question"], limit=K, pool=max(25, K*2), recency_half_life_days=0)
            s.close()
            return [h["document_id"] for h in hits]
        return run
    if method == "simgraph":
        from context_runtime.adapters.store_hipporag import SimGraphRetriever
        def run(item, docs):
            r = SimGraphRetriever([{"chunk_id": d["chunk_id"], "filename": d["chunk_id"], "text": d["text"]} for d in docs])
            return [h.chunk_id for h in r.search(item["question"], k=K, method="graph")]
        return run
    if method == "hipporag":
        from context_runtime.adapters.store_hipporag import HippoRAGRetriever
        def run(item, docs):
            text2id = {d["text"]: d["chunk_id"] for d in docs}
            hr = HippoRAGRetriever(save_dir=f"/tmp/hr_mx/{item['qid']}", llm_model_name="Qwen3.6-35B-A3B",
                                   llm_base_url=QWEN, llm_api_key="EMPTY",
                                   embedding_model_name="facebook/contriever")
            hr.index([d["text"] for d in docs])
            return [text2id.get(h.text, "?") for h in hr.search(item["question"], k=K)]
        return run
    if method == "graphiti":
        # Graphiti's OpenAIGenericClient hardcodes max_tokens=16384 (ignoring config) which alone
        # overflows Qwen's 32k window; and on dense multi-session corpora the extraction context
        # (episode + previous_episodes + node list) can spike far past 32k — a ~163k-token prompt on
        # LongMemEval — 400-ing the entire ingest. FORK the client to (a) cap output and (b) enforce a
        # hard INPUT-char budget: truncate variable-size USER content to fit the window before the API
        # call, never the SYSTEM instructions. Source-agnostic — whatever field balloons, the prompt
        # still fits, so Graphiti ingests arbitrarily large/dense corpora and degrades gracefully
        # (a truncated extraction) instead of crashing. GRAPHITI_MAX_INPUT_CHARS tunes the budget.
        import graphiti_core.llm_client.openai_generic_client as _ogc
        _OC = _ogc.OpenAIGenericClient
        # Both sides of the window must be bounded TOGETHER: input + output < WINDOW. Two failure modes
        # to avoid — (a) a huge extraction context 400-overflows the input (LongMemEval's dense sessions),
        # (b) too small an output budget TRUNCATES the edge-extraction JSON mid-object → JSONDecodeError
        # (many entities/edges need >2k output tokens). Fix: bound input to leave room for a generous
        # output, then size output DYNAMICALLY to whatever room remains — big enough for the JSON, never
        # overflowing. tempo succeeded at n=15 already; this is for the edge-heavy long-memory sessions.
        _WINDOW = int(os.environ.get("GRAPHITI_CTX_WINDOW", "32768"))
        _OUT_CAP = int(os.environ.get("GRAPHITI_MAX_OUTPUT_TOKENS", "6144"))   # headroom for edge JSON
        _MARGIN = 512
        _MAX_IN = (_WINDOW - _OUT_CAP - _MARGIN) * 4    # input char budget that preserves output room
        class _BoundedClient(_OC):
            def __init__(self, *a, **k): k.setdefault("max_tokens", _OUT_CAP); super().__init__(*a, **k)
            async def _generate_response(self, messages, response_model=None, max_tokens=None, *a, **k):
                if sum(len(m.content or "") for m in messages) > _MAX_IN:   # bound USER content, keep SYSTEM
                    sys_c = sum(len(m.content or "") for m in messages if m.role == "system")
                    budget, used = max(4000, _MAX_IN - sys_c), 0
                    for m in messages:
                        if m.role == "system":
                            continue
                        c = m.content or ""
                        if used >= budget:
                            m.content = ""
                        elif used + len(c) > budget:
                            m.content = c[: budget - used] + "\n…[truncated to fit context window]"
                            used = budget
                        else:
                            used += len(c)
                in_tok = sum(len(m.content or "") for m in messages) // 4   # size output to the room left
                out = max(1024, min(_OUT_CAP, _WINDOW - in_tok - _MARGIN))
                return await super()._generate_response(messages, response_model, out, *a, **k)
        _ogc.OpenAIGenericClient = _BoundedClient
        from context_runtime.adapters.store_temporal import GraphitiTemporalRetriever
        def parse_dt(s):
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try: return dt.datetime.strptime(str(s), fmt)
                except Exception: pass
            return dt.datetime(2023, 1, 1)
        def run(item, docs):
            gid = "mx_" + hashlib.md5(item["qid"].encode()).hexdigest()[:12]
            g = GraphitiTemporalRetriever(neo4j_uri="bolt://192.168.40.105:7687",
                                          llm_base_url=QWEN, llm_model="Qwen3.6-35B-A3B", group_id=gid)
            base = parse_dt(item.get("question_date", "2023-01-01"))
            CAP = int(os.environ.get("GRAPHITI_CAP", "6000"))  # cap episode body: Graphiti's LLM
            eps = [{"name": d["chunk_id"], "body": d["text"][:CAP],  # extraction must fit the 32k ctx
                    "reference_time": (base - dt.timedelta(hours=i)).isoformat()} for i, d in enumerate(docs)]
            g.index(eps)
            hits = g.search(item["question"], k=K)
            ids = []
            for h in hits:
                ids += h.meta.get("source_sessions") or [h.chunk_id]
            try: g.close()
            except Exception: pass
            return ids
        return run
    if method == "cr-auto":
        # Competence-routed recall: per item, resolve the cube-best retriever (same precedence as
        # eval_cube2.cr_route — by_dataset ceiling, else by_rep runtime route) and return ITS ranking.
        # Sub-engines are built lazily so we only pay for the retrievers the routes actually name.
        routes = {}
        rp = os.environ.get("CR_ROUTES")
        if rp:
            try: routes = json.load(open(rp))
            except Exception: routes = {}   # a bad routes file falls back to CR_DEFAULT_METHOD, never crashes
        by_dataset, by_rep = routes.get("by_dataset", {}), routes.get("by_rep", {})
        default_method = os.environ.get("CR_DEFAULT_METHOD", "hybrid")
        _engines, _router = {}, {"m": None}
        def _classify(q):
            if _router["m"] is None:
                from context_runtime.planner.llm_intent import OpenAICompatModel
                _router["m"] = OpenAICompatModel(QWEN, "Qwen3.6-35B-A3B")
            return _router["m"].classify(q) or "document"
        def _engine_for(m):
            if m == "cr-auto": m = default_method   # guard against a self-referential route entry
            if m not in _engines:
                _engines[m] = make_engine(m)
            return _engines[m]
        def _route_method(item):
            r = by_dataset.get(item.get("_dataset"))
            if not r and by_rep:
                r = by_rep.get(_classify(item["question"]))
            return (r or {}).get("method", default_method)
        def run(item, docs):
            return _engine_for(_route_method(item))(item, docs)
        return run
    raise ValueError(method)

# ---------- run ----------
os.makedirs(OUT, exist_ok=True)
print(f"methods={METHODS} datasets={DATASETS} N={N} POLLUTE={POLLUTE} K={K}", flush=True)
data = {name: load(name) for name in DATASETS}
pools = {name: [d for it in rows for d in it["docs"]] for name, rows in data.items()}

for method in METHODS:
    run = make_engine(method)
    for name in DATASETS:
        rec, nd, n_ok = [], [], 0
        for it in data[name]:
            it["_dataset"] = name   # cr-auto reads this to resolve the by_dataset route
            try:
                ids = run(it, corpus_for(it, pools[name]))
                rec.append(recall_at_k(ids, it["gold"], K)); nd.append(ndcg_at_k(ids, it["gold"], K)); n_ok += 1
            except Exception as e:
                import traceback
                print(f"  [{method}/{name}/{it['qid']}] FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
                if os.environ.get("TRACE"): traceback.print_exc()
        r = {"method": method, "dataset": name, "regime": REGIME[name], "n": n_ok,
             "recall": round(st.mean(rec), 4) if rec else None, "ndcg": round(st.mean(nd), 4) if nd else None}
        json.dump(r, open(f"{OUT}/{method}__{name}.json", "w"))
        print(f"  {method:10} {name:12} recall@{K}={r['recall']}  ndcg@{K}={r['ndcg']}  (n={n_ok})", flush=True)
print("MATRIX_METHODS_DONE", flush=True)
