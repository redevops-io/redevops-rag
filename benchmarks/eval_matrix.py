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

  METHODS=hybrid,diver,simgraph DATASETS=popqa,musique,longmemeval N=15 POLLUTE=15 \
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
        # Graphiti's OpenAIGenericClient hardcodes max_tokens=16384 in its __init__ (ignoring
        # config), which alone overflows Qwen's 32k context. Cap the client (extraction JSON is short).
        import graphiti_core.llm_client.openai_generic_client as _ogc
        _OC = _ogc.OpenAIGenericClient
        class _CappedClient(_OC):
            def __init__(self, *a, **k): k.setdefault("max_tokens", 2048); super().__init__(*a, **k)
        _ogc.OpenAIGenericClient = _CappedClient
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
