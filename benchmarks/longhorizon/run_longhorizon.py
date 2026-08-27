"""Long-horizon context-management runner: arms × horizons over a Wikimedia haystack with synthetic
needles. Writes one result cell per (corpus, arm, horizon, model) and prints the quality×cost×latency
frontier. See DESIGN.md.

  # plumbing smoke — no LLM/GPU (real corpus+needles+retrieval, oracle answerer):
  python benchmarks/longhorizon/run_longhorizon.py --dry --horizons 10000,30000 --needles 4

  # real run, local fixed-window answerer (Qwen 32k window, thinking off):
  MODEL=qwen python benchmarks/longhorizon/run_longhorizon.py --horizons 10000,100000,1000000

The answerer is a plain fixed-window consumer (MODEL=qwen local, or MODEL=api → Kimi/grok). It must NOT be
a model that manages context itself (e.g. DeepSeek-V4-Flash) or the frontier is no longer attributable to
the arm. WINDOW (env, default 32000) is the answerer's context window; the `full` arm truncates to it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO / "src"))          # redevops_rag
sys.path.insert(0, str(_REPO / "benchmarks"))   # package-relative imports below

from longhorizon.corpus import load_articles            # noqa: E402
from longhorizon.tasks import build_item                # noqa: E402
from longhorizon import arms                            # noqa: E402
from longhorizon import provenance                      # noqa: E402

RES = Path(os.environ.get("OUT", _HERE.parent / "results" / "longhorizon"))
WINDOW = int(os.environ.get("WINDOW", "32000"))         # answerer's fixed context window (tokens)
RETRIEVE_LIMIT = int(os.environ.get("RETRIEVE_LIMIT", "8"))   # cr arm top-k (a knob CR itself optimizes)
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def make_client(model: str):
    """Fixed-window answerer client. qwen = local vLLM; api = Kimi/grok via ANSWERER_*/KIMI_*."""
    from openai import OpenAI
    if model == "api":
        url = os.environ.get("ANSWERER_URL") or os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
        name = os.environ.get("ANSWERER_MODEL") or os.environ.get("KIMI_MODEL", "kimi-k2.6")
        key = os.environ.get("ANSWERER_KEY") or os.environ.get("KIMI_API_KEY", "")
        return OpenAI(base_url=url, api_key=key or "EMPTY", timeout=900), name, {}
    url = os.environ.get("QWEN_URL", "http://192.168.40.105:30807/v1")
    return OpenAI(base_url=url, api_key="EMPTY", timeout=900), os.environ.get("QWEN_MODEL", "Qwen3.6-35B-A3B"), NOTHINK


def build_store(emb, item, embeddings):
    from redevops_rag.store import Store
    ch = [dict(c) for c in item.chunks]
    for c, e in zip(ch, embeddings):
        c["embedding"] = e
    s = Store(emb, ":memory:")
    s.add_chunks(ch, reindex=True)
    return s


def build_enterprise_retriever(emb, store, item, embeddings, limit):
    """The F4 `cr-enterprise` arm, if the enterprise overlay is importable. Builds a region index over
    this item's chunks and wraps hybrid_search as the region-scoped backend. Returns None if CR-enterprise
    is not installed — the arm is simply skipped (the harness runs on the AGPL stack alone)."""
    try:
        from context_runtime_enterprise.sparse_regions import RegionIndex, SparseRegionRetriever
    except Exception:
        return None
    from redevops_rag.retrieve import hybrid_search
    index = RegionIndex.build(item.chunks, embeddings)

    def scoped_search(query, k, method, doc_ids):
        return hybrid_search(store, query, limit=k, pool=max(50, k * 4), document_ids=doc_ids)

    return SparseRegionRetriever(index, scoped_search, lambda q: emb.encode([q])[0],
                                 top_regions=int(os.environ.get("TOP_REGIONS", "4")),
                                 floor=float(os.environ.get("REGION_FLOOR", "0.15")))


STATE_TOKENS = int(os.environ.get("STATE_TOKENS", "500"))   # the STATE_ONLY depth's tiny state window


def f2_available():
    try:
        import context_runtime_enterprise.materialization  # noqa: F401
        return True
    except Exception:
        return False


def f2_materialize(nd, item, store, ent, limit):
    """F2 adaptive materialization: escalate STATE_ONLY → STATE_SPARSE(F4) → STATE_DEEP → FULL_CONTEXT and
    stop at the cheapest depth whose retrieval surfaces the query's subject — the facility KEY the question
    names. That is a confidence signal (did we retrieve the relevant evidence), NOT the gold value, so the
    ladder does not peek at the answer. Lazy: only depths up to the chosen one are actually retrieved, so
    the cost is the cheapest sufficient depth's — the whole point of F2."""
    from context_runtime_enterprise.materialization import Depth, MaterializationLadder
    key = nd.key.lower()
    built: dict = {}

    def probe(depth, make):
        built[depth] = make()
        return key in built[depth].lower()

    probes = {
        Depth.STATE_ONLY: lambda: probe(Depth.STATE_ONLY, lambda: arms.ctx_full(item, STATE_TOKENS)),
        Depth.STATE_SPARSE: lambda: probe(Depth.STATE_SPARSE,
                                          lambda: arms.ctx_cr_enterprise(ent, nd.question, limit=limit)),
        Depth.STATE_DEEP: lambda: probe(Depth.STATE_DEEP,
                                        lambda: arms.ctx_cr(store, item, nd.question, limit=limit)),
    }
    choice = MaterializationLadder().select("hybrid:local", probes)
    ctx = built.get(choice.depth)
    if ctx is None:                                  # FULL_CONTEXT (ceiling): materialize the whole window
        ctx = arms.ctx_full(item, WINDOW)
    return ctx, choice.depth.label


def run(horizons, n_needles, model, dry, corpus, dump, seed, limit):
    RES.mkdir(parents=True, exist_ok=True)
    from redevops_rag.embed import Embedder
    emb = Embedder()
    articles = load_articles(dump, corpus=corpus, limit=limit)
    print(f"[longhorizon] corpus={corpus} articles={len(articles)} window={WINDOW} "
          f"model={'oracle(dry)' if dry else model}", flush=True)

    client = name = extra = None
    if not dry:
        client, name, extra = make_client(model)

    print(f"{'horizon':>9} {'arm':<14} {'acc':>5} {'in_tok':>8} {'lat_s':>7}", flush=True)
    prov = None
    for H in horizons:
        item = build_item(articles, H, n_needles=n_needles, item_id=f"{corpus}-{H}", seed=seed)
        embeddings = emb.encode([c["text"] for c in item.chunks])
        store = build_store(emb, item, embeddings)
        ent = build_enterprise_retriever(emb, store, item, embeddings, RETRIEVE_LIMIT)
        arm_list = ["full", "cr"]
        if ent is not None:
            arm_list.append("cr-enterprise")
            if f2_available():
                arm_list.append("cr-materialize")
        if prov is None:   # capture once — same code runs across horizons; stamp it into every cell
            prov = provenance.capture(arm_list)
            print(f"[provenance] {provenance.one_line(prov)}", flush=True)
        for arm in arm_list:
            oks, toks, lats, depths = [], [], [], []
            for nd in item.needles:
                if arm == "full":
                    ctx = arms.ctx_full(item, WINDOW)
                elif arm == "cr":
                    ctx = arms.ctx_cr(store, item, nd.question, limit=RETRIEVE_LIMIT)
                elif arm == "cr-enterprise":
                    ctx = arms.ctx_cr_enterprise(ent, nd.question, limit=RETRIEVE_LIMIT)
                else:  # cr-materialize (F2)
                    ctx, depth = f2_materialize(nd, item, store, ent, RETRIEVE_LIMIT)
                    depths.append(depth)
                t0 = time.perf_counter()
                out = arms.answer_oracle(ctx, nd.question) if dry else \
                    arms.answer_llm(client, name, ctx, nd.question, extra_body=extra)
                lats.append(time.perf_counter() - t0)
                oks.append(arms.scored(nd.answer, out))
                toks.append(arms.est_tokens(ctx))
            cell = {"corpus": corpus, "arm": arm, "horizon": H,
                    "model": "oracle" if dry else name, "window": WINDOW, "n": len(item.needles),
                    "acc": round(sum(oks) / len(oks), 3),
                    "input_tokens": int(st.mean(toks)),
                    "latency_s": round(st.mean(lats), 4),
                    "est_haystack_tokens": item.est_tokens}
            if depths:   # F2: which materialization depths the ladder chose across the needles
                cell["depth_dist"] = {d: depths.count(d) for d in sorted(set(depths))}
            cell["provenance"] = prov   # self-describing: which code produced this cell
            tag = "dry" if dry else name
            (RES / f"{corpus}__{arm}__{H}__{tag}.json").write_text(json.dumps(cell, indent=2))
            extra_col = f"  depths={cell['depth_dist']}" if depths else ""
            print(f"{H:>9} {arm:<14} {cell['acc']:>5} {cell['input_tokens']:>8} "
                  f"{cell['latency_s']:>7}{extra_col}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="10000,30000,100000,300000,1000000")
    ap.add_argument("--needles", type=int, default=int(os.environ.get("NEEDLES", "8")))
    ap.add_argument("--model", default=os.environ.get("MODEL", "qwen"))
    ap.add_argument("--dry", action="store_true", help="no LLM: oracle answerer (plumbing-only smoke)")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", "strategywiki"))
    ap.add_argument("--dump", default=os.environ.get("WIKI_DUMP"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("ARTICLE_LIMIT", "0")) or None,
                    help="cap #articles loaded (speeds up small smokes)")
    a = ap.parse_args()
    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]
    run(horizons, a.needles, a.model, a.dry, a.corpus, a.dump, a.seed, a.limit)


if __name__ == "__main__":
    main()
