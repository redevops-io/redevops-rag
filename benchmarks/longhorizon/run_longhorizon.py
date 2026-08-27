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


def build_store(emb, item):
    from redevops_rag.store import Store
    ch = [dict(c) for c in item.chunks]
    for c, e in zip(ch, emb.encode([c["text"] for c in ch])):
        c["embedding"] = e
    s = Store(emb, ":memory:")
    s.add_chunks(ch, reindex=True)
    return s


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

    print(f"{'horizon':>9} {'arm':<5} {'acc':>5} {'in_tok':>8} {'lat_s':>7}", flush=True)
    for H in horizons:
        item = build_item(articles, H, n_needles=n_needles, item_id=f"{corpus}-{H}", seed=seed)
        store = build_store(emb, item)
        for arm in ("full", "cr"):
            oks, toks, lats = [], [], []
            for nd in item.needles:
                ctx = arms.ctx_full(item, WINDOW) if arm == "full" \
                    else arms.ctx_cr(store, item, nd.question, limit=RETRIEVE_LIMIT)
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
            tag = "dry" if dry else name
            (RES / f"{corpus}__{arm}__{H}__{tag}.json").write_text(json.dumps(cell, indent=2))
            print(f"{H:>9} {arm:<5} {cell['acc']:>5} {cell['input_tokens']:>8} {cell['latency_s']:>7}", flush=True)


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
