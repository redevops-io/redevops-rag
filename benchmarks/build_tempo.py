"""Rebuild the temporal-reasoning dataset from TempReason → the cube schema.

The old tempo.jsonl was workplace career-advice essays mislabeled `temporal` — 300-800-word HTML gold
no terse answer can match, so tempo=0 was a SCORING ARTIFACT (the Run-2 audit). This builds a REAL
temporal-reasoning set from TempReason (arXiv:2306.08952): L2 (time→event: "which employer in Jan 1948?")
and L3 (event→event: ordering), point-in-time / ordering questions with CRISP entity answers over a set
of time-scoped facts. The reasoning is genuine — the model must pick the fact whose time range covers the
query date (or order events), not extract a single string; `neg_answers` are the confusable other-time
entities.

Emits the cube schema {qid, question, answer, aliases, docs:[{chunk_id,text,created_at}], gold, regime}:
docs = the person's/entity's time-scoped facts (the timeline); gold = the whole timeline (oracle then
tests reasoning-over-the-timeline, not lookup); created_at parsed from each fact's start date.

    TEMPO_LEVELS=l2,l3 TEMPO_N=300 OUT=/path/datasets/tempo.jsonl python benchmarks/build_tempo.py
"""
import hashlib
import json
import os
import re

from huggingface_hub import hf_hub_download

REPO = os.environ.get("TEMPO_REPO", "tonytan48/TempReason")
LEVELS = [x.strip() for x in os.environ.get("TEMPO_LEVELS", "l2,l3").split(",") if x.strip()]
N = int(os.environ.get("TEMPO_N", "300"))        # per level (deterministic sample)
OUT = os.environ.get("OUT", "tempo.jsonl")

_MON = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _created(fact):
    """First date in a fact ('... from Jan, 1949 to ...') → YYYY-MM-01, for the cube's created_at."""
    m = re.search(r"([A-Za-z]{3})[a-z]*\.?,?\s*(\d{4})", fact)
    if m:
        return f"{m.group(2)}-{_MON.get(m.group(1).lower(), 1):02d}-01"
    m = re.search(r"\b(\d{4})\b", fact)
    return f"{m.group(1)}-01-01" if m else None


def convert(ex):
    ans = ((ex.get("text_answers") or {}).get("text")) or []
    ans = [a for a in ans if str(a).strip()]
    facts = [f.strip() for f in (ex.get("fact_context") or "").split("\n") if f.strip()]
    if not ans or len(facts) < 2:            # need a real timeline to reason over
        return None
    qid = ex.get("id") or "t_" + hashlib.md5(ex["question"].encode()).hexdigest()[:12]
    docs = [{"chunk_id": f"{qid}::f{i}", "text": f, "created_at": _created(f)} for i, f in enumerate(facts)]
    return {"qid": qid, "question": ex["question"].strip(), "answer": str(ans[0]).strip(),
            "aliases": [str(a).strip() for a in ans[1:]],
            "docs": docs, "gold": [d["chunk_id"] for d in docs], "regime": "temporal"}


def main():
    rows = []
    for lvl in LEVELS:
        path = hf_hub_download(REPO, f"test_{lvl}.json", repo_type="dataset")
        data = [json.loads(l) for l in open(path) if l.strip()]
        # deterministic sample by id hash so re-runs are stable
        data.sort(key=lambda e: hashlib.md5(str(e.get("id", e.get("question", ""))).encode()).hexdigest())
        kept = 0
        for ex in data:
            it = convert(ex)
            if it:
                it["level"] = lvl
                rows.append(it)
                kept += 1
            if kept >= N:
                break
        print(f"  {lvl}: {kept} questions", flush=True)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} temporal-reasoning questions → {OUT}")
    if rows:
        ex = rows[0]
        print(f"  sample: Q={ex['question']!r}  A={ex['answer']!r}  n_facts={len(ex['docs'])}")


if __name__ == "__main__":
    main()
