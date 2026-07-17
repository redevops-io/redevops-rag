"""Turn an eval_cube2 generation-strategy ablation into a Context Runtime warm-start priors file.

Reads the per-cell JSONs eval_cube2 writes (``{dataset}__{method}__{model}__{strategy}.json``), computes
the per-intent-bucket strategy ladder from the **oracle** accuracies (generation isolated from
retrieval), and writes a compact ``{bucket: [strategies]}`` file. Point CR at it with
``CR_GENSTRATEGY_PRIORS=/path/to/genstrategy_priors.json`` and the answer-plane bandit warm-starts from
measured numbers instead of the hand-seeded defaults — closing the loop from Phase 0 → Phase 1.

    # after running the oracle ablation (CONDS=oracle STRATEGIES=direct,reason,decompose,mapreduce …):
    RES=/path/to/cube2_res OUT=/path/to/genstrategy_priors.json python benchmarks/build_priors.py
"""
import json
import os

from context_runtime.reasoner.strategies import model_competence_from_ablation, priors_from_ablation

RES = os.environ.get("RES", os.environ.get("OUT_DIR", "cube2_res"))
OUT = os.environ.get("OUT", "genstrategy_priors.json")
COND = os.environ.get("COND", "oracle")
MARGIN = float(os.environ.get("MARGIN", "0.1"))

priors = priors_from_ablation(RES, cond=COND, margin=MARGIN)
competence = model_competence_from_ablation(RES, cond=COND)
if not priors:
    raise SystemExit(f"no ablation cells found under {RES!r} — run eval_cube2 with a strategy sweep first")

# richer format: reasoning-strategy ladders + per-class model competence ('DeepSeek here, Qwen there')
payload = {"strategies": {b: list(s) for b, s in priors.items()},
           "model_competence": competence}
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)

print(f"wrote {OUT} from {RES} (cond={COND}, margin={MARGIN}):")
for bucket, strats in sorted(priors.items()):
    comp = competence.get(bucket) or {}
    best = f"  ·  best model: {max(comp, key=comp.get)}" if comp else ""
    print(f"  {bucket:14} → {', '.join(strats):22} (entry: {strats[0]}){best}")
print("\nApply it:  CR_GENSTRATEGY=1 CR_GENSTRATEGY_PRIORS=" + os.path.abspath(OUT))
