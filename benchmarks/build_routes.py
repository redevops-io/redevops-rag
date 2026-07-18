"""Turn the answer cube into a COMPETENCE ROUTE table for cr-auto (finding #5).

The v5 cube showed cr-auto (regime→(retriever,strategy) heuristics) underperforming the best fixed
cell on every dataset — nutrition routed to hybrid (0.75) when reasonir scores 0.917. The fix is to
route from the MEASURED cube, not hand-coded regime rules. This reads the cube and emits, per class,
the (method, strategy) that actually scored best — the table cr-auto routes from.

Input: the cube CSV (``CUBE_CSV``; columns dataset,method,model,strategy,n,acc_closed,acc_oracle,
acc_retrieved) — the ``fixed`` method rows only (cr-auto's own row is skipped). Selection is by
``acc_oracle`` (generation isolated from retrieval — the promote signal), tie-broken by acc_retrieved
then the cheaper method. Emits two granularities:

  by_dataset — the competence CEILING: the best cell per dataset (what cr-auto could reach if it knew
               the class perfectly). Used by the benchmark to measure the routing LIFT.
  by_rep     — the realistic runtime route: best cell aggregated over the datasets sharing a router
               representation (document = popqa+nutrition, temporal = longmemeval+tempo, graph = musique).
               NB rep is coarse — document/temporal are internally heterogeneous, so by_rep is a lower
               bound than by_dataset; the gap is the argument for finer classification / the online bandit.

    CUBE_CSV=/path/cube.csv OUT=benchmarks/routes.json python benchmarks/build_routes.py
"""
import csv
import json
import os

CUBE_CSV = os.environ.get("CUBE_CSV", "cube_results/cube.csv")
OUT = os.environ.get("OUT", "routes.json")
SELECT = os.environ.get("SELECT_COND", "acc_oracle")   # promote on oracle; acc_retrieved to route on served
FIXED_METHODS = {"bm25", "hybrid", "reasonir", "diver", "graphiti", "hipporag"}

# dataset → the router representation it classifies as (aggregation key for by_rep).
DATASET_REP = {"popqa": "document", "nutrition": "document", "musique": "graph",
               "longmemeval": "temporal", "tempo": "temporal"}
# rough retriever cost order for the tie-break (cheaper wins a tie).
METHOD_COST = {"bm25": 0, "hybrid": 1, "reasonir": 2, "diver": 3, "graphiti": 4, "hipporag": 5}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def best_cell(cells):
    """The (method, strategy) with the highest SELECT score, tie-broken by acc_retrieved then cheaper."""
    def key(c):
        return (_num(c.get(SELECT)) or -1.0, _num(c.get("acc_retrieved")) or -1.0,
                -METHOD_COST.get(c["method"], 9))
    best = max(cells, key=key)
    return {"method": best["method"], "strategy": best["strategy"],
            "oracle": _num(best.get("acc_oracle")), "retrieved": _num(best.get("acc_retrieved"))}


def main():
    rows = [r for r in csv.DictReader(open(CUBE_CSV)) if r["method"] in FIXED_METHODS]
    if not rows:
        raise SystemExit(f"no fixed-method rows in {CUBE_CSV!r}")

    by_dataset = {}
    per_dataset = {}
    for r in rows:
        per_dataset.setdefault(r["dataset"], []).append(r)
    for ds, cells in per_dataset.items():
        by_dataset[ds] = best_cell(cells)

    # by_rep: pool each (method,strategy) across the rep's datasets, mean the SELECT score, pick argmax.
    rep_cells = {}
    for r in rows:
        rep = DATASET_REP.get(r["dataset"], "document")
        v = _num(r.get(SELECT))
        if v is None:
            continue
        rep_cells.setdefault(rep, {}).setdefault((r["method"], r["strategy"]), []).append(v)
    by_rep = {}
    for rep, combos in rep_cells.items():
        (method, strategy), mean = max(
            ((k, sum(v) / len(v)) for k, v in combos.items()),
            key=lambda kv: (kv[1], -METHOD_COST.get(kv[0][0], 9)))
        by_rep[rep] = {"method": method, "strategy": strategy, "score": round(mean, 4)}

    # AGG_TEMPORAL=1: point temporal at the NEW deterministic map-reduce aggregator (strategy
    # "aggregate") instead of the cube's mapreduce — the 35B can't compose across sessions, so the
    # next run reduces in code. Only affects temporal datasets/rep; the retriever is kept.
    if os.environ.get("AGG_TEMPORAL", "").lower() in ("1", "true", "yes", "on"):
        for ds, r in by_dataset.items():
            if DATASET_REP.get(ds) == "temporal":
                r["strategy"] = "aggregate"
        if "temporal" in by_rep:
            by_rep["temporal"]["strategy"] = "aggregate"

    payload = {"select_cond": SELECT, "by_dataset": by_dataset, "by_rep": by_rep,
               "dataset_rep": DATASET_REP}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {OUT} from {CUBE_CSV} (select={SELECT}):")
    print("  by_dataset (competence ceiling):")
    for ds, r in sorted(by_dataset.items()):
        print(f"    {ds:12} → {r['method']:9}/{r['strategy']:9} (oracle {r['oracle']}, retr {r['retrieved']})")
    print("  by_rep (runtime route):")
    for rep, r in sorted(by_rep.items()):
        print(f"    {rep:10} → {r['method']:9}/{r['strategy']:9} (mean {r['score']})")
    print("\nApply it:  CR_ROUTES=" + os.path.abspath(OUT) + "  (cr-auto routes from the cube, not regime rules)")


if __name__ == "__main__":
    main()
