"""Compute transitivity-ceiling brackets for the Sushi-B test graphs (seeds 0-2) with the
LP/local-search bounds; written to experiments/results/sushi_b_ceilings.json and merged by make_tables.py.
Usage: PYTHONPATH=. python experiments/ceilings_post.py [time_limit_seconds]"""
import json, sys, time
from experiments.bench_sushi import make_split
from experiments.ceiling import counts_from_outcomes, transitive_ceiling_bounds, pair_ceiling
tl = float(sys.argv[1]) if len(sys.argv) > 1 else 1500
p = "experiments/results/sushi_b_ceilings.json"     # separate file: merged by make_tables.py
res = {"extra": {}}
for seed in (0, 1, 2):
    tr, va, te, info = make_split(seed)
    counts = counts_from_outcomes(zip(te.ia.tolist(), te.ib.tolist()), te.y.tolist())
    t0 = time.time()
    bd = transitive_ceiling_bounds(counts, exact_ilp=True, time_limit=tl, max_lp_rounds=150)
    bd = {k: v for k, v in bd.items() if k != "order"}
    bd["pair_ceiling"] = pair_ceiling(counts); bd["seconds"] = time.time() - t0
    res["extra"][str(seed)] = {"test_graph_ceilings": bd}
    print(seed, bd, flush=True)
    json.dump(res, open(p, "w"), indent=1)
