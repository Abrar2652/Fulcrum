"""Print a markdown summary of every benchmark result file (for EXPERIMENTS_ADDED.md).
Usage: PYTHONPATH=. python experiments/summarize_md.py > summary.md"""
from __future__ import annotations

import glob
import json
import os

from experiments.metrics import mean_ci, paired_t_interval

RES = "experiments/results"
KEY = ["BT-linear", "BT-MLP", "lexicographic", "Condorcet", "MAX-GAP rule", "linear-in-d",
       "context-BT (sum ctx)", "random-gate", "mixture-of-utilities", "antisym-bilinear", "blade-chest",
       "pair-MLP (matched)", "pair-MLP (large)", "FULCRUM-hard", "FULCRUM-soft",
       "FULCRUM-soft (fixed beta)", "FULCRUM-soft (no gate bias)"]


def ps(res, m, k):
    return [res["per_seed"][s][m].get(k, float("nan")) for s in res["per_seed"]]


def main():
    for path in sorted(glob.glob(f"{RES}/*.json")):
        tag = os.path.basename(path)[:-5]
        if tag in ("synthetic_all_summaries", "exhaustive_checks", "capacity_check", "intransitivity_counts"):
            continue
        res = json.load(open(path))
        if "per_seed" not in res:
            continue
        seeds = list(res["per_seed"])
        print(f"\n### {tag}  (seeds: {len(seeds)})")
        info0 = res["info"][seeds[0]]
        keys = [k for k in ("transitive_ceiling", "f3", "triangle_bound", "n_train", "n_test", "test_cyclic_triples") if k in info0]
        print("info (seed 0): " + ", ".join(f"{k}={info0[k]}" for k in keys))
        if "extra" in res:
            for s, ex in res["extra"].items():
                if ex:
                    print(f"ceilings seed {s}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ex.get('test_graph_ceilings', {}).items() if k != 'order'}))
        ref = ps(res, "BT-linear", "acc")
        print("| model | acc | ± | gap vs BT (paired t95) | NLL | Brier | ECE | on-cycle | off-cycle | cyc.-orient. | params |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in KEY:
            if m not in res["per_seed"][seeds[0]]:
                continue
            a = ps(res, m, "acc"); mu, h = mean_ci(a)
            g, gh = paired_t_interval([x - r for x, r in zip(a, ref)])
            def f(k, d=3):
                v = mean_ci(ps(res, m, k))[0]
                return "--" if v != v else f"{v:.{d}f}"
            print(f"| {m} | {mu:.4f} | {h:.4f} | {g:+.4f} ± {gh:.4f} | {f('nll')} | {f('brier')} | {f('ece')} | {f('acc_on_cycle')} | {f('acc_off_cycle')} | {f('cycle_orientation_acc', 2)} | {res['per_seed'][seeds[0]][m]['params']} |")
        # faithfulness
        for m in ("FULCRUM-hard", "FULCRUM-soft", "linear-in-d"):
            if m in res["per_seed"][seeds[0]] and "faithfulness" in res["per_seed"][seeds[0]][m]:
                fa = {}
                for grp in ("attributed", "random_other", "least"):
                    fa[grp] = {met: mean_ci([res["per_seed"][s][m]["faithfulness"][grp][met] for s in seeds])[0]
                               for met in ("flip_sign", "flip_delete", "sufficiency", "comprehensiveness")}
                print(f"faithfulness {m}: " + json.dumps({g: {k: round(v, 3) for k, v in d.items()} for g, d in fa.items()}))
        # slices
        sl = [k for k in res["per_seed"][seeds[0]]["BT-linear"] if k.startswith("acc_slice_")]
        if sl:
            print("| model | " + " | ".join(k[10:] for k in sl) + " |")
            print("|---|" + "---|" * len(sl))
            for m in KEY:
                if m in res["per_seed"][seeds[0]]:
                    print(f"| {m} | " + " | ".join(f"{mean_ci(ps(res, m, k))[0]:.4f}" for k in sl) + " |")


if __name__ == "__main__":
    main()
