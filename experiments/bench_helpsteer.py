"""HelpSteer2-Preference: human pairwise preference between two responses, each annotated
on five aspects (helpfulness, correctness, coherence, complexity, verbosity; 0-4).

The preference annotations (Wang et al., 2024, 'HelpSteer2-Preference') were collected
separately from the aspect ratings (HelpSteer2). We join the two by exact (prompt,
response) text. Criteria for the comparator: the five aspect scores of each response.
Labels: 1 if response_1 is preferred (preference_strength < 0), 0 if response_2 is
preferred (> 0); ties (strength 0) are excluded from the main analysis and counted.
Split: the released `train` rows are split into train/validation by prompt (seeded); the
released `validation` rows are the fixed test set. Each response appears once, so the
comparison graph is a matching: no cycle statistics are available here, and the
experiment tests only the multi-criterion pairwise comparator.

Data files are read from the HuggingFace cache directory given by --hs_dir (downloaded
with huggingface_hub; dataset licence CC-BY-4.0).

Usage: PYTHONPATH=. python experiments/bench_helpsteer.py --hs_dir <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random

import torch

from experiments.bench_common import print_summary, run_benchmark
from experiments.zoo import PairData, make_zoo

ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
M = K = 5
SEEDS = list(range(10))


def load(hs_dir):
    def rd(p):
        with gzip.open(p, "rt") as fh:
            return [json.loads(l) for l in fh]
    attr = {}
    for fn in ("train.jsonl.gz", "validation.jsonl.gz"):
        for r in rd(os.path.join(hs_dir, fn)):
            attr[(r["prompt"], r["response"])] = [float(r[a]) for a in ATTRS]
    pref = rd(os.path.join(hs_dir, "preference", "preference.jsonl.gz"))
    rows = {"train": [], "val": []}
    stats = {"pref_rows": len(pref), "ties": 0, "unmatched": 0}
    for r in pref:
        k1, k2 = (r["prompt"], r["response_1"]), (r["prompt"], r["response_2"])
        if k1 not in attr or k2 not in attr:
            stats["unmatched"] += 1; continue
        s = r["preference_strength"]
        if s == 0:
            stats["ties"] += 1; continue
        rows[r["split"]].append((attr[k1], attr[k2], 1.0 if s < 0 else 0.0, r["prompt"], abs(s)))
    stats["train_pairs"] = len(rows["train"]); stats["test_pairs"] = len(rows["val"])
    return rows, stats


ROWS, STATS = None, None


def to_pairdata(items):
    xa = torch.tensor([it[0] for it in items]); xb = torch.tensor([it[1] for it in items])
    y = torch.tensor([it[2] for it in items])
    strength = torch.tensor([it[4] for it in items])
    return PairData(xa - xb, xa, xb, y, meta={"strength": strength, "help_gap": (xa[:, 0] - xb[:, 0]).abs()})


def make_split_factory(hs_dir):
    def make_split(seed):
        global ROWS, STATS
        if ROWS is None:
            ROWS, STATS = load(hs_dir)
        rng = random.Random(seed)
        prompts = sorted({it[3] for it in ROWS["train"]})
        rng.shuffle(prompts)
        va_p = set(prompts[: int(0.1 * len(prompts))])
        tr = [it for it in ROWS["train"] if it[3] not in va_p]
        va = [it for it in ROWS["train"] if it[3] in va_p]
        te = ROWS["val"]
        info = {"dataset": "helpsteer2_preference", **STATS}
        return to_pairdata(tr), to_pairdata(va), to_pairdata(te), info
    return make_split


def slices(te: PairData) -> dict:
    s = te.meta["strength"]
    hg = te.meta["help_gap"]   # raw helpfulness gap (0-4 scale)
    return {"strength1": s == 1, "strength2": s == 2, "strength3": s == 3,
            "help_gap0": hg < 0.5, "help_gap1": (hg >= 0.5) & (hg < 1.5), "help_gap2+": hg >= 1.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs_dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    zoo = make_zoo(M, K)
    res = run_benchmark("helpsteer2_pref", make_split_factory(args.hs_dir), zoo, seeds,
                        out_dir=args.out, epochs=600, lr=0.05, verbose=True, slices=slices)
    print_summary(res, "acc")
    print(json.dumps(res["info"][str(seeds[0])], indent=1))


if __name__ == "__main__":
    main()
