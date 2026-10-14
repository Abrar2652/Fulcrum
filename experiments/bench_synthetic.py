"""Synthetic benchmarks (paper Section 6.1 and appendix): pair-dependent comparators
versus transitive and non-transitive baselines on two controlled worlds.

World C (Condorcet oracle, misspecified for every model): items with m = 5 criteria
correlated at level r through a shared latent factor; i beats j iff i is higher on a
majority of criteria (ties dropped). The oracle is NOT MAX-GAP. r = 1 is the transitive
negative control. Train, validation and test use disjoint fresh item sets.

World M (MAX-GAP world, well specified for FULCRUM-hard): items with m = 5 independent
criteria; hidden per-criterion scales sigma_h; i beats j by MAX-GAP on sigma * c.

Sweeps: criteria correlation r; label noise eps (train/validation only; test labels are
clean); training-set fraction; criterion-identity shuffle (a random permutation of the
criterion axes per pair, applied consistently to a and b, in train/val/test).

For every test tournament we compute the exact transitivity ceiling (lazy-constraint
linear-ordering ILP), the cyclic-triple fraction f3 and the closed-form bound 1 - f3/3.

Usage:
  PYTHONPATH=. python experiments/bench_synthetic.py --smoke        # 1 seed, 2 configs
  PYTHONPATH=. python experiments/bench_synthetic.py                # full (10 seeds)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import torch

from experiments.bench_common import print_summary, run_benchmark
from experiments.ceiling import (counts_from_outcomes, transitive_ceiling_exact_lazy,
                                 triangle_bound)
from experiments.zoo import PairData, SoftGate, make_zoo

M = 5
N_ITEMS = 60
SEEDS = list(range(10))


def make_items(n, m, rho, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 1, generator=g)
    eps = torch.randn(n, m, generator=g)
    return math.sqrt(max(rho, 0.0)) * z + math.sqrt(1 - max(rho, 0.0)) * eps


def condorcet(xi, xj):
    w = int((xi > xj).sum()); l = int((xi < xj).sum())
    return 1 if w > l else (-1 if l > w else 0)


def maxgap_truth(xi, xj, sigma):
    d = (xi - xj) * sigma
    h = int(d.abs().argmax())
    return 1 if d[h] > 0 else (-1 if d[h] < 0 else 0)


def pairs_from_items(X, rule, seed, noise=0.0, frac=1.0, shuffle=False):
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    xa, xb, y, ia, ib = [], [], [], [], []
    for i, j in itertools.combinations(range(n), 2):
        lab = rule(X[i], X[j])
        if lab == 0:
            continue
        xa.append(X[i]); xb.append(X[j]); y.append(1.0 if lab == 1 else 0.0); ia.append(i); ib.append(j)
    xa = torch.stack(xa); xb = torch.stack(xb); y = torch.tensor(y)
    ia = torch.tensor(ia); ib = torch.tensor(ib)
    if frac < 1.0:
        k = max(20, int(frac * len(y)))
        perm = torch.randperm(len(y), generator=g)[:k]
        xa, xb, y, ia, ib = xa[perm], xb[perm], y[perm], ia[perm], ib[perm]
    if noise > 0:
        flip = torch.rand(len(y), generator=g) < noise
        y = torch.where(flip, 1 - y, y)
    if shuffle:
        # random permutation of the criterion axes per pair (same for a and b)
        perms = torch.stack([torch.randperm(X.shape[1], generator=g) for _ in range(len(y))])
        xa = torch.gather(xa, 1, perms); xb = torch.gather(xb, 1, perms)
    return PairData(xa - xb, xa, xb, y, ia, ib)


def test_ceilings(te: PairData) -> dict:
    counts = counts_from_outcomes(zip(te.ia.tolist(), te.ib.tolist()), te.y.tolist())
    tb = triangle_bound(counts)
    ex = transitive_ceiling_exact_lazy(counts, time_limit=900)
    return {"transitive_ceiling": ex["ceiling"], "ceiling_status": ex["status"],
            "ceiling_seconds": ex["seconds"], "f3": tb["f3"], "triangle_bound": tb["bound"],
            "cyclic_triples": tb["cyclic_triples"]}


def make_split_factory(world="C", r=0.0, noise=0.0, frac=1.0, shuffle=False, ceilings=True):
    def make_split(seed):
        if world == "C":
            Xtr = make_items(N_ITEMS, M, r, seed)
            Xva = make_items(N_ITEMS, M, r, seed + 500)
            Xte = make_items(N_ITEMS, M, r, seed + 1000)
            rule = condorcet
        else:
            g = torch.Generator().manual_seed(seed + 9000)
            sigma = torch.exp((torch.rand(M, generator=g) * 2 - 1) * math.log(2.0))
            Xtr = make_items(N_ITEMS, M, 0.0, seed)
            Xva = make_items(N_ITEMS, M, 0.0, seed + 500)
            Xte = make_items(N_ITEMS, M, 0.0, seed + 1000)
            rule = lambda a, b: maxgap_truth(a, b, sigma)  # noqa: E731
        tr = pairs_from_items(Xtr, rule, seed + 1, noise=noise, frac=frac, shuffle=shuffle)
        va = pairs_from_items(Xva, rule, seed + 2, noise=noise, frac=frac, shuffle=shuffle)
        te = pairs_from_items(Xte, rule, seed + 3, noise=0.0, frac=1.0, shuffle=shuffle)
        info = {"world": world, "r": r, "noise": noise, "train_frac": frac, "shuffle": shuffle}
        if ceilings:
            info.update(test_ceilings(te))
        return tr, va, te, info
    return make_split


def zoo_with_variants():
    z = make_zoo(M, M)
    z["FULCRUM-soft (fixed beta)"] = lambda: SoftGate(M, M, beta_init=4.0, learn_beta=False)
    z["FULCRUM-soft (no gate bias)"] = lambda: SoftGate(M, M, gate_bias=False)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--only", default=None, help="run only the config with this tag")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    zoo = zoo_with_variants()
    configs = []
    # S1: correlation sweep (main table)
    for r in (1.0, 0.7, 0.4, 0.2, 0.0):
        configs.append((f"synC_r{r:.1f}", dict(world="C", r=r)))
    # S2: label noise at r = 0
    for eps in (0.1, 0.2, 0.3):
        configs.append((f"synC_r0.0_noise{eps:.1f}", dict(world="C", r=0.0, noise=eps)))
    # S3: training size at r = 0
    for frac in (0.1, 0.25, 0.5):
        configs.append((f"synC_r0.0_frac{frac:.2f}", dict(world="C", r=0.0, frac=frac)))
    # S4: criterion identity shuffled at r = 0
    configs.append(("synC_r0.0_shuffle", dict(world="C", r=0.0, shuffle=True)))
    # S5: MAX-GAP world with noise 0.1, full and small training sets
    configs.append(("synM_noise0.1", dict(world="M", noise=0.1)))
    for frac in (0.1, 0.25):
        configs.append((f"synM_noise0.1_frac{frac:.2f}", dict(world="M", noise=0.1, frac=frac)))
    # criterion identity shuffled in World M (where criterion-specific scales matter)
    configs.append(("synM_noise0.1_shuffle", dict(world="M", noise=0.1, shuffle=True)))
    if args.smoke:
        configs = [configs[4], configs[-4]]
    if args.only:
        configs = [c for c in configs if c[0] == args.only]
    all_summaries = {}
    for tag, kw in configs:
        t0 = time.time()
        res = run_benchmark(tag, make_split_factory(**kw), zoo, seeds, out_dir=args.out,
                            epochs=400, lr=0.05, verbose=args.smoke)
        print_summary(res, "acc")
        info0 = res["info"][str(seeds[0])]
        print(f"  ceiling(seed {seeds[0]}) = {info0.get('transitive_ceiling')}, f3 = {info0.get('f3')}, "
              f"bound = {info0.get('triangle_bound')}  [{time.time() - t0:.0f}s]")
        all_summaries[tag] = {"summary": res["summary"], "info": res["info"]}
        if not args.only:
            json.dump(all_summaries, open(os.path.join(args.out, "synthetic_all_summaries.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
