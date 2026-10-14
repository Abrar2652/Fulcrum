"""Sushi-B: human rankings over 100 sushi items with item attributes (Kamishima 2003).

Each of 5,000 respondents ranked 10 items sampled from a set of 100; every ranking yields
45 pairwise comparisons, so the same item pairs are compared repeatedly by different
people (a dense, natural comparison graph). Criteria: style (maki / other), major group
(seafood / other), heaviness-oiliness, eating frequency, normalized price, sale frequency
(six per-item attributes). Extra per-item features for the transitive baselines: the
12-way minor-group one-hot. The task: predict a held-out respondent's pairwise preference
from item attributes alone; the target is therefore a population preference and the pair
ceiling (majority per pair) bounds every model. Split by respondent (60/20/20).

Usage: PYTHONPATH=. python experiments/bench_sushi.py [--smoke]
"""
from __future__ import annotations

import argparse
import itertools
import os

import torch

from experiments.bench_common import print_summary, run_benchmark
from experiments.ceiling import counts_from_outcomes, transitive_ceiling_exact_lazy
from experiments.zoo import PairData, make_zoo

DATA = "data/sushi3-2016"
SEEDS = list(range(10))
M = 6
K = 6 + 12


def load_items():
    attrs, minor = {}, {}
    for line in open(f"{DATA}/sushi3.idata"):
        p = line.rstrip("\n").split("\t")
        iid = int(p[0])
        attrs[iid] = [float(p[2]), float(p[3]), float(p[5]), float(p[6]), float(p[7]), float(p[8])]
        minor[iid] = int(p[4])
    A = torch.tensor([attrs[i] for i in range(100)])
    OH = torch.zeros(100, 12)
    for i in range(100):
        OH[i, minor[i]] = 1.0
    return torch.cat([A, OH], 1)


def load_orders():
    out = []
    with open(f"{DATA}/sushi3b.5000.10.order") as f:
        next(f)
        for line in f:
            p = line.split()
            if not p:
                continue
            k = int(p[1]); out.append([int(x) for x in p[2:2 + k]])
    return out


def build_pairs(orders, X, seed):
    g = torch.Generator().manual_seed(seed)
    pa, pb, uid = [], [], []
    for u, r in enumerate(orders):
        for i, j in itertools.combinations(range(len(r)), 2):      # r[i] preferred to r[j]
            pa.append(r[i]); pb.append(r[j]); uid.append(u)
    pa = torch.tensor(pa); pb = torch.tensor(pb); uid = torch.tensor(uid)
    flip = torch.rand(len(pa), generator=g) < 0.5                  # random presentation order
    ia = torch.where(flip, pb, pa); ib = torch.where(flip, pa, pb)
    y = torch.where(flip, torch.zeros(len(pa)), torch.ones(len(pa)))
    xa, xb = X[ia], X[ib]
    return PairData(xa[:, :M] - xb[:, :M], xa, xb, y, ia, ib, meta={"user": uid})


X_ITEMS = load_items()
ORDERS = load_orders()


def make_split(seed):
    allp = build_pairs(ORDERS, X_ITEMS, seed)
    g = torch.Generator().manual_seed(seed + 77)
    users = torch.randperm(len(ORDERS), generator=g)
    n_te = int(0.2 * len(users)); n_va = int(0.2 * len(users))
    te_u = set(users[:n_te].tolist()); va_u = set(users[n_te:n_te + n_va].tolist())
    u = allp.meta["user"]
    te_m = torch.tensor([int(x) in te_u for x in u.tolist()])
    va_m = torch.tensor([int(x) in va_u for x in u.tolist()])
    tr_m = ~(te_m | va_m)
    def sub(msk):
        p = allp.subset(msk.nonzero().squeeze(1)); p.meta = {}
        return p
    info = {"dataset": "sushi_b", "n_items": 100, "users": len(ORDERS)}
    return sub(tr_m), sub(va_m), sub(te_m), info


def ceilings(seed, tr, va, te, info):
    if seed >= 3:                      # exact ILP on the 100-item test graph; three seeds suffice
        return {}
    counts = counts_from_outcomes(zip(te.ia.tolist(), te.ib.tolist()), te.y.tolist())
    ex = transitive_ceiling_exact_lazy(counts, time_limit=600)
    from experiments.ceiling import pair_ceiling
    return {"test_graph_ceilings": {"transitive_ceiling": ex["ceiling"], "status": ex["status"],
                                    "seconds": ex["seconds"], "pair_ceiling": pair_ceiling(counts)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    zoo = make_zoo(M, K, n_items=100)
    res = run_benchmark("sushi_b", make_split, zoo, seeds, out_dir=args.out, epochs=600, lr=0.05,
                        batch_size=None, verbose=True, extra=ceilings)
    print_summary(res, "acc")


if __name__ == "__main__":
    main()
