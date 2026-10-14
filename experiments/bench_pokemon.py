"""Pokemon battle outcomes with the FULL multi-criterion comparator (paper Section 6.2).

Data: Kaggle 'Weedle's Cave' (data/pokemon/combats.csv, pokemon.csv): 50,000 observed battle
outcomes among 784 Pokemon with six base stats and one or two types. Criteria for the
comparator: the six standardized stat differences plus the pair-level type-advantage
term log off(i,j) - log off(j,i) (antisymmetric; it has no per-item counterpart, which is
exactly what makes it inexpressible by a single ordering). Per-item features for the
transitive baselines: six stats and 18 type one-hots. A shared intercept absorbs the
first-listed-position effect (the first-listed Pokemon wins 47.2% of battles).

Prespecified test slices: all battles; cross-type battles (non-zero type term); battles in
the lowest 30/20/10% of |total base-stat difference| (a model-free closeness measure).
Ceilings on the test comparison graph: pair ceiling (majority per pair), transitive ceiling
bounds (achievable ordering after local search; LP/lazy-ILP upper bound).

Usage: PYTHONPATH=. python experiments/bench_pokemon.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from experiments.bench_common import print_summary, run_benchmark
from experiments.ceiling import counts_from_outcomes, transitive_ceiling_bounds
from experiments.pokemon_intransitivity import TI, TYPES, load, type_adv
from experiments.zoo import PairData, make_zoo

SEEDS = list(range(10))
M = 7          # 6 stats + type advantage
K = 6 + 1 + len(TYPES)


def build_all():
    mons, battles = load()
    ids = sorted(mons)
    idx = {i: k for k, i in enumerate(ids)}
    S = torch.tensor([mons[i]["stats"] for i in ids])
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    OH = torch.zeros(len(ids), len(TYPES))
    for k, i in enumerate(ids):
        for t in mons[i]["types"]:
            OH[k, TI[t]] = 1.0
    X = torch.cat([S, torch.zeros(len(ids), 1), OH], 1)          # per-item features
    a = torch.tensor([idx[b[0]] for b in battles]); b_ = torch.tensor([idx[b[1]] for b in battles])
    y = torch.tensor([b[2] for b in battles])
    tadv = torch.tensor([type_adv(mons, b[0], b[1]) for b in battles])
    d = torch.cat([S[a] - S[b_], tadv[:, None]], 1)                 # criteria differences
    raw_tot = torch.tensor([sum(mons[i]["stats"]) for i in ids])
    closeness = (raw_tot[a] - raw_tot[b_]).abs()
    return PairData(d, X[a], X[b_], y, a, b_, meta={"closeness": closeness, "tadv": tadv}), len(ids)


ALL, N_ITEMS = None, None


def make_split(seed):
    global ALL, N_ITEMS
    if ALL is None:
        ALL, N_ITEMS = build_all()
    g = torch.Generator().manual_seed(seed)
    n = len(ALL)
    perm = torch.randperm(n, generator=g)
    n_te = int(0.2 * n); n_va = int(0.1 * n)
    te_i, va_i, tr_i = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    def sub(ix):
        p = ALL.subset(ix)
        p.meta = {k: v[ix] for k, v in ALL.meta.items()}
        return p
    tr, va, te = sub(tr_i), sub(va_i), sub(te_i)
    info = {"dataset": "pokemon", "n_items": N_ITEMS}
    return tr, va, te, info


def slices(te: PairData) -> dict:
    c = te.meta["closeness"]; t = te.meta["tadv"]
    out = {"all": torch.ones(len(te), dtype=torch.bool), "cross_type": t.abs() > 1e-6}
    for q in (0.3, 0.2, 0.1):
        thr = torch.quantile(c.float(), q)
        out[f"close{int(q * 100)}"] = c <= thr
    out["cross_type_close30"] = out["cross_type"] & out["close30"]
    return out


def ceilings(seed, tr, va, te, info):
    if seed >= 3:                      # the LP bound is expensive; three seeds suffice
        return {}
    counts = counts_from_outcomes(zip(te.ia.tolist(), te.ib.tolist()), te.y.tolist())
    t0 = time.time()
    bd = transitive_ceiling_bounds(counts, exact_ilp=False, time_limit=1500, max_lp_rounds=120)
    bd = {k: v for k, v in bd.items() if k != "order"}
    bd["seconds"] = time.time() - t0
    return {"test_graph_ceilings": bd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--no-ceiling", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    global ALL, N_ITEMS
    ALL, N_ITEMS = build_all()
    zoo = make_zoo(M, K, n_items=N_ITEMS)
    res = run_benchmark("pokemon_full", make_split, zoo, seeds, out_dir=args.out, epochs=600,
                        lr=0.05, batch_size=None, verbose=True, slices=slices,
                        extra=None if args.no_ceiling else ceilings)
    print_summary(res, "acc")
    for s in ("close30", "close20", "close10", "cross_type"):
        print(f"\n--- slice {s} ---")
        per = res["per_seed"]
        for mname in per[str(seeds[0])]:
            xs = [per[str(sd)][mname][f"acc_slice_{s}"] for sd in seeds]
            print(f"  {mname:24s} {sum(xs) / len(xs):.4f}")


if __name__ == "__main__":
    main()
