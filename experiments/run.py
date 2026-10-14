"""Phase-6 H1 (accuracy) + H3 (auditability) on the synthetic generator, with the
pre-registered ablations A1 (multi-head off) and A2 (context off = single ordering).
No external deps; deterministic. H2 (learned calibration) is deferred to Phase 5.
Baselines here are the honest best-single-ordering / best-single-head fits (NOT
strawmen); established-library ASPIC+/ABA+/VAF adapters are env-blocked and deferred
(the pre-registration notes prerequisite (c)).

Usage: python experiments/run.py
"""
from __future__ import annotations

import itertools
import math
import statistics

from experiments.synthetic import N_HEADS, make_dataset, split

ORDERINGS = list(itertools.permutations(range(N_HEADS)))
SEEDS = [11, 22, 33, 44, 55, 66, 77]  # >=5 (the pre-registration notes §5)


def _lex_decide(va, vb, order) -> int:
    for h in order:
        if va[h] != vb[h]:
            return 1 if va[h] > vb[h] else -1
    return 1  # total tie -> a (rare; scores chosen so deciding head differs)


def _acc(data, predict) -> float:
    return sum(predict(c) == c.winner for c in data) / len(data)


def fit_single_ordering(train):
    """A2 baseline: one global ordering maximizing train accuracy."""
    best = max(ORDERINGS, key=lambda o: _acc(train, lambda c: _lex_decide(c.va, c.vb, o)))
    return lambda c: _lex_decide(c.va, c.vb, best)


def fit_single_head(train):
    """A1 ablation: decide by a single global head (no multi-head lexicographic)."""
    best_h = max(range(N_HEADS),
                 key=lambda h: _acc(train, lambda c: (1 if c.va[h] > c.vb[h] else -1)))
    return lambda c: (1 if c.va[best_h] > c.vb[best_h] else -1)


def fit_context_calculus(train):
    """Full calculus: per-context best ordering (the fitted reorder policy)."""
    by_ctx: dict[int, list] = {}
    for c in train:
        by_ctx.setdefault(c.ctx, []).append(c)
    best_order = {}
    for ctx, items in by_ctx.items():
        best_order[ctx] = max(
            ORDERINGS, key=lambda o: _acc(items, lambda c: _lex_decide(c.va, c.vb, o))
        )
    # fallback ordering for unseen contexts = global best
    glob = max(ORDERINGS, key=lambda o: _acc(train, lambda c: _lex_decide(c.va, c.vb, o)))

    def predict(c):
        return _lex_decide(c.va, c.vb, best_order.get(c.ctx, glob))
    return predict, best_order, glob


def _deciding_head(va, vb, order):
    for h in order:
        if va[h] != vb[h]:
            return h
    return order[0]


def auditability(test, best_order, glob) -> float:
    """H3 intervention test: flip the deciding head's scores; verdict must flip."""
    flips = 0
    for c in test:
        order = best_order.get(c.ctx, glob)
        h = _deciding_head(c.va, c.vb, order)
        v0 = _lex_decide(c.va, c.vb, order)
        # intervene: swap the deciding head's two scores
        va2 = list(c.va); vb2 = list(c.vb)
        va2[h], vb2[h] = vb2[h], va2[h]
        v1 = _lex_decide(tuple(va2), tuple(vb2), order)
        flips += (v1 == -v0)
    return flips / len(test)


def ci95(xs):
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = statistics.stdev(xs)
    t = 2.447 if len(xs) == 7 else 2.776  # t(.975, df) approx for n=7/5
    return m, t * sd / math.sqrt(len(xs))


def run_regime(regime: str, n=1200):
    rows = {"full (context calculus)": [], "A2 single ordering": [], "A1 single head": []}
    audit = []
    for seed in SEEDS:
        data, _ = make_dataset(n, seed=seed, regime=regime)
        tr, te = split(data, 0.6, seed=seed + 1)
        full_pred, best_order, glob = fit_context_calculus(tr)
        rows["full (context calculus)"].append(_acc(te, full_pred))
        rows["A2 single ordering"].append(_acc(te, fit_single_ordering(tr)))
        rows["A1 single head"].append(_acc(te, fit_single_head(tr)))
        audit.append(auditability(te, best_order, glob))
    return rows, audit


def main():
    print("=" * 68)
    print("PHASE 6 — H1 (accuracy) + H3 (auditability), synthetic ground truth")
    print(f"seeds={SEEDS}  (mean ± 95% CI on held-out test)")
    print("=" * 68)
    for regime, note in [("context", "context-dependent world -> calculus SHOULD win"),
                         ("transitive", "single-ordering world -> NEGATIVE CONTROL: tie")]:
        rows, audit = run_regime(regime)
        print(f"\n[{regime}]  {note}")
        for name, accs in rows.items():
            m, h = ci95(accs)
            print(f"    {name:26} acc = {m:.3f} ± {h:.3f}")
        m, h = ci95(audit)
        print(f"    H3 auditability (verdict flips on intervention) = {m:.3f} ± {h:.3f}")
    print("\n(Interpretation is pre-registered in the pre-registration notes; "
          "H2/learned + established-lib baselines + real data are deferred.)")


if __name__ == "__main__":
    main()
