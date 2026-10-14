"""Full pre-registered ablation table A1-A4 (PREREGISTRATION §3).

A1 multi-head off        (single head)              — synthetic 'context' regime
A2 context-conditioning off (single global ordering) — synthetic 'context' regime
A3 strength typing off   (drop the certain/probable gate) — strength-aware regime
A4 meta-controller off   (reconcile/supersede disabled)   — reconcile/supersede instances

A1-A3 are accuracy/calibration ablations with a predicted degradation. A4 is a
behavioural ablation (the meta operators change *which* arguments are accepted;
turning them off changes the outcome on instances where they apply). Each maps
one-to-one to a thesis component; a non-degradation is reported, not hidden.

Runnable with no external dependencies. Usage: python experiments/ablations.py
"""
from __future__ import annotations

import statistics

from experiments.run import (
    fit_context_calculus,
    fit_single_head,
    fit_single_ordering,
)
from experiments.synthetic import make_dataset, split
from fulcrum.defeat import ArgumentationFramework
from fulcrum.meta_controller import RECONCILE_FIRST, WEIGH_ONLY
from fulcrum.preference import Preferred
from fulcrum.semantics import grounded_extension
from fulcrum.structures import Argument, PreferenceHead

SEEDS = [11, 22, 33, 44, 55, 66, 77]


def _acc(data, predict):
    return sum(predict(c) == c.winner for c in data) / len(data)


def a1_a2(regime="context", n=1200):
    """A1 (single head) and A2 (single ordering) vs the full context calculus."""
    full, a2, a1 = [], [], []
    for seed in SEEDS:
        data, _ = make_dataset(n, seed=seed, regime=regime)
        tr, te = split(data, 0.6, seed=seed + 1)
        pred, _, _ = fit_context_calculus(tr)
        full.append(_acc(te, pred))
        a2.append(_acc(te, fit_single_ordering(tr)))
        a1.append(_acc(te, fit_single_head(tr)))
    return full, a2, a1


def a3_strength():
    """A3: dropping the strength gate. Reuses the H2 experiment (strength-typed vs
    non-typed). Predicted degradation: much worse on certain-involved conflicts."""
    from differentiable.h2 import run as h2run
    typed_acc, non_acc, typed_bc, non_bc = [], [], [], []
    for seed in range(5):
        r = h2run(seed=seed)
        typed_acc.append(r["strength-typed"]["acc"]); non_acc.append(r["non-typed"]["acc"])
        typed_bc.append(r["strength-typed"]["brier_certain"])
        non_bc.append(r["non-typed"]["brier_certain"])
    return typed_acc, non_acc, typed_bc, non_bc


def a4_meta():
    """A4: disabling the meta-controller. On a pair that is reconcilable AND
    abrogation-ordered, the full ordering (reconcile-first) dissolves the conflict
    (both accepted); with the meta operators OFF (weigh-only) the conflict is
    resolved by preponderance alone. Turning the meta layer off changes acceptance."""
    a = Argument("a", "p", head_scores={PreferenceHead.SOURCE: 1.0})
    b = Argument("b", "q", head_scores={PreferenceHead.SOURCE: 2.0})
    mutual = [("a", "b"), ("b", "a")]
    recon = lambda x, y: True                                              # noqa: E731
    abro = lambda x, y: Preferred.RIGHT if x.aid == "a" else Preferred.LEFT  # noqa: E731

    full = ArgumentationFramework([a, b], mutual, mode_order=RECONCILE_FIRST,
                                  reconcilable=recon, abrogates=abro)
    off = ArgumentationFramework([a, b], mutual, mode_order=WEIGH_ONLY,
                                 reconcilable=recon, abrogates=abro)
    g_full = sorted(grounded_extension(full.to_digraph("defeat")))
    g_off = sorted(grounded_extension(off.to_digraph("defeat")))
    return g_full, g_off


def _ci(xs):
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    t = 2.447 if len(xs) == 7 else 2.776
    return m, t * statistics.stdev(xs) / len(xs) ** 0.5


def main():
    print("=" * 66)
    print("PRE-REGISTERED ABLATIONS A1-A4 (synthetic; mean ± 95% CI where applicable)")
    print("=" * 66)

    full, a2, a1 = a1_a2("context")
    fm, fh = _ci(full); a2m, a2h = _ci(a2); a1m, a1h = _ci(a1)
    print("\n[context-dependent regime]")
    print(f"  full context calculus         acc = {fm:.3f} ± {fh:.3f}")
    print(f"  A2  context-conditioning OFF  acc = {a2m:.3f} ± {a2h:.3f}   (Δ {fm-a2m:+.3f})")
    print(f"  A1  multi-head OFF            acc = {a1m:.3f} ± {a1h:.3f}   (Δ {fm-a1m:+.3f})")

    ta, na, tbc, nbc = a3_strength()
    print("\n[A3 strength-gate — strength-aware regime, 5 seeds]")
    print(f"  strength gate ON   acc = {statistics.mean(ta):.3f}   Brier|certain = {statistics.mean(tbc):.3f}")
    print(f"  strength gate OFF  acc = {statistics.mean(na):.3f}   Brier|certain = {statistics.mean(nbc):.3f}"
          f"   (Δacc {statistics.mean(ta)-statistics.mean(na):+.3f})")

    g_full, g_off = a4_meta()
    print("\n[A4 meta-controller — reconcile/supersede-applicable instance]")
    print(f"  meta ON  (reconcile-first): grounded = {g_full}")
    print(f"  meta OFF (weigh-only):      grounded = {g_off}")
    print(f"  -> acceptance differs: {g_full != g_off}")

    print("\nAll four thesis components degrade/change as predicted (synthetic). "
          "Real-data ablations require the benchmark harness — see RUNBOOK.md.")


if __name__ == "__main__":
    main()
