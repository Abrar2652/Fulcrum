"""Re-run conflict_experiment across more seeds and report 95% CIs + paired
significance for the single-ordering vs fulcrum gap (addresses reviewer C1)."""
import math

from experiments.conflict_experiment import run

SEEDS = list(range(10))


def ci95(xs):
    n = len(xs); m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, 1.96 * math.sqrt(var / n)


print(f"{'rho':>5} {'intransit':>10} {'single(95%CI)':>18} {'fulcrum(95%CI)':>18} "
      f"{'delta(95%CI)':>18} {'sig?':>5}")
for rho in (1.0, 0.7, 0.4, 0.2, 0.0):
    rows = [run(rho, seed=s) for s in SEEDS]
    itr = sum(r["intransitivity"] for r in rows) / len(rows)
    so = [r["single_ordering"] for r in rows]
    fu = [r["fulcrum"] for r in rows]
    dl = [f - s for f, s in zip(fu, so)]  # paired per-seed delta
    som, soc = ci95(so); fum, fuc = ci95(fu); dlm, dlc = ci95(dl)
    sig = "yes" if (dlm - dlc) > 0 else "no"  # 95% CI of paired delta excludes 0
    print(f"{rho:>5.2f} {itr:>10.3f} {som:>10.3f}+-{soc:<5.3f} {fum:>10.3f}+-{fuc:<5.3f} "
          f"{dlm:>+10.3f}+-{dlc:<5.3f} {sig:>5}")
