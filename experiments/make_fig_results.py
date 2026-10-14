"""Figure 2 (two large panels) for the paper, from stored results.
(a) World C: accuracy vs criteria correlation r for the key models, with the exact
    transitive ceiling; (b) World M: accuracy vs training-set fraction (sample efficiency
    where FULCRUM is well specified), with the ceiling.
Output: output/figures/fig_results.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.metrics import mean_ci

RES = "experiments/results"
OUT = "output/figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42,
})
STYLE = {
    "BT-linear": dict(color="#D55E00", marker="s", ls="--", label="BT (linear utility), transitive"),
    "BT-MLP": dict(color="#CC79A7", marker="v", ls="--", label="BT (MLP utility), transitive"),
    "pair-MLP (matched)": dict(color="#999999", marker="^", ls="-.", label="pair-MLP (matched)"),
    "pair-MLP (large)": dict(color="#000000", marker="D", ls=":", label="pair-MLP (large)"),
    "mixture-of-utilities": dict(color="#56B4E9", marker="P", ls="-.", label="mixture of utilities"),
    "FULCRUM-hard": dict(color="#009E73", marker="o", ls="-", label="FULCRUM-hard"),
    "FULCRUM-soft": dict(color="#0072B2", marker="o", ls="-", label="FULCRUM-soft"),
}


def load(tag):
    p = f"{RES}/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def series(res, model, key="acc"):
    xs = [res["per_seed"][s][model][key] for s in res["per_seed"]]
    return mean_ci(xs)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45))
    # (a)
    ax = axes[0]
    rs = [1.0, 0.7, 0.4, 0.2, 0.0]
    runs = {r: load(f"synC_r{r:.1f}") for r in rs}
    rs = [r for r in rs if runs[r] is not None]
    for mname, st in STYLE.items():
        ys, hs = zip(*[series(runs[r], mname) for r in rs])
        ax.errorbar(rs, ys, yerr=hs, ms=4, lw=1.3, capsize=2, elinewidth=0.7, markeredgecolor="black",
                    markeredgewidth=0.5, **st)
    ceil = [mean_ci([runs[r]["info"][s]["transitive_ceiling"] for s in runs[r]["info"]])[0] for r in rs]
    bt = [series(runs[r], "BT-linear")[0] for r in rs]
    ax.fill_between(rs, ceil, [1.0] * len(rs), facecolor="none", edgecolor="0.4", hatch="////", linewidth=0,
                    label="unreachable by any single ordering")
    ax.fill_between(rs, bt, ceil, color="#F6DAAF", alpha=0.6, linewidth=0,
                    label="gap of fitted BT to the ceiling")
    ax.plot(rs, ceil, color="black", lw=1.2, label="hindsight ceiling (exact, Thm. 5)")
    ax.set_xlim(1.05, -0.05); ax.set_ylim(0.78, 1.005)
    ax.set_xlabel(r"criteria correlation $r$ (conflict increases $\rightarrow$)")
    ax.set_ylabel("held-out pairwise accuracy")
    ax.set_title("(a) World C: majority-generated", fontsize=9)
    # (b)
    ax = axes[1]
    fr = [(0.1, "synM_noise0.1_frac0.10"), (0.25, "synM_noise0.1_frac0.25"), (1.0, "synM_noise0.1")]
    fr = [(f, load(t)) for f, t in fr]
    fr = [(f, r) for f, r in fr if r is not None]
    xs = [f for f, _ in fr]
    if fr:
        for mname, st in STYLE.items():
            ys, hs = zip(*[series(r, mname) for _, r in fr])
            ax.errorbar(xs, ys, yerr=hs, ms=4, lw=1.3, capsize=2, elinewidth=0.7, markeredgecolor="black",
                        markeredgewidth=0.5, **st)
        ceil = [mean_ci([r["info"][s]["transitive_ceiling"] for s in r["info"]])[0] for _, r in fr]
        ax.plot(xs, ceil, color="black", lw=1.2)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{int(f * 100)}%" for f in xs])
    else:
        ax.text(0.5, 0.5, "World M results pending", ha="center", transform=ax.transAxes)
    ax.set_xlabel("fraction of training pairs")
    ax.set_title("(b) World M: gap-generated", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig_results.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/fig_results_preview.png", dpi=200, bbox_inches="tight")
    print("wrote fig_results")


if __name__ == "__main__":
    main()
