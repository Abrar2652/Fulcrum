"""Figure: paired per-seed accuracy gain over BT (linear utility) on the three empirical
datasets, with 95% paired t-intervals, plus the hindsight-ceiling row of Theorem 5 computed
pairwise on the seeds where the ceiling solver was run (seeds 0-2). Reads the stored JSON
result files only. Output: output/figures/fig_wins.pdf"""
from __future__ import annotations
import json, math, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "experiments/results"; OUT = "output/figures"
T = {9: 2.262, 2: 4.303}
AMBER, CRIMSON, BROWN = "#E7961B", "#A6123A", "#8C6D4F"

ROWS = [("FULCRUM-soft", "FULCRUM-soft", "gate"), ("FULCRUM-hard", "FULCRUM-hard", "gate"),
        ("pair-MLP (large)", "pair-MLP (large)", "other"), ("pair-MLP (matched)", "pair-MLP (matched)", "other"),
        ("blade-chest", "blade-chest (item emb.)", "other"), ("antisym-bilinear", "antisym. bilinear", "other"),
        ("mixture-of-utilities", "mixture of utilities", "other"), ("random-gate", "random gate", "other"),
        ("context-BT (sum ctx)", "context-BT (sum ctx.)", "other"), ("linear-in-d", "linear in $d$", "other"),
        ("lexicographic", "lexicographic", "trans"), ("BT-MLP", "BT (MLP utility)", "trans")]
COL = {"gate": CRIMSON, "other": BROWN, "trans": AMBER}
MK = {"gate": "o", "other": "D", "trans": "v"}


def load(tag):
    return json.load(open(f"{RES}/{tag}.json"))


def gaps(ps, m):
    d = [100 * (ps[s][m]["acc"] - ps[s]["BT-linear"]["acc"]) for s in ps if m in ps[s]]
    if not d:
        return None
    return st.mean(d), (T[len(d) - 1] * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0)


def ceiling_row(tag, ceil_tag=None):
    d = load(tag); ps = d["per_seed"]
    src = load(ceil_tag)["extra"] if ceil_tag else d["extra"]
    lo, hi = [], []
    for s in ("0", "1", "2"):
        c = src[s].get("test_graph_ceilings") if isinstance(src.get(s), dict) else None
        if not c:
            continue
        bt = ps[s]["BT-linear"]["acc"]
        lo.append(100 * (c["lower"] - bt)); hi.append(100 * (c["upper"] - bt))
    return (st.mean(lo), st.mean(hi), len(lo)) if lo else None


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("pokemon_full", None, "Pok\u00e9mon"), ("sushi_b", "sushi_b_ceilings", "Sushi-B"),
              ("helpsteer2_pref", None, "HelpSteer2-Pref.")]
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 3.9), sharey=True)
    labels = ["hindsight ceiling\n(best single ordering)"] + [r[1] for r in ROWS]
    ys = [len(ROWS) + 0.9] + list(range(len(ROWS)))[::-1]
    for ax, (tag, ctag, title) in zip(axs, panels):
        d = load(tag); ps = d["per_seed"]
        ax.axvline(0, color="0.5", lw=0.7)
        # ceiling row
        y0 = ys[0]
        if tag == "helpsteer2_pref":
            bt = st.mean(ps[s]["BT-linear"]["acc"] for s in ps)
            ax.text(0.3, y0, f"trivial (1.000; +{100*(1-bt):.1f})", fontsize=7, va="center")
        else:
            cr = ceiling_row(tag, ctag)
            if cr:
                lo, hi, n = cr
                if hi - lo > 0.05:
                    ax.barh(y0, hi - lo, left=lo, height=0.55, color="white", edgecolor="black", hatch="////", lw=0.8)
                else:
                    ax.plot([lo], [y0], marker="s", color="black", ms=6)
                ax.text(max(hi, lo) + 0.4, y0, f"seeds 0\u2013{n-1}", fontsize=6.5, va="center")
        for y, (key, lab, fam) in zip(ys[1:], ROWS):
            g = gaps(ps, key)
            if g is None:
                continue
            m, h = g
            filled = abs(m) > h
            ax.errorbar([m], [y], xerr=[[h], [h]], fmt=MK[fam], color=COL[fam], ms=6,
                        markerfacecolor=COL[fam] if filled else "white", markeredgecolor="black",
                        ecolor="black", elinewidth=0.9, capsize=2.5)
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1.5, 8.5); ax.set_ylim(-0.7, len(ROWS) + 1.6); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="0.9", lw=0.6)
    fig.supxlabel("paired gain over BT (linear utility), accuracy points; markers: mean $\\pm$ 95% t-interval (10 seeds)", fontsize=8, y=0.06)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hdl = [Line2D([], [], marker="v", color=AMBER, markeredgecolor="black", ls="", ms=6, label="transitive"),
           Line2D([], [], marker="o", color=CRIMSON, markeredgecolor="black", ls="", ms=6, label="gated (FULCRUM)"),
           Line2D([], [], marker="D", color=BROWN, markeredgecolor="black", ls="", ms=6, label="other non-transitive"),
           Line2D([], [], marker="o", color="white", markeredgecolor="black", ls="", ms=6, label="interval includes 0"),
           Patch(facecolor="white", edgecolor="black", hatch="////", label="solver bracket (not a CI)")]
    fig.legend(handles=hdl, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{OUT}/fig_wins.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_wins")


if __name__ == "__main__":
    main()
