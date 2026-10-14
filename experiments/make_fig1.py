"""Figure 1 (vector): FULCRUM on the three-argument witness, four aligned stages.
(1) criterion scores, (2) MAX-GAP table, (3) defeat cycle, (4) no preorder realises it.
Layout is done in POINTS (each panel's data coordinates are points), all text >= 8 pt,
criteria distinguished by colour AND hatching (grayscale-safe).
Output: output/figures/fig_overview.pdf
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = "output/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.6,
})
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
COL = [BLUE, ORANGE, GREEN]
HATCH = ["", "////", "...."]
CRIT = ["src", "spec", "mod"]
S = {"a": (0, 1, 2), "b": (1, 2, 0), "c": (2, 0, 1)}
FS = 8.0
FSH = 8.5

W = [86, 130, 68, 112]          # panel widths in points (sum 396 = 5.5in)
H = 142                         # panel height in points
TOP = 18                        # heading band
fig = plt.figure(figsize=(5.5, (H + TOP + 6) / 72))
gs = fig.add_gridspec(1, 4, width_ratios=W, wspace=0.0, left=0, right=1,
                      top=1 - 4 / (H + TOP + 6), bottom=2 / (H + TOP + 6))
axes = []
for i in range(4):
    ax = fig.add_subplot(gs[i]); ax.axis("off")
    ax.set_xlim(0, W[i]); ax.set_ylim(0, H + TOP); ax.set_aspect("equal")
    axes.append(ax)


def heading(ax, txt):
    ax.text(2, H + TOP - 4, txt, fontsize=FSH, weight="bold", va="top")


def rect(ax, x, y, w, h, h_):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=COL[h_], edgecolor="black", lw=0.5, hatch=HATCH[h_]))


# ------------------------------------------------------------ (1) criterion scores
ax = axes[0]; heading(ax, "(1) Scores")
unit, bh = 13, 7                      # points per score unit, bar height
for r, name in enumerate("abc"):
    ytop = H - 4 - r * 40
    ax.text(6, ytop - 11, f"${name}$", fontsize=FS + 2, ha="center", va="center")
    for h in range(3):
        v = S[name][h]; yb = ytop - h * (bh + 1.5) - bh
        rect(ax, 12, yb, max(v * unit, 1.2), bh, h)
        ax.text(12 + max(v * unit, 1.2) + 2.5, yb + bh / 2, str(v), fontsize=FS, va="center")
    ax.text(52, ytop - 11, f"$({S[name][0]},{S[name][1]},{S[name][2]})$", fontsize=FS, va="center")
for h in range(3):
    rect(ax, 4 + h * 28, 4, 8, 7, h)
    ax.text(14 + h * 28, 7.5, CRIT[h], fontsize=FS, va="center")

# ------------------------------------------------------------ (2) MAX-GAP table
ax = axes[1]; heading(ax, "(2) MAX-GAP")
cx = [11, 37, 59, 82, 103, 122]
hdr = ["pair", "src", "spec", "mod", r"$h{\star}$", "verdict"]
y0 = H - 8
for x, t in zip(cx, hdr):
    ax.text(x, y0, t, fontsize=FS, ha="center", va="center", weight="bold")
ax.plot([2, W[1] - 2], [y0 - 8, y0 - 8], color="black", lw=0.6)
rows = [("(a,b)", (1, 1, 2), 2, r"$a\!\succ\! b$"), ("(b,c)", (1, 2, 1), 1, r"$b\!\succ\! c$"),
        ("(c,a)", (2, 1, 1), 0, r"$c\!\succ\! a$")]
for i, (pair, gaps, hs, verdict) in enumerate(rows):
    y = y0 - 22 - i * 24
    ax.text(cx[0], y, f"${pair}$", fontsize=FS, ha="center", va="center")
    for h in range(3):
        if h == hs:
            ax.add_patch(FancyBboxPatch((cx[1 + h] - 8, y - 9), 16, 18, boxstyle="round,pad=0.5",
                                        facecolor=COL[h], alpha=0.35, edgecolor="black", lw=0.6))
            ax.text(cx[1 + h], y, str(gaps[h]), fontsize=FS + 1, ha="center", va="center", weight="bold")
        else:
            ax.text(cx[1 + h], y, str(gaps[h]), fontsize=FS, ha="center", va="center")
    ax.text(cx[4], y, CRIT[hs], fontsize=FS, ha="center", va="center", color=COL[hs], weight="bold")
    ax.text(cx[5], y, verdict, fontsize=FS + 0.5, ha="center", va="center")
ax.text(3, 20, r"shaded: largest gap $h{\star}$ decides;", fontsize=FS, va="center")
ax.text(3, 8, r"higher score on $h{\star}$ wins", fontsize=FS, va="center")

# ------------------------------------------------------------ (3) defeat cycle
ax = axes[2]; heading(ax, "(3) Cycle")
cxm, cym, R = W[2] / 2, H / 2 + 6, 25
P = {"a": (cxm, cym + R), "b": (cxm + R * 0.87, cym - R * 0.5), "c": (cxm - R * 0.87, cym - R * 0.5)}
edges = [("a", "b", 2, (cxm + R * 0.95, cym + R * 0.45)), ("b", "c", 1, (cxm, cym - R * 0.85)),
         ("c", "a", 0, (cxm - R * 0.95, cym + R * 0.45))]
for u, v, h, lp in edges:
    ax.add_patch(FancyArrowPatch(P[u], P[v], arrowstyle="-|>", mutation_scale=9, lw=1.4,
                                 color="black", shrinkA=10, shrinkB=10))
    ax.text(lp[0], lp[1], CRIT[h], fontsize=FS, ha="center", va="center", color=COL[h], style="italic")
for k, (x, y) in P.items():
    ax.add_patch(Circle((x, y), 9, facecolor="white", edgecolor="black", lw=0.9))
    ax.text(x, y, f"${k}$", fontsize=FS + 1.5, ha="center", va="center")
ax.text(cxm, 10, r"$a \succ b \succ c \succ a$", fontsize=FS + 0.5, ha="center", va="center")

# ------------------------------------------------------------ (4) no preorder
ax = axes[3]; heading(ax, "(4) No single ordering")
lines = [(H - 7, "the strict part of a"), (H - 19, "preorder is transitive"),
         (H - 31, "(Lemma 1):"), (H - 44, r"$a\succ b\succ c \Rightarrow a\succ c$,"),
         (H - 57, r"so $c\succ a$ is impossible.")]
for y, t in lines:
    ax.text(3, y, t, fontsize=FS, va="center")
ax.add_patch(FancyBboxPatch((3, 3), W[3] - 8, 66, boxstyle="round,pad=1", facecolor="0.93",
                            edgecolor="black", lw=0.6))
box = [(60, "Semantics (Thm 2):"), (47, r"grounded $=\emptyset$,"),
       (34, r"preferred $=\{\emptyset\}$,"), (21, "no stable extension"), (8, "(credulous separation)")]
for y, t in box:
    ax.text(6, y, t, fontsize=FS, va="center", weight="bold" if y == 60 else "normal")

fig.savefig(f"{OUT}/fig_overview.pdf")
fig.savefig(f"{OUT}/fig_overview_preview.png", dpi=220)
print("wrote", OUT)
