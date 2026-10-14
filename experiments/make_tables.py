"""Generate every number and table in the paper from the stored result files.

Outputs (output/):
  numbers.tex             \newcommand macros for numbers quoted in prose
  tab_main.tex            main results table (accuracy, key models, all datasets, ceilings)
  tab_synthetic_full.tex  full synthetic table (all models, r sweep) with paired gaps
  tab_observational.tex   full observational table (all models, 3 datasets, NLL/Brier/ECE)
  tab_ablations.tex       ablation table (gates, shuffle, noise, training size, world M)
  tab_faith.tex           attribution faithfulness diagnostics
  tab_ceilings.tex        ceilings vs achieved accuracy (representational vs estimation gap)
  tab_frameworks.tex, tab_aggregation.tex, tab_ablations_discrete.tex, tab_counts.tex
  fig_results.pdf         two-panel results figure (via make_fig_results)
Usage: PYTHONPATH=. python experiments/make_tables.py
"""
from __future__ import annotations

import json
import math
import os

from experiments.metrics import mean_ci, paired_bootstrap_interval, paired_t_interval

RES = "experiments/results"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

SHORT = {
    "BT-linear": "BT (linear utility)", "BT-MLP": "BT (MLP utility)", "lexicographic": "lexicographic",
    "Condorcet": "Condorcet rule", "MAX-GAP rule": "\\maxgap{} rule (fixed)", "linear-in-d": "linear in $d$",
    "context-BT (sum ctx)": "context-BT (sum ctx.)", "random-gate": "random gate",
    "mixture-of-utilities": "mixture of utilities", "antisym-bilinear": "antisym.\\ bilinear",
    "pair-MLP (matched)": "pair-MLP (matched)", "pair-MLP (large)": "pair-MLP (large)",
    "blade-chest": "blade-chest (item emb.)", "FULCRUM-hard": "\\fulcrum-hard",
    "FULCRUM-soft": "\\fulcrum-soft", "FULCRUM-soft (fixed beta)": "\\fulcrum-soft, fixed $\\beta$",
    "FULCRUM-soft (no gate bias)": "\\fulcrum-soft, no bias",
}
TRANSITIVE = {"BT-linear", "BT-MLP", "lexicographic"}
ORDER = ["BT-linear", "BT-MLP", "lexicographic", "Condorcet", "MAX-GAP rule", "linear-in-d",
         "context-BT (sum ctx)", "random-gate", "mixture-of-utilities", "antisym-bilinear",
         "blade-chest", "pair-MLP (matched)", "pair-MLP (large)", "FULCRUM-hard", "FULCRUM-soft",
         "FULCRUM-soft (fixed beta)", "FULCRUM-soft (no gate bias)"]


def load(tag):
    p = f"{RES}/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def per_seed(res, model, key):
    return [res["per_seed"][s][model].get(key, float("nan")) for s in res["per_seed"]]


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def pm(xs, d=3):
    m, h = mean_ci(xs)
    return f"{m:.{d}f}\\,{{\\scriptsize$\\pm${h:.{d}f}}}"


def gap(xs, ref, d=3, bold_if_excludes_zero=True):
    diffs = [x - r for x, r in zip(xs, ref)]
    m, h = paired_t_interval(diffs)
    s = f"{m:+.{d}f}\\,{{\\scriptsize$\\pm${h:.{d}f}}}"
    if bold_if_excludes_zero and (m - h > 0):
        s = f"\\textbf{{{s}}}"
    return s


def escape(s):
    return s.replace("_", "\\_")


# ------------------------------------------------------------------ numbers
import re as _re

def write_numbers(macros: dict):
    """Write the macros; any macro name used in the .tex sources but not produced (because a
    run has not finished) is defined as a red TBD so the paper compiles and the gap is visible."""
    used = set()
    for fn in os.listdir(OUT):
        if fn.endswith(".tex") and fn not in ("numbers.tex", "math_commands.tex"):
            used |= set(_re.findall(r"\\((?:acc|gap|ceil|rep|est|flip|suff|orient|onCycle|offCycle|pairCeil|hs|n|netflix|sushi|tri|fthree)[A-Za-z]+)", open(f"{OUT}/{fn}").read()))
    skip = {"ne", "newcommand", "newpage", "newtheorem", "noindent", "nMeanViol", "newline", "not", "neg", "nabla", "nu", "null", "hspace", "hline", "hfill", "hphantom", "estimate"}
    with open(f"{OUT}/numbers.tex", "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        for k in sorted(used - set(macros)):
            if k in skip or k.startswith("newcommand"):
                continue
            f.write(f"\\newcommand{{\\{k}}}{{\\textcolor{{red}}{{TBD}}}}\n")


PLACEHOLDER_TABLES = {"tab_main": "tab:main", "tab_synthetic_full": "tab:synfull", "tab_observational": "tab:obsfull", "tab_observational2": "tab:obsfull2",
                      "tab_ablations": "tab:ablations", "tab_faith": "tab:faith", "tab_worldM": "tab:worldM",
                      "tab_pokemon_slices": "tab:pkslices", "tab_structure": "tab:structure", "tab_params": "tab:params"}


def main():
    macros = {}
    for fn, lab in PLACEHOLDER_TABLES.items():   # placeholders so the paper compiles before every run has finished
        open(f"{OUT}/{fn}.tex", "w").write(f"\\begin{{table}}[t]\\centering\\small\\caption{{Results pending.}}\\label{{{lab}}}\\textcolor{{red}}{{TBD}}\\end{{table}}\n")
    ex = json.load(open(f"{RES}/exhaustive_checks.json"))
    macros["nMeanViol"] = ex["aggregation"]["random_sets"]["mean"]["violations"]
    cnt = json.load(open(f"{RES}/intransitivity_counts.json"))
    macros["netflixProfiles"] = cnt["netflix"]["profiles"]
    macros["sushiBfrac"] = f"{100 * cnt['sushi_b_majority']['fraction']:.1f}"
    macros["sushiAttrFrac"] = f"{100 * cnt['sushi_attribute_majority']['fraction']:.2f}"

    # ---------------------------------------------------------------- synthetic
    syn = {r: load(f"synC_r{r:.1f}") for r in (1.0, 0.7, 0.4, 0.2, 0.0)}
    synM = load("synM_noise0.1")
    key_models = ["BT-linear", "BT-MLP", "lexicographic", "mixture-of-utilities", "antisym-bilinear",
                  "pair-MLP (matched)", "pair-MLP (large)", "FULCRUM-hard", "FULCRUM-soft"]
    # full synthetic table
    with open(f"{OUT}/tab_synthetic_full.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{World C (Condorcet oracle, $m=5$, 60 items per split, 10 seeds): held-out pairwise accuracy (mean\\,$\\pm$\\,t-based 95\\% half-width) as the criteria correlation $r$ decreases (conflict increases). T = transitive model. The Condorcet rule is the oracle itself. Ceilings are exact Kemeny/Slater optima on each test tournament (Theorem~\\ref{thm:ceiling}); $f_3$ is the cyclic-triple fraction and $1-f_3/3$ the closed-form bound.}\n\\label{tab:synfull}\n")
        cols = "l" + "c" * 5
        f.write(f"\\begin{{tabular}}{{{cols}}}\\toprule\nmodel & $r=1.0$ & $r=0.7$ & $r=0.4$ & $r=0.2$ & $r=0.0$\\\\\\midrule\n")
        for mname in ORDER:
            if any(syn[r] is None or mname not in syn[r]["per_seed"]["0"] for r in syn):
                continue
            tag = "$^{\\mathrm{T}}$" if mname in TRANSITIVE else ""
            row = [SHORT[mname] + tag]
            for r in (1.0, 0.7, 0.4, 0.2, 0.0):
                row.append(pm(per_seed(syn[r], mname, "acc")))
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\midrule\n")
        for label, key in (("transitive ceiling (exact)", "transitive_ceiling"), ("$f_3$ (cyclic triples)", "f3"),
                           ("$1-f_3/3$", "triangle_bound")):
            row = [label]
            for r in (1.0, 0.7, 0.4, 0.2, 0.0):
                xs = [syn[r]["info"][s][key] for s in syn[r]["info"]]
                row.append(pm(xs))
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    # macros from synthetic r=0
    s0 = syn[0.0]
    ref = per_seed(s0, "BT-linear", "acc")
    for mname, mk in (("FULCRUM-soft", "gapSoftRzero"), ("FULCRUM-hard", "gapHardRzero"),
                      ("pair-MLP (large)", "gapMLPlargeRzero"), ("pair-MLP (matched)", "gapMLPmatchedRzero"),
                      ("mixture-of-utilities", "gapMixRzero"), ("BT-MLP", "gapBTMLPRzero")):
        m, h = paired_t_interval([x - r_ for x, r_ in zip(per_seed(s0, mname, "acc"), ref)])
        macros[mk] = f"{100 * m:+.1f}"; macros[mk + "CI"] = f"{100 * h:.1f}"
    for mname, mk in (("BT-linear", "accBTRzero"), ("FULCRUM-soft", "accSoftRzero"), ("FULCRUM-hard", "accHardRzero"),
                      ("pair-MLP (large)", "accMLPlargeRzero"), ("pair-MLP (matched)", "accMLPmatchedRzero"),
                      ("BT-MLP", "accBTMLPRzero"), ("mixture-of-utilities", "accMixRzero")):
        macros[mk] = f"{mean_ci(per_seed(s0, mname, 'acc'))[0]:.3f}"
    ceil0 = [s0["info"][s]["transitive_ceiling"] for s in s0["info"]]
    macros["ceilRzero"] = f"{mean_ci(ceil0)[0]:.3f}"
    macros["fthreeRzero"] = f"{mean_ci([s0['info'][s]['f3'] for s in s0['info']])[0]:.3f}"
    macros["triRzero"] = f"{mean_ci([s0['info'][s]['triangle_bound'] for s in s0['info']])[0]:.3f}"
    macros["estGapBTRzero"] = f"{mean_ci(ceil0)[0] - mean_ci(ref)[0]:.3f}"
    macros["repGapRzero"] = f"{1 - mean_ci(ceil0)[0]:.3f}"
    # world M
    if synM is not None:
        refM = per_seed(synM, "BT-linear", "acc")
        for mname, mk in (("FULCRUM-soft", "Soft"), ("FULCRUM-hard", "Hard"), ("pair-MLP (large)", "MLPlarge"),
                          ("pair-MLP (matched)", "MLPmatched"), ("mixture-of-utilities", "Mix"), ("BT-linear", "BT")):
            macros["accM" + mk] = f"{mean_ci(per_seed(synM, mname, 'acc'))[0]:.3f}"
            m, h = paired_t_interval([x - r_ for x, r_ in zip(per_seed(synM, mname, "acc"), refM)])
            macros["gapM" + mk] = f"{100 * m:+.1f}"; macros["gapM" + mk + "CI"] = f"{100 * h:.1f}"
        macros["ceilM"] = f"{mean_ci([synM['info'][s]['transitive_ceiling'] for s in synM['info']])[0]:.3f}"
        for frac, fk in ((0.1, "Ten"), (0.25, "Quarter")):
            fr = load(f"synM_noise0.1_frac{frac:.2f}")
            if fr is None:
                continue
            refF = per_seed(fr, "BT-linear", "acc")
            for mname, mk in (("FULCRUM-soft", "Soft"), ("FULCRUM-hard", "Hard"), ("pair-MLP (large)", "MLPlarge"), ("BT-linear", "BT")):
                macros["accMfrac" + fk + mk] = f"{mean_ci(per_seed(fr, mname, 'acc'))[0]:.3f}"
                m, h = paired_t_interval([x - r_ for x, r_ in zip(per_seed(fr, mname, "acc"), refF)])
                macros["gapMfrac" + fk + mk] = f"{100 * m:+.1f}"; macros["gapMfrac" + fk + mk + "CI"] = f"{100 * h:.1f}"
            macros["nTrainMfrac" + fk] = f"{fr['info']['0']['n_train']:,}"
        shM = load("synM_noise0.1_shuffle")
        if shM is not None:
            refS = per_seed(shM, "BT-linear", "acc")
            for mname, mk in (("FULCRUM-soft", "Soft"), ("pair-MLP (large)", "MLPlarge"), ("MAX-GAP rule", "Rule")):
                m, h = paired_t_interval([x - r_ for x, r_ in zip(per_seed(shM, mname, "acc"), refS)])
                macros["gapMShuffle" + mk] = f"{100 * m:+.1f}"; macros["gapMShuffle" + mk + "CI"] = f"{100 * h:.1f}"

    # ---------------------------------------------------------------- observational
    obs = {"pokemon_full": ("Pok\\'emon", load("pokemon_full")), "sushi_b": ("Sushi-B", load("sushi_b")),
           "helpsteer2_pref": ("HelpSteer2-Pref.", load("helpsteer2_pref"))}
    su_ceil = load("sushi_b_ceilings")          # brackets from ceilings_post.py override the in-run values
    if su_ceil is not None and obs["sushi_b"][1] is not None:
        obs["sushi_b"][1]["extra"] = su_ceil["extra"]
    for fname, mets, metnames in (("tab_observational", ("acc", "nll"), ("acc", "NLL")),
                                  ("tab_observational2", ("brier", "ece"), ("Brier", "ECE"))):
        with open(f"{OUT}/{fname}.tex", "w") as f:
            f.write("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n")
            if fname == "tab_observational":
                f.write("\\caption{Observational datasets (10 seeds): held-out accuracy and NLL (mean\\,$\\pm$\\,t-based 95\\% half-width; all models temperature-scaled on validation data). T = transitive. Blade-chest requires repeated items and is absent on HelpSteer2.}\n\\label{tab:obsfull}\n")
            else:
                f.write("\\caption{Observational datasets (10 seeds): Brier score and ECE (15 bins), mean\\,$\\pm$\\,t-based 95\\% half-width.}\n\\label{tab:obsfull2}\n")
            f.write("\\begin{tabular}{l" + "cc" * 3 + "}\\toprule\n & \\multicolumn{2}{c}{Pok\\'emon} & \\multicolumn{2}{c}{Sushi-B} & \\multicolumn{2}{c}{HelpSteer2-Pref.}\\\\\n")
            f.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nmodel & " + " & ".join(f"{a} & {b}" for a, b in [metnames] * 3) + "\\\\\\midrule\n")
            for mname in ORDER:
                if mname.startswith("FULCRUM-soft ("):
                    continue
                tag = "$^{\\mathrm{T}}$" if mname in TRANSITIVE else ""
                row = [SHORT[mname] + tag]
                any_ = False
                for tag_, (_, res) in obs.items():
                    if res is None or mname not in res["per_seed"][list(res["per_seed"])[0]]:
                        row += ["--", "--"]; continue
                    any_ = True
                    row += [pm(per_seed(res, mname, mets[0])), pm(per_seed(res, mname, mets[1]))]
                if any_:
                    f.write(" & ".join(row) + "\\\\\n")
            f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    # observational macros
    for tag_, (label, res) in obs.items():
        if res is None:
            continue
        short = {"pokemon_full": "Pk", "sushi_b": "Su", "helpsteer2_pref": "Hs"}[tag_]
        ref = per_seed(res, "BT-linear", "acc")
        for mname, mk in (("BT-linear", "BT"), ("FULCRUM-soft", "Soft"), ("FULCRUM-hard", "Hard"),
                          ("pair-MLP (large)", "MLPlarge"), ("pair-MLP (matched)", "MLPmatched"),
                          ("mixture-of-utilities", "Mix"), ("blade-chest", "BC"), ("lexicographic", "Lex"),
                          ("linear-in-d", "LinD"), ("BT-MLP", "BTMLP")):
            if mname not in res["per_seed"]["0"]:
                continue
            macros["acc" + short + mk] = f"{mean_ci(per_seed(res, mname, 'acc'))[0]:.3f}"
            m, h = paired_t_interval([x - r_ for x, r_ in zip(per_seed(res, mname, "acc"), ref)])
            macros["gap" + short + mk] = f"{100 * m:+.1f}"; macros["gap" + short + mk + "CI"] = f"{100 * h:.1f}"
        info0 = res["info"]["0"]
        macros["nTrain" + short] = f"{info0['n_train']:,}"; macros["nTest" + short] = f"{info0['n_test']:,}"
        if "extra" in res:
            ce = [res["extra"][s].get("test_graph_ceilings") for s in res["extra"] if res["extra"][s]]
            ce = [c for c in ce if c]
            if ce:
                if all(c.get("status") == "optimal" and "transitive_ceiling" in c for c in ce):
                    macros["ceil" + short] = f"{sum(c['transitive_ceiling'] for c in ce) / len(ce):.3f}"
                else:
                    los = [c.get("lower") for c in ce if c.get("lower") is not None]
                    his = [c.get("upper", c.get("transitive_ceiling")) for c in ce]
                    his = [h for h in his if h is not None]
                    if los:
                        macros["ceilLo" + short] = f"{sum(los) / len(los):.3f}"
                    if his:
                        macros["ceilHi" + short] = f"{sum(his) / len(his):.3f}"
                macros["pairCeil" + short] = f"{sum(c['pair_ceiling'] for c in ce) / len(ce):.3f}"
    # Pokemon slices
    pk = obs["pokemon_full"][1]
    if pk is not None:
        with open(f"{OUT}/tab_pokemon_slices.tex", "w") as f:
            f.write("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n")
            f.write("\\caption{Pok\\'emon: accuracy on prespecified test slices (10 seeds; mean\\,$\\pm$\\,t-based 95\\% half-width). Slices: all test battles; cross-type battles (non-zero type-advantage term); battles in the lowest 30/20/10\\% of $|$total base-stat difference$|$ (a model-free closeness measure).}\n\\label{tab:pkslices}\n")
            f.write("\\begin{tabular}{lccccc}\\toprule\nmodel & all & cross-type & closest 30\\% & closest 20\\% & closest 10\\%\\\\\\midrule\n")
            for mname in ORDER:
                if mname not in pk["per_seed"]["0"] or mname.startswith("FULCRUM-soft ("):
                    continue
                tag = "$^{\\mathrm{T}}$" if mname in TRANSITIVE else ""
                row = [SHORT[mname] + tag] + [pm(per_seed(pk, mname, f"acc_slice_{s}")) for s in ("all", "cross_type", "close30", "close20", "close10")]
                f.write(" & ".join(row) + "\\\\\n")
            f.write("\\bottomrule\\end{tabular}\\end{table}\n")
        for mname, mk in (("BT-linear", "BT"), ("FULCRUM-soft", "Soft"), ("pair-MLP (matched)", "MLPmatched"), ("linear-in-d", "LinD")):
            macros["accPkCloseTen" + mk] = f"{mean_ci(per_seed(pk, mname, 'acc_slice_close10'))[0]:.3f}"
    # HelpSteer info
    hs = obs["helpsteer2_pref"][1]
    if hs is not None:
        i0 = hs["info"]["0"]
        macros["hsTies"] = f"{i0['ties']:,}"; macros["hsUnmatched"] = f"{i0['unmatched']:,}"
        macros["hsPrefRows"] = f"{i0['pref_rows']:,}"

    # ---------------------------------------------------------------- main table
    with open(f"{OUT}/tab_main.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3.2pt}\n")
        f.write("\\caption{Held-out pairwise accuracy (10 seeds; mean\\,$\\pm$\\,t-based 95\\% half-width) of transitive (T) and non-transitive models on the two controlled worlds and three observational datasets. Ceilings: exact best single ordering on each test comparison set (Theorem~\\ref{thm:ceiling}); for Pok\\'emon a bracket (achievable ordering, LP bound) over three seeds. Full tables with NLL, Brier and ECE, paired gaps and every baseline are in Appendix~\\ref{app:exp}.}\n\\label{tab:main}\n")
        f.write("\\begin{tabular}{lccccc}\\toprule\nmodel & World C, $r{=}0$ & World M & Pok\\'emon & Sushi-B & HelpSteer2-Pref.\\\\\\midrule\n")
        cols = [syn[0.0], synM, obs["pokemon_full"][1], obs["sushi_b"][1], obs["helpsteer2_pref"][1]]
        for mname in key_models + ["blade-chest"]:
            tag = "$^{\\mathrm{T}}$" if mname in TRANSITIVE else ""
            row = [SHORT[mname] + tag]
            for res in cols:
                if res is None or mname not in res["per_seed"]["0"]:
                    row.append("--")
                else:
                    row.append(pm(per_seed(res, mname, "acc")))
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\midrule\n")
        row = ["transitive ceiling"]
        for res, key in zip(cols, ("synC", "synM", "pk", "su", "hs")):
            if res is None:
                row.append("--"); continue
            if key in ("synC", "synM"):
                row.append(pm([res["info"][s]["transitive_ceiling"] for s in res["info"]]))
            elif key == "pk":
                lo = macros.get("ceilLoPk"); hi = macros.get("ceilHiPk")
                row.append(f"[{lo}, {hi}]" if (lo or hi) else "--")
            elif key == "su":
                if "ceilSu" in macros:
                    row.append(macros["ceilSu"])
                elif "ceilHiSu" in macros:
                    lo, hi = macros.get("ceilLoSu", "?"), macros["ceilHiSu"]
                    row.append(hi if lo == hi else f"[{lo}, {hi}]")
                else:
                    row.append("--")
            else:
                row.append("n/a (matching)")
        f.write(" & ".join(row) + "\\\\\n")
        row = ["pair ceiling"]
        for res, key in zip(cols, ("synC", "synM", "pk", "su", "hs")):
            row.append("1.000" if key in ("synC", "synM") else macros.get("pairCeil" + {"pk": "Pk", "su": "Su", "hs": "Hs"}[key], "n/a" if key == "hs" else "--"))
        f.write(" & ".join(row) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    # ---------------------------------------------------------------- ablations
    abl_rows = []
    def add(label, res, mname, ref_res=None, ref_m="BT-linear"):
        if res is None or mname not in res["per_seed"]["0"]:
            return
        xs = per_seed(res, mname, "acc")
        rr = per_seed(ref_res or res, ref_m, "acc")
        abl_rows.append((label, pm(xs), gap(xs, rr)))
    s0 = syn[0.0]
    add("\\fulcrum-soft (full)", s0, "FULCRUM-soft")
    add("\\quad hard gate", s0, "FULCRUM-hard")
    add("\\quad soft gate, fixed $\\beta$", s0, "FULCRUM-soft (fixed beta)")
    add("\\quad soft gate, no gate bias", s0, "FULCRUM-soft (no gate bias)")
    add("\\quad random gate (context conditioning off)", s0, "random-gate")
    add("\\quad sum context instead of gap", s0, "context-BT (sum ctx)")
    add("\\quad gate and weights fixed (\\maxgap{} rule)", s0, "MAX-GAP rule")
    add("\\quad single criterion (lexicographic)", s0, "lexicographic")
    add("\\quad shared weights (linear in $d$)", s0, "linear-in-d")
    sh = load("synC_r0.0_shuffle")
    add("\\quad criterion identity shuffled: \\fulcrum-soft", sh, "FULCRUM-soft")
    add("\\quad criterion identity shuffled: pair-MLP (large)", sh, "pair-MLP (large)")
    add("\\quad criterion identity shuffled: BT (linear)", sh, "BT-linear")
    shM = load("synM_noise0.1_shuffle"); baseM = load("synM_noise0.1")
    add("\\quad World M, unshuffled: \\fulcrum-soft", baseM, "FULCRUM-soft")
    add("\\quad World M, unshuffled: pair-MLP (large)", baseM, "pair-MLP (large)")
    add("\\quad World M, criterion identity shuffled: \\fulcrum-soft", shM, "FULCRUM-soft")
    add("\\quad World M, criterion identity shuffled: pair-MLP (large)", shM, "pair-MLP (large)")
    for eps in (0.1, 0.2, 0.3):
        add(f"\\quad label noise $\\varepsilon={eps}$: \\fulcrum-soft", load(f"synC_r0.0_noise{eps:.1f}"), "FULCRUM-soft")
    for frac in (0.1, 0.25, 0.5):
        add(f"\\quad training pairs {int(frac * 100)}\\%: \\fulcrum-soft", load(f"synC_r0.0_frac{frac:.2f}"), "FULCRUM-soft")
        add(f"\\quad training pairs {int(frac * 100)}\\%: pair-MLP (large)", load(f"synC_r0.0_frac{frac:.2f}"), "pair-MLP (large)")
    with open(f"{OUT}/tab_ablations.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Ablations (10 seeds): accuracy and paired gap to BT (linear utility) of the same run (t-based 95\\% half-width; bold when the interval excludes zero). Unless stated, World C at $r=0$; rows vary one ingredient at a time; shuffle, noise and training-size rows are separate runs with their own BT reference.}\n\\label{tab:ablations}\n")
        f.write("\\begin{tabular}{lcc}\\toprule\nvariant & accuracy & gap vs BT (paired)\\\\\\midrule\n")
        for label, a, g in abl_rows:
            f.write(f"{label} & {a} & {g}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    # sample efficiency / world M table
    rowsM = []
    for tag, label in (("synM_noise0.1_frac0.10", "10\\%"), ("synM_noise0.1_frac0.25", "25\\%"), ("synM_noise0.1", "100\\%")):
        res = load(tag)
        if res is None:
            continue
        rowsM.append((label, res))
    if not rowsM:
        open(f"{OUT}/tab_worldM.tex", "w").write("\\begin{table}[t]\\centering\\small\\caption{World M results pending.}\\label{tab:worldM}\\textcolor{red}{TBD}\\end{table}\n")
    with open(f"{OUT}/tab_worldM.tex", "w" if rowsM else "a") as f:
      if rowsM:
          f.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n")
          f.write("\\caption{World M (\\maxgap{} ground truth with hidden per-criterion scales, label noise $0.1$, 10 seeds): accuracy as the training set shrinks. Sample efficiency in the well-specified regime.}\n\\label{tab:worldM}\n")
          f.write("\\begin{tabular}{l" + "c" * len(rowsM) + "}\\toprule\nmodel & " + " & ".join(l for l, _ in rowsM) + "\\\\\\midrule\n")
          for mname in ["BT-linear", "BT-MLP", "mixture-of-utilities", "antisym-bilinear", "pair-MLP (matched)", "pair-MLP (large)", "FULCRUM-hard", "FULCRUM-soft"]:
              tag = "$^{\\mathrm{T}}$" if mname in TRANSITIVE else ""
              f.write(SHORT[mname] + tag + " & " + " & ".join(pm(per_seed(res, mname, "acc")) for _, res in rowsM) + "\\\\\n")
          f.write("\\midrule transitive ceiling & " + " & ".join(pm([res["info"][s]["transitive_ceiling"] for s in res["info"]]) for _, res in rowsM) + "\\\\\n")
          f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    # ---------------------------------------------------------------- faithfulness
    def faith_rows(res, mname, key="faithfulness"):
        out = {}
        for grp in ("attributed", "random_other", "least"):
            for met in ("flip_sign", "flip_delete", "sufficiency", "comprehensiveness"):
                xs = []
                for s in res["per_seed"]:
                    rec = res["per_seed"][s][mname]
                    fa = rec.get(key) or rec.get("faithfulness_occlusion") or rec.get("faithfulness") or {}
                    xs.append(fa.get(grp, {}).get(met, float("nan")))
                out[(grp, met)] = xs
        return out
    with open(f"{OUT}/tab_faith.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{Attribution faithfulness (World C, $r=0$, and Pok\\'emon; 10 seeds, mean over seeds). Interventions on the attributed criterion $h^\\star=\\arg\\max_h|\\kappa_h|$ versus a random other criterion (in parentheses): sign reversal and deletion report the fraction of verdicts that flip; sufficiency the agreement when only $h^\\star$ is kept; comprehensiveness the mean drop in confidence after deletion. For the pair-MLP the attributed criterion is the one whose deletion changes the logit most (occlusion), because it has no built-in contributions.}\n\\label{tab:faith}\n")
        f.write("\\begin{tabular}{llcccc}\\toprule\ndata & model & sign flip & deletion flip & sufficiency & comprehensiveness\\\\\\midrule\n")
        for dname, res in (("World C, $r{=}0$", syn[0.0]), ("Pok\\'emon", obs["pokemon_full"][1])):
            if res is None:
                continue
            for mname, key in (("FULCRUM-hard", "faithfulness"), ("FULCRUM-soft", "faithfulness"), ("linear-in-d", "faithfulness"), ("pair-MLP (large)", "faithfulness_occlusion")):
                if mname not in res["per_seed"]["0"]:
                    continue
                fr = faith_rows(res, mname, key)
                cells = []
                for met in ("flip_sign", "flip_delete", "sufficiency", "comprehensiveness"):
                    a = mean_ci(fr[("attributed", met)])[0]; r_ = mean_ci(fr[("random_other", met)])[0]
                    cells.append(f"{a:.2f} ({r_:.2f})")
                f.write(f"{dname} & {SHORT[mname]} & " + " & ".join(cells) + "\\\\\n")
                if dname.startswith("World") and mname in ("FULCRUM-soft", "FULCRUM-hard"):
                    mk = "Soft" if "soft" in mname else "Hard"
                    macros["flipSign" + mk] = f"{100 * mean_ci(fr[('attributed', 'flip_sign')])[0]:.0f}"
                    macros["flipSignRand" + mk] = f"{100 * mean_ci(fr[('random_other', 'flip_sign')])[0]:.0f}"
                    macros["suff" + mk] = f"{100 * mean_ci(fr[('attributed', 'sufficiency')])[0]:.0f}"
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    # ---------------------------------------------------------------- structure metrics (cycles)
    with open(f"{OUT}/tab_structure.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n")
        f.write("\\caption{Structure metrics (10 seeds, means): accuracy on test pairs that lie on a cyclic triple of the reference relation (on) vs.\\ the rest (off), and cycle-orientation accuracy (cyc.) (fraction of cyclic test triples on which all three predicted orientations are correct; a transitive model can never exceed 0). Reference relation: the oracle (World C) or the observed majority among test comparisons (Pok\\'emon, Sushi-B).}\n\\label{tab:structure}\n")
        f.write("\\begin{tabular}{l" + "ccc" * 3 + "}\\toprule\n & \\multicolumn{3}{c}{World C, $r{=}0$} & \\multicolumn{3}{c}{Pok\\'emon} & \\multicolumn{3}{c}{Sushi-B}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nmodel & on & off & cyc. & on & off & cyc. & on & off & cyc.\\\\\\midrule\n")
        for mname in ["BT-linear", "BT-MLP", "mixture-of-utilities", "antisym-bilinear", "blade-chest", "pair-MLP (matched)", "pair-MLP (large)", "FULCRUM-hard", "FULCRUM-soft"]:
            tag = "$^{\\mathrm{T}}$" if mname in TRANSITIVE else ""
            row = [SHORT[mname] + tag]
            for res in (syn[0.0], obs["pokemon_full"][1], obs["sushi_b"][1]):
                if res is None or mname not in res["per_seed"]["0"]:
                    row += ["--"] * 3; continue
                row += [fmt(mean_ci(per_seed(res, mname, k))[0]) for k in ("acc_on_cycle", "acc_off_cycle", "cycle_orientation_acc")]
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\midrule cyclic test triples / pairs on cycles & ")
        cells = []
        for res in (syn[0.0], obs["pokemon_full"][1], obs["sushi_b"][1]):
            if res is None:
                cells.append("\\multicolumn{3}{c}{--}"); continue
            i0 = res["info"]["0"]
            cells.append(f"\\multicolumn{{3}}{{c}}{{{i0['test_cyclic_triples']} / {i0['test_pairs_on_cycle']} (seed 0)}}")
        f.write(" & ".join(cells) + "\\\\\n\\bottomrule\\end{tabular}\\end{table}\n")
        for res, short in ((syn[0.0], "Rzero"), (obs["pokemon_full"][1], "Pk")):
            if res is None:
                continue
            for mname, mk in (("FULCRUM-soft", "Soft"), ("BT-linear", "BT"), ("pair-MLP (large)", "MLPlarge")):
                macros["orient" + short + mk] = f"{mean_ci(per_seed(res, mname, 'cycle_orientation_acc'))[0]:.2f}"
                macros["onCycle" + short + mk] = f"{mean_ci(per_seed(res, mname, 'acc_on_cycle'))[0]:.3f}"
                macros["offCycle" + short + mk] = f"{mean_ci(per_seed(res, mname, 'acc_off_cycle'))[0]:.3f}"

    # ---------------------------------------------------------------- params / runtime
    with open(f"{OUT}/tab_params.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Parameter counts and mean fit time (seconds, one GPU, full-batch Adam with early stopping) on World C ($m=5$) and Pok\\'emon ($m=7$ criteria, 25 per-item features, 784 items).}\n\\label{tab:params}\n")
        f.write("\\begin{tabular}{lcccc}\\toprule\nmodel & params (C) & time (C) & params (Pk) & time (Pk)\\\\\\midrule\n")
        for mname in ORDER:
            if mname.startswith("FULCRUM-soft ("):
                continue
            row = [SHORT[mname]]
            for res in (syn[0.0], obs["pokemon_full"][1]):
                if res is None or mname not in res["per_seed"]["0"]:
                    row += ["--", "--"]; continue
                row += [str(int(res["per_seed"]["0"][mname]["params"])), f"{mean_ci(per_seed(res, mname, 'fit_seconds'))[0]:.1f}"]
            f.write(" & ".join(row) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    # ---------------------------------------------------------------- static appendix tables
    with open(f"{OUT}/tab_frameworks.tex", "w") as f:
        f.write("""\\begin{table}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}
\\caption{Preference mechanisms in argumentation and preference learning (citations in the text). Pair-dependent: the priority over criteria can change with the pair; derived cycle: an intransitive defeat cycle is produced from argument features rather than written by hand; postulates: closure and consistency proved ($\\dagger$: under the conditions of Section~\\ref{sec:rational}).}
\\label{tab:frameworks}
\\begin{tabular}{lccccc}\\toprule
framework & multi-criteria & pair-dependent & derived cycle & postulates & learnable\\\\\\midrule
preference-based AF & -- & -- & -- & -- & --\\\\
value-based AF & per audience & -- & -- & -- & --\\\\
\\aspic{} & rules, premises & -- & -- & \\checkmark & --\\\\
ABA$^{+}$ & -- & -- & -- & \\checkmark & --\\\\
extended AF & -- & -- & by hand & -- & --\\\\
gradual / neural AF & -- & -- & -- & -- & \\checkmark\\\\
Bradley--Terry reward model & -- & -- & -- & n/a & \\checkmark\\\\
blade-chest, bilinear, pair-MLP & -- & implicit & \\checkmark & n/a & \\checkmark\\\\
\\fulcrum{} (this paper) & \\checkmark & \\checkmark & \\checkmark & \\checkmark$^\\dagger$ & \\checkmark\\\\\\bottomrule
\\end{tabular}\\end{table}
""")
    agg = ex["aggregation"]["grid_68"]
    with open(f"{OUT}/tab_aggregation.tex", "w") as f:
        f.write(f"""\\begin{{table}}[h]\\centering\\small
\\caption{{Clause 18-2 under conjunction introduction by the rule $g$ that scores a conjunction for preponderance. Middle: the symmetric $k$-cycle witness (proved for all $k$, Theorem~\\ref{{thm:agg}}; checked $k\\le8$). Right: exhaustive count over all {agg['total']} preponderance 3-cycles with integer scores in $\\{{0,\\dots,3\\}}$ over three criteria.}}
\\label{{tab:agg}}
\\begin{{tabular}}{{lcc}}\\toprule
aggregation $g$ & $k$-cycle witness & grid 3-cycles with clause holding\\\\\\midrule
minimum (weakest link) & violated, all $k$ & {agg['min(weakest-link)']}/{agg['total']}\\\\
mean & holds on the witness & {agg['mean']}/{agg['total']}\\\\
maximum & holds, all $k$ & {agg['max']}/{agg['total']}\\\\
sum & holds, all $k$ & {agg['sum']}/{agg['total']}\\\\\\bottomrule
\\end{{tabular}}\\end{{table}}
""")
    with open(f"{OUT}/tab_ablations_discrete.tex", "w") as f:
        f.write("""\\begin{table}[h]\\centering\\small
\\caption{Tag-based discrete study (7 seeds, 1,200 conflicts each, 60/40 split): accuracy $\\pm$ t-based 95\\% half-width and paired gap. The preregistered labels are A1 = multi-criterion structure off (one criterion) and A2 = context conditioning off (one ordering).}
\\label{tab:discrete}
\\begin{tabular}{llccc}\\toprule
regime & model & accuracy & gap vs.\\ full (paired) & flip\\\\\\midrule
context-dependent & full (one ordering per tag) & $0.898\\pm0.017$ & -- & 1.000\\\\
 & A2: one global ordering & $0.656\\pm0.017$ & $-0.242\\pm0.016$ & --\\\\
 & A1: one criterion & $0.627\\pm0.013$ & $-0.271\\pm0.015$ & --\\\\
transitive (control) & full / A1 / A2 & $0.900\\pm0.015$ & $0.000$ & 1.000\\\\\\bottomrule
\\end{tabular}\\end{table}
""")
    c = cnt
    with open(f"{OUT}/tab_counts.tex", "w") as f:
        f.write(f"""\\begin{{table}}[h]\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\caption{{Intransitivity in the studied data, regenerated from the raw files. A triple is decisive if every pair has a strict majority; the fraction is over decisive triples.}}
\\label{{tab:counts}}
\\begin{{tabular}}{{lcc}}\\toprule
data & unit & fraction cyclic\\\\\\midrule
Netflix (PrefLib 00004), {c['netflix']['profiles']} strict-order profiles & profiles with a cycle & {c['netflix']['fraction']:.3f}\\\\
Sushi-A majority, 10 items (5,000 rankings) & triples & {c['sushi_a_majority']['fraction']:.3f}\\\\
Sushi-B majority, 100 items (pairs with $\\ge20$ voters) & triples & {c['sushi_b_majority']['fraction']:.4f} ({c['sushi_b_majority']['cyclic_triples']}/{c['sushi_b_majority']['decisive_triples']})\\\\
Sushi, majority of four item attributes & triples & {c['sushi_attribute_majority']['fraction']:.4f} ({c['sushi_attribute_majority']['cyclic_triples']}/{c['sushi_attribute_majority']['decisive_triples']})\\\\
World C oracle, $r=0$ (Table~\\ref{{tab:synfull}}) & triples & \\fthreeRzero\\\\
Pok\\'emon 18-type chart & type triples & {c['pokemon']['type_chart']['fraction']:.3f} ({c['pokemon']['type_chart']['cyclic']}/{c['pokemon']['type_chart']['decisive']})\\\\
Pok\\'emon pairs, dual-type advantage (20k samples) & triples & {c['pokemon']['pokemon_pairs']['fraction']:.3f}\\\\\\bottomrule
\\end{{tabular}}\\end{{table}}
""")
    write_numbers(macros)
    print("wrote tables; macros:", len(macros))
    return macros


if __name__ == "__main__":
    main()
