"""Shared benchmark runner: trains every model in a zoo on identical splits across seeds and
records pairwise metrics, structural metrics (accuracy on pairs that lie on cyclic
triples of the reference relation, accuracy by criterion-conflict level, cycle
orientation accuracy), parameter counts, fit time, and attribution diagnostics.
Results are written as JSON; `summarize` turns them into paired comparisons."""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import torch

from experiments.faithfulness import builtin_attribution, diagnostics, occlusion_attribution
from experiments.metrics import (basic_metrics, conflict_level, cycle_orientation_accuracy,
                                 cyclic_triples, pairs_on_cycles, summarize_paired)
from experiments.zoo import (DEVICE, Lexicographic, PairData, calibrate_temperature,
                             predict_proba, predict_proba_core, standardizer, train_model)


# ------------------------------------------------------------------ reference relation
def reference_relation(p: PairData) -> dict[tuple[int, int], int]:
    """Majority outcome per unordered item pair in `p` (requires item ids)."""
    if p.ia is None:
        return {}
    cnt: dict = defaultdict(lambda: [0, 0])
    ia, ib, y = p.ia.tolist(), p.ib.tolist(), p.y.tolist()
    for a, b, yy in zip(ia, ib, y):
        key = (min(a, b), max(a, b))
        first_won = (yy == 1.0) == (a < b)
        cnt[key][0 if first_won else 1] += 1
    return {k: (1 if v[0] > v[1] else (-1 if v[1] > v[0] else 0)) for k, v in cnt.items()}


def predicted_relation(p: PairData, prob: torch.Tensor) -> dict[tuple[int, int], int]:
    cnt: dict = defaultdict(lambda: [0.0, 0.0])
    ia, ib, pr = p.ia.tolist(), p.ib.tolist(), prob.tolist()
    for a, b, q in zip(ia, ib, pr):
        key = (min(a, b), max(a, b))
        q_first = q if a < b else 1 - q
        cnt[key][0] += q_first; cnt[key][1] += 1 - q_first
    return {k: (1 if v[0] > v[1] else (-1 if v[1] > v[0] else 0)) for k, v in cnt.items()}


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, te: PairData, pol: torch.Tensor | None, ref: dict, on_cycle: set,
             conflict_bins=(0, 1, 2), slice_masks: dict | None = None) -> dict:
    prob = predict_proba(model, te)
    out = basic_metrics(prob, te.y)
    pred = (prob >= 0.5).float()
    correct = (pred == te.y).float()
    if te.ia is not None and ref:
        keys = [(min(a, b), max(a, b)) for a, b in zip(te.ia.tolist(), te.ib.tolist())]
        mask = torch.tensor([k in on_cycle for k in keys], device=te.d.device)
        out["acc_on_cycle"] = float(correct[mask].mean()) if mask.any() else float("nan")
        out["acc_off_cycle"] = float(correct[~mask].mean()) if (~mask).any() else float("nan")
        out["n_on_cycle"] = int(mask.sum())
        # relation-level metric from the antisymmetric core (see zoo.predict_proba_core)
        frac, ntri = cycle_orientation_accuracy(ref, predicted_relation(te, predict_proba_core(model, te)))
        out["cycle_orientation_acc"] = frac
        out["n_cyclic_triples"] = ntri
    lv = conflict_level(te.d, pol)
    for b in conflict_bins:
        msk = (lv == b) if b < conflict_bins[-1] else (lv >= b)
        out[f"acc_conflict_{b}{'+' if b == conflict_bins[-1] else ''}"] = (
            float(correct[msk].mean()) if msk.any() else float("nan"))
        out[f"n_conflict_{b}{'+' if b == conflict_bins[-1] else ''}"] = int(msk.sum())
    if slice_masks:
        for sname, msk in slice_masks.items():
            msk = msk.to(correct.device)
            out[f"acc_slice_{sname}"] = float(correct[msk].mean()) if msk.any() else float("nan")
            out[f"n_slice_{sname}"] = int(msk.sum())
    return out


def fit_and_eval(name, ctor, tr, va, te, pol, ref, on_cycle, seed, epochs, lr, batch_size,
                 weight_decay, faith: bool, slice_masks=None):
    model = ctor()
    model, secs = train_model(model, tr, va, epochs=epochs, lr=lr, seed=seed,
                              batch_size=batch_size, weight_decay=weight_decay)
    calibrate_temperature(model, va)
    res = evaluate(model, te, pol, ref, on_cycle, slice_masks=slice_masks)
    res["fit_seconds"] = secs
    res["params"] = model.n_params() if hasattr(model, "n_params") else 0
    res["temperature"] = float(model.temperature)
    if hasattr(model, "beta"):
        res["beta"] = float(model.beta)
    if faith and not isinstance(model, Lexicographic):
        try:
            if hasattr(model, "contributions"):
                res["faithfulness"] = diagnostics(model, te, builtin_attribution(model, te), seed)
            else:
                res["faithfulness_occlusion"] = diagnostics(model, te, occlusion_attribution(model, te), seed)
        except Exception as e:  # pragma: no cover
            res["faithfulness_error"] = repr(e)
    return res, model


def run_benchmark(tag: str, make_split, zoo: dict, seeds, out_dir="experiments/results",
                  epochs=400, lr=0.05, batch_size=None, weight_decay=1e-4, faith=True,
                  extra=None, verbose=True, slices=None):
    """make_split(seed) -> (train, val, test, info) as PairData on CPU. Standardisation is
    fitted on train. Returns the results dict and writes <out_dir>/<tag>.json."""
    os.makedirs(out_dir, exist_ok=True)
    results = {"tag": tag, "seeds": list(seeds), "per_seed": {}, "info": {}}
    for seed in seeds:
        tr, va, te, info = make_split(seed)
        std = standardizer(tr)
        tr, va, te = std(tr).to(DEVICE), std(va).to(DEVICE), std(te).to(DEVICE)
        yc = 2 * tr.y - 1
        pol = torch.sign((tr.d * yc[:, None]).mean(0)); pol[pol == 0] = 1
        ref = reference_relation(te)
        on_cycle = pairs_on_cycles(ref) if ref else set()
        info = dict(info)
        info["n_train"], info["n_val"], info["n_test"] = len(tr), len(va), len(te)
        info["test_cyclic_triples"] = len(cyclic_triples(ref)) if ref else None
        info["test_pairs_on_cycle"] = len(on_cycle)
        results["info"][str(seed)] = info
        row = {}
        slice_masks = slices(te) if slices is not None else None
        for name, ctor in zoo.items():
            res, _ = fit_and_eval(name, ctor, tr, va, te, pol, ref, on_cycle, seed, epochs, lr,
                                  batch_size, weight_decay, faith, slice_masks)
            row[name] = res
            if verbose:
                print(f"[{tag} seed {seed}] {name:24s} acc={res['acc']:.4f} nll={res['nll']:.3f} "
                      f"brier={res['brier']:.3f} ece={res['ece']:.3f} params={res['params']} "
                      f"t={res['fit_seconds']:.1f}s", flush=True)
        results["per_seed"][str(seed)] = row
        if extra is not None:
            results.setdefault("extra", {})[str(seed)] = extra(seed, tr, va, te, info)
        json.dump(results, open(f"{out_dir}/{tag}.json", "w"), indent=1)
    results["summary"] = summarize(results)
    json.dump(results, open(f"{out_dir}/{tag}.json", "w"), indent=1)
    return results


def summarize(results: dict, reference="BT-linear", metrics=("acc", "nll", "brier", "ece",
              "acc_on_cycle", "acc_off_cycle", "cycle_orientation_acc")) -> dict:
    per = results["per_seed"]
    seeds = list(per)
    models = list(per[seeds[0]])
    out = {}
    for met in metrics:
        table = {}
        for mname in models:
            xs = [per[s][mname].get(met, float("nan")) for s in seeds]
            if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in xs):
                continue
            table[mname] = xs
        if reference in table:
            out[met] = summarize_paired(table, reference)
    return out


def print_summary(results: dict, metric="acc", reference="BT-linear"):
    s = results["summary"].get(metric, {})
    print(f"\n{results['tag']}: {metric} (mean ± t95; paired gap vs {reference} [t95] [boot 95%])")
    for name, r in sorted(s.items(), key=lambda kv: -kv[1]["mean"]):
        lo, hi = r["gap_boot"]
        print(f"  {name:24s} {r['mean']:.4f} ± {r['ci']:.4f}   gap {r['gap']:+.4f} ± {r['gap_t']:.4f} "
              f"[{lo:+.4f},{hi:+.4f}]")
