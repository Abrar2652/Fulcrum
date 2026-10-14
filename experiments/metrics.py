"""Metrics and interval procedures shared by all benchmark scripts.

Pairwise metrics : accuracy, NLL, Brier, ECE (15 equal-width confidence bins).
Structure metrics: accuracy on pairs that lie on a cyclic triple of the reference
                   relation vs the rest; accuracy by criterion-conflict level; cycle
                   orientation accuracy (all three arcs of a cyclic triple correct).
Intervals        : paired t-interval (df = n_seeds - 1) and paired percentile bootstrap.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict

import torch

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045}


def t975(df: int) -> float:
    if df in T975:
        return T975[df]
    keys = sorted(T975)
    for k in keys:
        if k >= df:
            return T975[k]
    return 1.96


# ------------------------------------------------------------------ pairwise metrics
@torch.no_grad()
def basic_metrics(p: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict:
    p = p.clamp(1e-6, 1 - 1e-6)
    pred = (p >= 0.5).float()
    acc = float((pred == y).float().mean())
    nll = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())
    brier = float(((p - y) ** 2).mean())
    conf = torch.maximum(p, 1 - p)
    correct = (pred == y).float()
    edges = torch.linspace(0.5, 1.0, bins + 1, device=p.device)
    ece = 0.0
    n = len(p)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        msk = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if msk.sum() == 0:
            continue
        ece += float(msk.sum()) / n * abs(float(correct[msk].mean()) - float(conf[msk].mean()))
    return {"acc": acc, "nll": nll, "brier": brier, "ece": ece, "n": n}


# ------------------------------------------------------------------ structure metrics
def cyclic_triples(edges: dict[tuple[int, int], int]) -> list[tuple[int, int, int]]:
    """edges: {(i,j): +1 if i beats j, -1 if j beats i} over unordered pairs (i<j).
    Returns the cyclic triples (i,j,k) as node triples."""
    win = defaultdict(set)
    nodes = set()
    for (i, j), s in edges.items():
        nodes.add(i); nodes.add(j)
        if s > 0:
            win[i].add(j)
        elif s < 0:
            win[j].add(i)
    out = []
    for i in nodes:
        for j in win[i]:
            for k in win[j]:
                if k in win and i in win[k] and i < j and i < k:
                    out.append((i, j, k))
    return out


def pairs_on_cycles(edges: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    on = set()
    for i, j, k in cyclic_triples(edges):
        for a, b in ((i, j), (j, k), (i, k)):
            on.add((min(a, b), max(a, b)))
    return on


def cycle_orientation_accuracy(edges_true: dict[tuple[int, int], int],
                               pred: dict[tuple[int, int], int]) -> tuple[float, int]:
    """Fraction of cyclic triples of the reference relation on which the model's three
    predicted orientations all agree with the reference. Returns (fraction, count)."""
    tri = cyclic_triples(edges_true)
    if not tri:
        return float("nan"), 0
    ok = 0
    for i, j, k in tri:
        good = True
        for a, b in ((i, j), (j, k), (i, k)):
            key = (min(a, b), max(a, b))
            if pred.get(key, 0) != edges_true[key]:
                good = False; break
        ok += good
    return ok / len(tri), len(tri)


def conflict_level(d: torch.Tensor, pol: torch.Tensor | None = None) -> torch.Tensor:
    """Number of criteria that disagree with the criterion majority for each pair
    (0 = all criteria agree). `pol` optionally flips criteria to a common polarity."""
    s = torch.sign(d if pol is None else d * pol)
    pos = (s > 0).sum(-1); neg = (s < 0).sum(-1)
    return torch.minimum(pos, neg)


# ------------------------------------------------------------------ intervals
def paired_t_interval(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    return m, t975(n - 1) * sd / math.sqrt(n)


def paired_bootstrap_interval(diffs: list[float], B: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(B):
        means.append(sum(rng.choice(diffs) for _ in range(n)) / n)
    means.sort()
    return sum(diffs) / n, means[int(0.025 * B)], means[int(0.975 * B) - 1]


def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and t-based 95% half-width (unpaired, per-model)."""
    return paired_t_interval(list(xs))


def summarize_paired(results: dict[str, list[float]], reference: str) -> dict[str, dict]:
    """results[model] = per-seed metric list. Returns per-model mean, per-model CI, and the
    paired gap (model - reference) with t and bootstrap intervals."""
    out = {}
    ref = results[reference]
    for name, xs in results.items():
        m, h = mean_ci(xs)
        diffs = [x - r for x, r in zip(xs, ref)]
        dm, dh = paired_t_interval(diffs)
        _, lo, hi = paired_bootstrap_interval(diffs)
        out[name] = {"mean": m, "ci": h, "gap": dm, "gap_t": dh, "gap_boot": (lo, hi)}
    return out
