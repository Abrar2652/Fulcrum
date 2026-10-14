"""Machine checks for the MAX-GAP capacity theorem (paper Theorem 4).

(a) The m = n construction realises EVERY tournament on n vertices, tie-free:
    criterion h_v for each vertex v; s_{h_v}(v) = B, s_{h_v}(u) = 1 if u -> v in T else 0.
    Checked exhaustively for n = 3, 4, 5 (all 2^{C(n,2)} tournaments) and on random
    tournaments for n = 6..12.
(b) For fixed m the realizable fraction shrinks: we count, by exhaustive enumeration over an
    integer grid, the number of distinct tie-free tournaments MAX-GAP produces on n points
    with m criteria (a lower bound on the exact count, an indication of the trend), and
    compare with n! (m = 1) and 2^{C(n,2)}.
(c) The single-ordering count: m = 1 realises exactly the n! linear orders.
Usage: PYTHONPATH=. python experiments/capacity_check.py
"""
from __future__ import annotations

import itertools
import json
import math
import random


def maxgap_tournament(scores):
    """Return the tournament (dict (i,j)->+1/-1 for i<j) if every pair has a unique max gap
    and a strict verdict; else None (tie somewhere)."""
    n = len(scores); m = len(scores[0])
    T = {}
    for i, j in itertools.combinations(range(n), 2):
        gaps = [abs(scores[i][h] - scores[j][h]) for h in range(m)]
        g = max(gaps)
        if g == 0 or gaps.count(g) > 1:
            return None
        h = gaps.index(g)
        T[(i, j)] = 1 if scores[i][h] > scores[j][h] else -1
    return T


def construction(T, n, B=3):
    """Scores realising tournament T with m = n criteria."""
    s = [[0] * n for _ in range(n)]
    for v in range(n):
        s[v][v] = B
        for u in range(n):
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            u_beats_v = (T[key] == 1) == (u < v)
            s[u][v] = 1 if u_beats_v else 0
    return s


def all_tournaments(n):
    pairs = list(itertools.combinations(range(n), 2))
    for bits in itertools.product((1, -1), repeat=len(pairs)):
        yield dict(zip(pairs, bits))


def check_construction():
    out = {}
    for n in (3, 4, 5):
        ok = tot = 0
        for T in all_tournaments(n):
            tot += 1
            ok += maxgap_tournament(construction(T, n)) == T
        out[f"n={n} exhaustive"] = {"tournaments": tot, "realised": ok}
    rng = random.Random(0)
    for n in (6, 8, 10, 12):
        ok = 0; trials = 300
        for _ in range(trials):
            pairs = list(itertools.combinations(range(n), 2))
            T = {p: rng.choice((1, -1)) for p in pairs}
            ok += maxgap_tournament(construction(T, n)) == T
        out[f"n={n} random"] = {"trials": trials, "realised": ok}
    return out


def realizable_counts(n_max=5, m_max=3, vmax=4):
    """Distinct tie-free tournaments produced by MAX-GAP on an integer grid {0..vmax}^m
    (exhaustive over all n-tuples of distinct grid points, order-sensitive labels)."""
    out = {}
    for n in range(3, n_max + 1):
        total = 2 ** math.comb(n, 2)
        row = {"n": n, "all_tournaments": total, "linear_orders(m=1)": math.factorial(n)}
        for m in range(1, m_max + 1):
            pts = list(itertools.product(range(vmax + 1), repeat=m))
            if len(pts) ** n > 4_000_000:
                row[f"m={m}"] = "skipped (grid too large)"
                continue
            seen = set()
            for combo in itertools.product(pts, repeat=n):
                if len(set(combo)) < n:
                    continue
                T = maxgap_tournament(list(combo))
                if T is not None:
                    seen.add(tuple(sorted(T.items())))
            row[f"m={m}"] = len(seen)
        out[n] = row
    return out


if __name__ == "__main__":
    res = {"construction": check_construction(), "realizable_counts_grid": realizable_counts()}
    print(json.dumps(res, indent=2))
    import os
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(res, open("experiments/results/capacity_check.json", "w"), indent=2)
