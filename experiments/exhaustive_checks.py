"""Deterministic exhaustive checks behind the theorems (paper: reproducibility statement).

  1. 29 preorders on three elements: none realises the MAX-GAP witness comparator table or
     the directed-cycle (preferred, stable) profile (Theorems 1-2).
  2. 355 preorders on four elements: how many match FULCRUM's grounded extension on the
     spoiler instance (grounded does not separate).
  3. Strict k-cycles under MAX-GAP for k = 3..7 (Theorem 3).
  4. Cyclic-triple counts on integer grids under three endogenous policies.
  5. Clause 18-2 under conjunction introduction: the 68 preponderance 3-cycles on the
     {0..3}^3 grid under min/mean/max/sum (Theorem 6 table), and the general aggregation
     theorem checked on random finite sets: every max-dominating aggregation satisfies the
     set clause on every sampled set; min-dominated aggregations violate it on every
     k-cycle witness (Theorem 'aggregation').
  6. Proposition 1: swap on the deciding criterion flips every MAX-GAP verdict; the
     endogenous 'favour-the-higher-total' policy is a counter-example.
Usage: PYTHONPATH=. python experiments/exhaustive_checks.py
"""
from __future__ import annotations

import itertools
import json
import random

from experiments.postulate_combining import RULES, def18_2_violated, is_cycle, maxgap, witness


# ------------------------------------------------------------------ MAX-GAP comparator
def maxgap_verdict(x, y, tiebreak=None):
    """+1 if x beats y, -1 if y beats x, 0 equipollent. Ties in the gap are broken by the
    fixed criterion order `tiebreak` (default: index order)."""
    m = len(x)
    order = sorted(range(m), key=lambda h: (-abs(x[h] - y[h]), (tiebreak or list(range(m))).index(h)))
    for h in order:
        if x[h] != y[h]:
            return 1 if x[h] > y[h] else -1
    return 0


# ------------------------------------------------------------------ preorders
def all_preorders(n):
    elems = range(n)
    off = [(i, j) for i in elems for j in elems if i != j]
    for bits in itertools.product((0, 1), repeat=len(off)):
        rel = {(i, i) for i in elems} | {p for p, b in zip(off, bits) if b}
        if all(not ((i, j) in rel and (j, k) in rel and (i, k) not in rel)
               for i in elems for j in elems for k in elems):
            yield rel


def strict_table(rel, n):
    t = {}
    for i, j in itertools.permutations(range(n), 2):
        gij, gji = (i, j) in rel, (j, i) in rel
        t[(i, j)] = 1 if gij and not gji else (-1 if gji and not gij else 0)
    return t


def defeat_graph_from_preorder(rel, n, attacks):
    """Mutual attacks resolved by the preorder; one-way attacks repelled iff target strict."""
    edges = set()
    for (u, v) in attacks:
        mutual = (v, u) in attacks
        guv, gvu = (u, v) in rel, (v, u) in rel
        if mutual:
            if guv and not gvu:
                edges.add((u, v))
            elif gvu and not guv:
                edges.add((v, u))
            else:
                edges.add((u, v)); edges.add((v, u))
        else:
            if gvu and not guv:      # target strictly preferred -> repelled
                continue
            edges.add((u, v))
    return edges


def dung(nodes, edges):
    nodes = list(nodes)
    att = {(u, v) for u, v in edges}

    def cf(S):
        return all((a, b) not in att for a in S for b in S)

    def defends(S, x):
        return all(any((d, y) in att for d in S) for (y, z) in att if z == x)
    subsets = [set(c) for r in range(len(nodes) + 1) for c in itertools.combinations(nodes, r)]
    adm = [S for S in subsets if cf(S) and all(defends(S, x) for x in S)]
    pref = {frozenset(S) for S in adm if not any(S < T for T in adm)}
    stab = {frozenset(S) for S in subsets if cf(S) and all(any((m, o) in att for m in S) for o in set(nodes) - S)}
    g = set()
    while True:
        nxt = {x for x in nodes if defends(g, x)} | g
        if nxt == g:
            break
        g = nxt
    return g, pref, stab


def check_preorders():
    A = {0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}
    tab = {(i, j): maxgap_verdict(A[i], A[j]) for i, j in itertools.permutations(range(3), 2)}
    pre3 = list(all_preorders(3))
    realizers = sum(strict_table(r, 3) == tab for r in pre3)
    attacks = {(i, j) for i, j in itertools.permutations(range(3), 2)}
    f_edges = {(0, 1), (1, 2), (2, 0)}
    g_f, p_f, s_f = dung(range(3), f_edges)
    same_def = same_prof = 0
    for r in pre3:
        e = defeat_graph_from_preorder(r, 3, attacks)
        same_def += e == f_edges
        _, p, s = dung(range(3), e)
        same_prof += (p, s) == (p_f, s_f)
    # spoiler: d -> a one-way, on top of the mutual triangle
    B = dict(A); B[3] = (5, 5, 5)
    attacks4 = attacks | {(3, 0)}
    f4 = {(0, 1), (1, 2), (2, 0), (3, 0)}
    g4, _, _ = dung(range(4), f4)
    pre4 = list(all_preorders(4))
    same_g = sum(dung(range(4), defeat_graph_from_preorder(r, 4, attacks4))[0] == g4 for r in pre4)
    return {"preorders_3": len(pre3), "realize_comparator": realizers, "same_defeat_graph": same_def,
            "same_pref_stable_profile": same_prof, "fulcrum_profile": {"grounded": sorted(g_f),
            "preferred": [sorted(x) for x in p_f], "stable": [sorted(x) for x in s_f]},
            "preorders_4": len(pre4), "spoiler_grounded": sorted(g4), "preorders_4_same_grounded": same_g}


# ------------------------------------------------------------------ k-cycles and grids
def check_kcycles():
    out = {}
    for k in range(3, 8):
        args = witness(k)
        out[k] = all(maxgap_verdict(list(args[s]), list(args[(s + 1) % k])) == 1 for s in range(k))
    return out


def policy_counts():
    def least_gap(x, y):
        m = len(x)
        order = sorted(range(m), key=lambda h: (abs(x[h] - y[h]), h))
        for h in order:
            if x[h] != y[h]:
                return 1 if x[h] > y[h] else -1
        return 0

    def largest_sum(x, y):
        m = len(x)
        order = sorted(range(m), key=lambda h: (-(x[h] + y[h]), h))
        for h in order:
            if x[h] != y[h]:
                return 1 if x[h] > y[h] else -1
        return 0
    res = {}
    for name, pol in (("max-gap", maxgap_verdict), ("least-gap", least_gap), ("largest-sum", largest_sum)):
        for vmax in (2, 3):
            if name != "max-gap" and vmax == 3:
                continue
            space = list(itertools.product(range(vmax + 1), repeat=3))
            cnt = 0
            for a, b, c in itertools.combinations(space, 3):
                ab, bc, ca = pol(a, b), pol(b, c), pol(c, a)
                if 0 in (ab, bc, ca):
                    continue
                if ab == bc == ca:
                    cnt += 1
            res[f"{name} scores 0..{vmax}"] = cnt
    return res


# ------------------------------------------------------------------ aggregation theorem
def g_max(vs): return [max(c) for c in zip(*vs)]
def g_sum(vs): return [sum(c) for c in zip(*vs)]
def g_min(vs): return [min(c) for c in zip(*vs)]
def g_mean(vs): return [sum(c) / len(c) for c in zip(*vs)]
def g_max_plus(vs): return [max(c) + 0.5 for c in zip(*vs)]          # max-dominating
def g_half_min(vs): return [0.5 * min(c) for c in zip(*vs)]           # min-dominated, >=0


def set_clause_holds(args, g, tiebreak=None):
    """18-2 set clause for {C_i} with continuation score g(others): holds iff NOT all i beat."""
    n = len(args)
    for i in range(n):
        others = [args[j] for j in range(n) if j != i]
        if maxgap_verdict(list(args[i]), g(others), tiebreak) != 1:
            return True
    return False


def check_aggregation(n_random=20000, seed=0):
    rng = random.Random(seed)
    out = {"grid_68": {}, "random_sets": {}, "witness_min_dominated": {}}
    # (a) the 68 grid cycles under the four rules (as in the paper's table)
    vals = range(4)
    space = list(itertools.product(vals, repeat=3))
    tot = 0; safe = {n_: 0 for n_ in RULES}
    for a, b, c in itertools.combinations(space, 3):
        if not (is_cycle((a, b, c)) or is_cycle((a, c, b))):
            continue
        ordered = (a, b, c) if is_cycle((a, b, c)) else (a, c, b)
        tot += 1
        for n_, g in RULES.items():
            safe[n_] += not def18_2_violated(ordered, g)
    out["grid_68"] = {"total": tot, **{n_: safe[n_] for n_ in RULES}}
    # (b) random finite sets (n in 2..6, m in 2..6, integer scores 0..9, random tie-break):
    #     every max-dominating g must satisfy the set clause on EVERY set.
    for gname, g in (("max", g_max), ("sum (nonneg)", g_sum), ("max+0.5", g_max_plus)):
        viol = 0
        for _ in range(n_random):
            n = rng.randint(2, 6); m = rng.randint(2, 6)
            args = [tuple(rng.randint(0, 9) for _ in range(m)) for _ in range(n)]
            tb = list(range(m)); rng.shuffle(tb)
            viol += not set_clause_holds(args, g, tb)
        out["random_sets"][gname] = {"sampled": n_random, "violations": viol}
    # mean is neither max-dominating nor min-dominated: count how often it fails
    viol = 0
    for _ in range(n_random):
        n = rng.randint(2, 6); m = rng.randint(2, 6)
        args = [tuple(rng.randint(0, 9) for _ in range(m)) for _ in range(n)]
        viol += not set_clause_holds(args, g_mean)
    out["random_sets"]["mean"] = {"sampled": n_random, "violations": viol}
    # (c) min-dominated nonnegative aggregations violate the clause on every k-cycle witness
    for gname, g in (("min", g_min), ("0.5*min", g_half_min)):
        out["witness_min_dominated"][gname] = {k: (not set_clause_holds(witness(k), g)) for k in range(3, 9)}
    return out


# ------------------------------------------------------------------ Proposition 1
def check_flip(n_random=20000, seed=1):
    rng = random.Random(seed)

    def deciding(x, y):
        m = len(x)
        order = sorted(range(m), key=lambda h: (-abs(x[h] - y[h]), h))
        for h in order:
            if x[h] != y[h]:
                return h
        return None
    fails = 0; tested = 0
    for _ in range(n_random):
        m = rng.randint(2, 6)
        x = [rng.randint(0, 9) for _ in range(m)]; y = [rng.randint(0, 9) for _ in range(m)]
        h = deciding(x, y)
        if h is None:
            continue
        tested += 1
        v0 = maxgap_verdict(x, y)
        x2, y2 = list(x), list(y); x2[h], y2[h] = y[h], x[h]
        fails += maxgap_verdict(x2, y2) != -v0

    # counter-example policy: criteria on which the higher-total argument is higher come first
    def total_policy_verdict(x, y):
        tx, ty = sum(x), sum(y)
        w, l = (x, y) if tx >= ty else (y, x)
        m = len(x)
        order = sorted(range(m), key=lambda h: (0 if w[h] > l[h] else 1, h))
        for h in order:
            if x[h] != y[h]:
                return 1 if x[h] > y[h] else -1
        return 0

    def total_policy_deciding(x, y):
        tx, ty = sum(x), sum(y)
        w, l = (x, y) if tx >= ty else (y, x)
        m = len(x)
        order = sorted(range(m), key=lambda h: (0 if w[h] > l[h] else 1, h))
        for h in order:
            if x[h] != y[h]:
                return h
        return None
    a, b = [2, 1, 5], [1, 0, 0]
    h = total_policy_deciding(a, b)
    v0 = total_policy_verdict(a, b)
    a2, b2 = list(a), list(b); a2[h], b2[h] = b[h], a[h]
    v1 = total_policy_verdict(a2, b2)
    return {"maxgap_tested": tested, "maxgap_flip_failures": fails,
            "counterexample": {"a": a, "b": b, "deciding": h, "verdict_before": v0,
                               "verdict_after_swap": v1, "flipped": v1 == -v0}}


if __name__ == "__main__":
    out = {"preorders": check_preorders(), "k_cycles": check_kcycles(), "policy_counts": policy_counts(),
           "aggregation": check_aggregation(), "proposition1": check_flip()}
    print(json.dumps(out, indent=2, default=str))
    import os
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(out, open("experiments/results/exhaustive_checks.json", "w"), indent=2, default=str)
