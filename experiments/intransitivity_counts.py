"""Intransitivity counts on the studied data (paper Section 6, appendix table).

Regenerates every 'fraction of decisive triples that cycle' number from the raw files:
  * Netflix (PrefLib 00004, 152 strict-order profiles): a profile is cyclic if its pairwise
    majority relation (over its 3-4 alternatives) contains a directed cycle.
  * Sushi set A (10 items, 5,000 complete rankings): majority relation over the 10 items.
  * Sushi set B (100 items, 5,000 rankings of 10 sampled items each): majority relation over
    the 100 items restricted to pairs compared by at least `min_voters` users.
  * Sushi attribute-majority rule: item i beats j if it is higher on a majority of the four
    numerical item attributes (oiliness, eating frequency, price, sale frequency) — the
    'correlated criteria' analysis of the earlier draft.
  * Pokemon 18-type chart and sampled Pokemon-pair triples under the dual-type advantage.
Usage: PYTHONPATH=. python experiments/intransitivity_counts.py
"""
from __future__ import annotations

import glob
import itertools
import json
import random
from collections import defaultdict

DATA = "data"


def _cyclic_fraction_from_majority(maj: dict[tuple[int, int], int], items) -> tuple[int, int, float]:
    """maj[(i,j)] for i<j: +1 i beats j, -1 j beats i, 0 tie/uncompared."""
    win = defaultdict(set)
    for (i, j), s in maj.items():
        if s > 0:
            win[i].add(j)
        elif s < 0:
            win[j].add(i)
    dec = cyc = 0
    for i, j, k in itertools.combinations(sorted(items), 3):
        ok = all(((a in win[b]) or (b in win[a])) for a, b in ((i, j), (j, k), (i, k)))
        if not ok:
            continue
        dec += 1
        if all(len(win[x] & {i, j, k}) == 1 for x in (i, j, k)):
            cyc += 1
    return cyc, dec, (cyc / dec if dec else float("nan"))


def _majority_from_rankings(rankings, min_voters: int = 1):
    pref = defaultdict(lambda: [0, 0])   # (i<j) -> [#i>j, #j>i]
    for r in rankings:
        pos = {it: t for t, it in enumerate(r)}
        for i, j in itertools.combinations(r, 2):
            key = (min(i, j), max(i, j))
            better_first = pos[key[0]] < pos[key[1]]
            pref[key][0 if better_first else 1] += 1
    maj = {}
    for key, (a, b) in pref.items():
        if a + b < min_voters:
            continue
        maj[key] = 1 if a > b else (-1 if b > a else 0)
    return maj


# ---------------------------------------------------------------- Netflix (PrefLib soc)
def netflix():
    files = sorted(glob.glob(f"{DATA}/preflib/00004-*.soc"))
    n_cyc = 0
    for f in files:
        rankings = []
        alts = set()
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cnt, order = line.split(":")
            order = [int(x) for x in order.split(",")]
            alts.update(order)
            rankings.extend([order] * int(cnt))
        maj = _majority_from_rankings(rankings)
        cyc, dec, _ = _cyclic_fraction_from_majority(maj, alts)
        n_cyc += cyc > 0
    return {"profiles": len(files), "profiles_with_majority_cycle": n_cyc,
            "fraction": n_cyc / len(files) if files else float("nan")}


# ---------------------------------------------------------------- Sushi
def _read_orders(path):
    out = []
    with open(path) as f:
        next(f)                       # header line "100 1" / "10 1"
        for line in f:
            parts = line.split()
            if not parts:
                continue
            k = int(parts[1])
            out.append([int(x) for x in parts[2:2 + k]])
    return out


def sushi_a():
    rk = _read_orders(f"{DATA}/sushi3-2016/sushi3a.5000.10.order")
    maj = _majority_from_rankings(rk)
    cyc, dec, frac = _cyclic_fraction_from_majority(maj, range(10))
    return {"users": len(rk), "cyclic_triples": cyc, "decisive_triples": dec, "fraction": frac}


def sushi_b(min_voters: int = 20):
    rk = _read_orders(f"{DATA}/sushi3-2016/sushi3b.5000.10.order")
    maj = _majority_from_rankings(rk, min_voters=min_voters)
    cyc, dec, frac = _cyclic_fraction_from_majority(maj, range(100))
    return {"users": len(rk), "min_voters_per_pair": min_voters, "pairs_compared": len(maj),
            "cyclic_triples": cyc, "decisive_triples": dec, "fraction": frac}


def sushi_attributes():
    items = {}
    for line in open(f"{DATA}/sushi3-2016/sushi3.idata"):
        p = line.split("\t")
        iid = int(p[0])
        # columns: id name style major minor oiliness eatfreq price salefreq
        items[iid] = [float(p[5]), float(p[6]), float(p[7]), float(p[8])]
    # polarity: lower oiliness value = heavier (0 heavy .. 4 light); we use the raw
    # numeric direction 'higher is better' for each attribute exactly as the earlier
    # draft did (a majority-of-attributes rule, not a preference claim).
    maj = {}
    for i, j in itertools.combinations(sorted(items), 2):
        w = sum(items[i][h] > items[j][h] for h in range(4))
        l = sum(items[i][h] < items[j][h] for h in range(4))
        maj[(i, j)] = 1 if w > l else (-1 if l > w else 0)
    cyc, dec, frac = _cyclic_fraction_from_majority(maj, items.keys())
    return {"items": len(items), "cyclic_triples": cyc, "decisive_triples": dec, "fraction": frac}


# ---------------------------------------------------------------- Pokemon
def pokemon(n_triples: int = 20000, seed: int = 0):
    from experiments.pokemon_intransitivity import load, mon_intransitivity, type_intransitivity
    cyc, dec = type_intransitivity()
    mons, _ = load()
    rate, mdec = mon_intransitivity(mons, n_triples=n_triples, seed=seed)
    return {"type_chart": {"cyclic": cyc, "decisive": dec, "fraction": cyc / dec},
            "pokemon_pairs": {"fraction": rate, "decisive_sampled": mdec}}


if __name__ == "__main__":
    out = {"netflix": netflix(), "sushi_a_majority": sushi_a(),
           "sushi_b_majority": sushi_b(), "sushi_attribute_majority": sushi_attributes(),
           "pokemon": pokemon()}
    print(json.dumps(out, indent=2))
    import os
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(out, open("experiments/results/intransitivity_counts.json", "w"), indent=2)
