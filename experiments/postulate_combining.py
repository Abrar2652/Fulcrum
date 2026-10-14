"""Deepen the postulate-safety characterization PAST the no-combining-rules regime.

Question: with conjunction introduction (A,B => A/\\B a strict rule) present, does the
strength-gated intransitive witness still satisfy Modgil-Prakken Def. 18-2?

Def. 18-2 (set condition) is VIOLATED iff for every i, C_i is strictly preferred to
C'_i, where C'_i is the strict continuation of {C_1..C_n}\\{C_i} -- here the
conjunction of the other arguments, scored by an aggregation rule g. If 18-2 is
violated the ordering is not reasonable and indirect consistency can fail.

We test which conjunction-scoring rules g preserve 18-2 for the symmetric k-cycle
witness (cyclic shifts of (0,..,k-1)). g in {min (weakest-link, ASPIC+ standard),
max, mean, sum}. min = the honest ASPIC+ weakest-link choice.
"""
from __future__ import annotations

import itertools


def maxgap(x, y):
    """+1 if x strictly beats y under MAX-GAP, -1 if y beats x, 0 if tie/equal."""
    gaps = [abs(a - b) for a, b in zip(x, y)]
    g = max(gaps)
    idx = [i for i, gi in enumerate(gaps) if gi == g]
    if g == 0 or len(idx) > 1:
        return 0  # equipollent or ambiguous argmax
    h = idx[0]
    return 1 if x[h] > y[h] else -1


def g_min(vs):  # weakest-link (ASPIC+ standard)
    return [min(c) for c in zip(*vs)]


def g_max(vs):
    return [max(c) for c in zip(*vs)]


def g_mean(vs):
    return [sum(c) / len(c) for c in zip(*vs)]


def g_sum(vs):
    return [sum(c) for c in zip(*vs)]


RULES = {"min(weakest-link)": g_min, "max": g_max, "mean": g_mean, "sum": g_sum}


def witness(k):
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]  # cyclic shifts


def def18_2_violated(args, g):
    """True iff for EVERY i, C_i beats g(others_i)  (=> 18-2 violated => not reasonable)."""
    n = len(args)
    for i in range(n):
        others = [args[j] for j in range(n) if j != i]
        conj = g(others)
        if maxgap(list(args[i]), conj) != 1:   # C_i does NOT strictly beat the conjunction
            return False                        # antecedent fails at i => 18-2 holds
    return True


def is_cycle(args):
    n = len(args)
    return all(maxgap(list(args[s]), list(args[(s + 1) % n])) == 1 for s in range(n))


print("=" * 70)
print("Def-18-2 under conjunction introduction, symmetric k-cycle witness")
print("VIOLATED = intransitivity UNSAFE (indirect consistency can fail)")
print("=" * 70)
for k in (3, 4, 5):
    args = witness(k)
    assert is_cycle(args), f"k={k} not a preponderance cycle"
    print(f"\nk={k}  witness={args}")
    for name, g in RULES.items():
        v = def18_2_violated(args, g)
        print(f"  g = {name:20s} -> 18-2 {'VIOLATED (unsafe)' if v else 'holds (SAFE)'}")

# Does ANY symmetric-ish random 3-config with a preponderance cycle stay safe under
# weakest-link? Search small integer configs.
print("\n" + "=" * 70)
print("Among ALL 3-arg preponderance cycles (scores 0..3, 3 heads): how many are")
print("SAFE (18-2 holds) under weakest-link min vs under max?")
print("=" * 70)
vals = range(4)
space = list(itertools.product(vals, repeat=3))
tot = 0
safe = {name: 0 for name in RULES}
for a, b, c in itertools.combinations(space, 3):
    if not (is_cycle((a, b, c)) or is_cycle((a, c, b))):
        continue
    ordered = (a, b, c) if is_cycle((a, b, c)) else (a, c, b)
    tot += 1
    for name, g in RULES.items():
        if not def18_2_violated(ordered, g):
            safe[name] += 1
print(f"total 3-cycles found: {tot}")
for name in RULES:
    print(f"SAFE under {name:20s}: {safe[name]}/{tot} = {safe[name]/max(tot,1):.3f}")
