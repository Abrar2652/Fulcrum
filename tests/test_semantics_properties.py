"""Phase-3 semantics/properties (theory/semantics_and_properties.md), executable.

These back Props 1-4 with a randomized battery run under the *intransitive*
MAX-GAP policy (the worst case for hygiene). Plain pytest, no hypothesis
dependency, deterministic seed.
"""
from __future__ import annotations

import itertools
import random

from fulcrum.defeat import ArgumentationFramework
from fulcrum.semantics import grounded_extension, is_admissible, is_conflict_free
from fulcrum.structures import HEAD_ORDER, Argument, EpistemicStrength

H3 = HEAD_ORDER[:3]
SEED = 20260701
N_TRIALS = 1500


def _max_gap(a, b, base):
    hs = list(base)
    hs.sort(key=lambda h: (-abs(a.score(h) - b.score(h)), base.index(h)))
    return tuple(hs)


def _rand_arg(aid, rng):
    return Argument(aid, aid, head_scores={h: float(rng.randint(0, 3)) for h in H3},
                    strength=EpistemicStrength.PROBABLE)


def _neutral(aid):
    return Argument(aid, aid, head_scores={h: 0.0 for h in H3})


def _random_attacks(ids, rng, p=0.4):
    return [(i, j) for i in ids for j in ids if i != j and rng.random() < p]


def _pref_stable(g):
    nodes = list(g.nodes())
    subs = [set(s) for r in range(len(nodes) + 1)
            for s in itertools.combinations(nodes, r)]
    adm = [s for s in subs if is_admissible(g, s)]
    pref = [s for s in adm if not any(s < t for t in adm)]
    stab = [s for s in subs if is_conflict_free(g, s)
            and all(any(g.has_edge(m, o) for m in s) for o in set(nodes) - s)]
    return pref, stab


def test_p1_recovery_conservative_extension():
    """Prop 2: trivial preference => D == R (recover Dung)."""
    rng = random.Random(SEED)
    for _ in range(N_TRIALS):
        ids = [f"x{i}" for i in range(rng.randint(2, 5))]
        args = [_neutral(i) for i in ids]
        atk = _random_attacks(ids, rng)
        af = ArgumentationFramework(args, atk, reorder=_max_gap)
        assert set(af.defeats()) == set(atk)


def test_p4_defeat_order_independent():
    """Prop 1: D does not depend on attack enumeration order."""
    rng = random.Random(SEED + 1)
    for _ in range(N_TRIALS):
        ids = [f"x{i}" for i in range(rng.randint(2, 5))]
        args = [_rand_arg(i, rng) for i in ids]
        atk = _random_attacks(ids, rng)
        d1 = set(ArgumentationFramework(args, atk, reorder=_max_gap).defeats())
        shuf = list(atk)
        rng.shuffle(shuf)
        d2 = set(ArgumentationFramework(args, shuf, reorder=_max_gap).defeats())
        assert d1 == d2


def test_p2_p3_consistency_under_intransitive_preference():
    """Prop 3 + Prop 4: mutual conflicts keep >=1 defeat direction, and no
    mutually-attacking pair is jointly accepted in any extension."""
    rng = random.Random(SEED + 2)
    for _ in range(N_TRIALS):
        ids = [f"x{i}" for i in range(rng.randint(2, 5))]
        args = [_rand_arg(i, rng) for i in ids]
        atk = _random_attacks(ids, rng)
        af = ArgumentationFramework(args, atk, reorder=_max_gap)
        D = set(af.defeats())
        mutual = {frozenset((i, j)) for (i, j) in atk if (j, i) in set(atk)}
        # Prop 3
        for pr in mutual:
            i, j = tuple(pr)
            assert (i, j) in D or (j, i) in D
        # Prop 4
        g = af.to_digraph("defeat")
        pref, stab = _pref_stable(g)
        for ext in [grounded_extension(g), *pref, *stab]:
            for pr in mutual:
                i, j = tuple(pr)
                assert not (i in ext and j in ext)
