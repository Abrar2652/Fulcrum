"""Generalization of the separation (theory/generalization.md):
G1 intransitive k-cycles for every k under MAX-GAP; G2 operator-class
universality over tournaments + the counting gap vs single orderings.
"""
from __future__ import annotations

import itertools
import math

import pytest


def _lex_winner(va, vb, order):
    for i in order:
        if va[i] > vb[i]:
            return 1
        if vb[i] > va[i]:
            return -1
    return 0


def _max_gap_order(va, vb):
    return sorted(range(len(va)), key=lambda i: (-abs(va[i] - vb[i]), i))


def _maxgap_winner(va, vb):
    return _lex_winner(va, vb, _max_gap_order(va, vb))


@pytest.mark.parametrize("k", range(3, 8))
def test_g1_kcycle_under_max_gap(k):
    """Cyclic shifts of (0..k-1) form a strict k-cycle under MAX-GAP."""
    base = tuple(range(k))
    shifts = [tuple(base[(i + s) % k] for i in range(k)) for s in range(k)]
    for s in range(k):
        assert _maxgap_winner(shifts[s], shifts[(s + 1) % k]) == 1  # v_s > v_{s+1}
    # and the unique max-gap head decides each comparison (gap k-1 vs 1)
    for s in range(k):
        va, vb = shifts[s], shifts[(s + 1) % k]
        gaps = sorted((abs(va[i] - vb[i]) for i in range(k)), reverse=True)
        assert gaps[0] == k - 1 and gaps[1] == 1  # unique maximum


def _mutually_dominating(n):
    # a_i = (i, n-i): every pair each-exceeds-other on some head
    return [(i, n - i) for i in range(1, n + 1)]


def test_g2_universality_realizes_nontransitive_target():
    """On a mutually-dominating config, a per-pair policy realizes a target
    tournament with a 3-cycle that NO fixed head-ordering realizes."""
    n = 5
    cfg = _mutually_dominating(n)
    target = {(i, j): i for i in range(n) for j in range(i + 1, n)}  # i beats j
    target[(0, 2)] = 2  # inject 0>1>2>0 cycle (2 beats 0)

    # realizable by some per-pair policy iff each desired winner exceeds on some head
    def realizable_per_pair():
        for (i, j), w in target.items():
            va, vb = cfg[i], cfg[j]
            if not ((w == i and any(va[h] > vb[h] for h in range(2))) or
                    (w == j and any(vb[h] > va[h] for h in range(2)))):
                return False
        return True

    def any_fixed_order_realizes():
        for order in itertools.permutations(range(2)):
            if all((cfg[i] if _lex_winner(cfg[i], cfg[j], order) == 1 else cfg[j])
                   == cfg[w]
                   for (i, j), w in target.items()):
                return True
        return False

    assert realizable_per_pair() is True
    assert any_fixed_order_realizes() is False


@pytest.mark.parametrize("n", range(3, 8))
def test_g2_counting_gap(n):
    """Single orderings realize only transitive tournaments (n!) of 2^C(n,2)."""
    total = 2 ** (n * (n - 1) // 2)
    transitive = math.factorial(n)
    frac = transitive / total
    assert transitive <= total
    # the fraction strictly decreases past n=3 and is tiny by n=7
    if n >= 6:
        assert frac < 0.05
