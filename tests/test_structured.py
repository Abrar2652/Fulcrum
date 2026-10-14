"""Phase-4 structured layer (fulcrum/structured.py) + empirical Theorem B.

Golden tests plus a postulate battery: with (a) certain encoded as strict-and-firm,
(b) no multi-antecedent strict rules, an INTRANSITIVE fulcrum preponderance still
yields extensions satisfying the Caminada-Amgoud postulates (direct + indirect
consistency, closure). This is the safe-regime discharge of
theory/postulates_structured.md, Theorem B.
"""
from __future__ import annotations

import random

import pytest

from fulcrum.semantics import (
    grounded_extension,
    preferred_extensions,
    stable_extensions,
)
from fulcrum.structured import (
    Literal,
    Rule,
    StructuredAF,
    Theory,
    build_arguments,
    check_postulates,
)


def _max_gap(a, b, base):
    hs = list(base)
    hs.sort(key=lambda h: (-abs(a.score(h) - b.score(h)), base.index(h)))
    return tuple(hs)


def _rulings_theory(vectors):
    """n mutually-exclusive rulings r0..r{n-1} as ordinary (probable) premises with
    the given head-score vectors; every pair declared contrary."""
    lits = [Literal(f"r{i}") for i in range(len(vectors))]
    contraries = {frozenset({lits[i], lits[j]})
                  for i in range(len(lits)) for j in range(i + 1, len(lits))}
    return Theory(ordinary=dict(zip(lits, vectors)), contraries=contraries), lits


def _conc_map(args):
    return {a.aid: str(a.conc) for a in args}


def test_witness_reproduces_directed_cycle_structurally():
    th, _ = _rulings_theory([(0, 1, 2), (1, 2, 0), (2, 0, 1)])
    args = build_arguments(th)
    saf = StructuredAF(args, th, reorder=_max_gap)
    cm = _conc_map(args)
    defeats = {(cm[s], cm[t]) for s, t in saf.defeats()}
    assert defeats == {("r0", "r1"), ("r1", "r2"), ("r2", "r0")}  # directed 3-cycle
    g = saf.to_digraph()
    assert grounded_extension(g) == set()
    assert [e for e in preferred_extensions(g)] == [set()]
    assert stable_extensions(g) == []


def test_theorem_b_postulates_hold_under_intransitivity():
    th, _ = _rulings_theory([(0, 1, 2), (1, 2, 0), (2, 0, 1)])
    args = build_arguments(th)
    saf = StructuredAF(args, th, reorder=_max_gap)
    g = saf.to_digraph()
    for ext in preferred_extensions(g) + stable_extensions(g) + [grounded_extension(g)]:
        flags = check_postulates(ext, args, th)
        assert all(flags.values()), flags


def test_certain_dominates_and_is_accepted():
    q, z = Literal("q"), Literal("z")
    th = Theory(axioms={q}, ordinary={z: (9, 9, 9)}, contraries={frozenset({q, z})})
    args = build_arguments(th)
    saf = StructuredAF(args, th, reorder=_max_gap)
    cm = _conc_map(args)
    defeats = {(cm[s], cm[t]) for s, t in saf.defeats()}
    assert ("q", "z") in defeats          # certain defeats conflicting probable
    assert ("z", "q") not in defeats      # and is never defeated by it (Def. 18-1)
    assert {cm[x] for x in grounded_extension(saf.to_digraph())} == {"q"}


def test_safe_regime_rejects_multi_antecedent_strict_rule():
    th = Theory(strict=[Rule("conj", frozenset({Literal("a"), Literal("b")}),
                             Literal("ab"), defeasible=False)])
    with pytest.raises(ValueError, match="multi-antecedent"):
        th.check_safe_regime()
    with pytest.raises(ValueError):
        build_arguments(th)


def test_indirect_consistency_via_strict_closure():
    # r0 wins a 2-way conflict, and a unary strict rule derives s from r0 whose
    # contrary ~s must not also be derivable in the extension.
    r0, s = Literal("r0"), Literal("s")
    th, _ = _rulings_theory([(3, 0, 0), (0, 0, 0)])  # r0 strictly stronger on source_reliability
    th.strict.append(Rule("r0_to_s", frozenset({r0}), s, defeasible=False))
    args = build_arguments(th)
    saf = StructuredAF(args, th, reorder=_max_gap)
    g = saf.to_digraph()
    for ext in preferred_extensions(g):
        flags = check_postulates(ext, args, th)
        assert flags["indirect_consistency"] and flags["direct_consistency"]


@pytest.mark.parametrize("seed", range(30))
def test_postulate_battery_safe_regime(seed):
    """Random safe-regime 'conflicting rulings' theories under the intransitive
    MAX-GAP policy: every extension must satisfy all postulates (Theorem B)."""
    rng = random.Random(1000 + seed)
    n = rng.randint(2, 4)
    vectors = [tuple(rng.randint(0, 3) for _ in range(3)) for _ in range(n)]
    th, lits = _rulings_theory(vectors)
    # optional unary strict continuations (safe: single antecedent)
    for k in range(rng.randint(0, 2)):
        src = rng.choice(lits)
        th.strict.append(Rule(f"s{k}", frozenset({src}), Literal(f"c{k}"), defeasible=False))
    args = build_arguments(th)
    saf = StructuredAF(args, th, reorder=_max_gap)
    g = saf.to_digraph()
    exts = preferred_extensions(g) + stable_extensions(g) + [grounded_extension(g)]
    for ext in exts:
        flags = check_postulates(ext, args, th)
        assert all(flags.values()), (seed, vectors, flags)
