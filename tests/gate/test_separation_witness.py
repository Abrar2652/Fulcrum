"""THE GATE, resolved direction (the project design notes Phase 2) — machine-checked separation.

Unlike ``test_separation_conjecture.py`` (which uses exogenous context tags and
only defeats *fixed lexicographic* orderings — necessary but not sufficient),
this test encodes the ACTUAL separation result of theory/separation_theorem.md:

  An ENDOGENOUS (score-only, no context tags) reorder policy induces a strict
  preponderance 3-cycle a>b>c>a. No single PREORDER over {a,b,c} — the most
  general 'single ordering' any of Dung/ASPIC+/ABA+/VAF reduces to — reproduces
  it, at either the comparator level or the (credulous) acceptance level.

The baseline here is the STRONGEST possible (all 29 preorders on 3 elements),
so this test cannot be 'passed by weakening the baseline' (the project design notes Trap #2):
strengthening the baseline only makes 0-realizers harder to achieve, and we
still get 0. If this ever regresses to a nonzero count, the separation is false
and outcome (B) must be recorded.
"""
from __future__ import annotations

import itertools

import networkx as nx
import pytest

from fulcrum.defeat import ArgumentationFramework
from fulcrum.preference import Preferred, weigh_compare
from fulcrum.semantics import is_admissible, is_conflict_free
from fulcrum.structures import HEAD_ORDER, Argument

HEADS = HEAD_ORDER[:3]
NAMES = ["a", "b", "c"]


def _max_gap(a: Argument, b: Argument, base):
    hs = list(base)
    hs.sort(key=lambda h: (-abs(a.score(h) - b.score(h)), base.index(h)))
    return tuple(hs)


def _arg(aid, vec):
    return Argument(aid, aid, head_scores={h: float(v) for h, v in zip(HEADS, vec)})


A, B, C = _arg("a", (0, 1, 2)), _arg("b", (1, 2, 0)), _arg("c", (2, 0, 1))


def _cmp(x, y):
    return weigh_compare(x, y, base_ordering=HEAD_ORDER, reorder=_max_gap)


def _all_preorders(n):
    e = range(n)
    off = [(i, j) for i in e for j in e if i != j]
    for bits in itertools.product((0, 1), repeat=len(off)):
        rel = {(i, i) for i in e} | {p for p, b in zip(off, bits) if b}
        if all(not ((i, j) in rel and (j, k) in rel and (i, k) not in rel)
               for i in e for j in e for k in e):
            yield rel


def _strict_table(rel):
    t = {}
    for i, j in itertools.permutations(range(3), 2):
        gij, gji = (i, j) in rel, (j, i) in rel
        t[(i, j)] = (Preferred.LEFT if gij and not gji
                     else Preferred.RIGHT if gji and not gij
                     else Preferred.INCOMPARABLE)
    return t


def _fulcrum_table(args):
    return {(i, j): _cmp(args[i], args[j])
            for i, j in itertools.permutations(range(3), 2)}


def _preferred_stable(g):
    nodes = list(g.nodes())
    subs = [set(s) for r in range(len(nodes) + 1)
            for s in itertools.combinations(nodes, r)]
    adm = [s for s in subs if is_admissible(g, s)]
    pref = {frozenset(s) for s in adm if not any(s < t for t in adm)}
    stab = {frozenset(s) for s in subs
            if is_conflict_free(g, s)
            and all(any(g.has_edge(m, o) for m in s) for o in set(nodes) - s)}
    return pref, stab


def _preorder_defeat_graph(rel):
    g = nx.DiGraph()
    g.add_nodes_from(NAMES)
    for i, j in itertools.combinations(range(3), 2):
        x, y = NAMES[i], NAMES[j]
        gij, gji = (i, j) in rel, (j, i) in rel
        if gij and not gji:
            g.add_edge(x, y)
        elif gji and not gij:
            g.add_edge(y, x)
        else:
            g.add_edge(x, y)
            g.add_edge(y, x)
    return g


@pytest.mark.gate
def test_enumerator_is_correct():
    # #preorders on 3 labelled elements = 29 (OEIS A000798). Guards the baseline.
    assert sum(1 for _ in _all_preorders(3)) == 29


@pytest.mark.gate
def test_endogenous_policy_induces_strict_3cycle():
    assert _cmp(A, B) is Preferred.LEFT
    assert _cmp(B, C) is Preferred.LEFT
    assert _cmp(C, A) is Preferred.LEFT  # a > b > c > a, non-transitive


@pytest.mark.gate
def test_comparator_separation_no_preorder_realizes():
    tj = _fulcrum_table([A, B, C])
    realizers = [rel for rel in _all_preorders(3) if _strict_table(rel) == tj]
    assert realizers == [], (
        "A single preorder reproduced the fulcrum preponderance relation -> "
        "outcome (B). Record it in theory/self_refutation_log.md."
    )


@pytest.mark.gate
def test_extension_separation_credulous():
    af = ArgumentationFramework(
        [A, B, C],
        [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "a"), ("a", "c")],
        reorder=_max_gap,
    )
    g = af.to_digraph("defeat")
    assert set(g.edges()) == {("a", "b"), ("b", "c"), ("c", "a")}  # directed cycle
    pref_tj, stab_tj = _preferred_stable(g)
    assert pref_tj == {frozenset()}      # only the empty set is preferred
    assert stab_tj == set()              # no stable extension

    for rel in _all_preorders(3):
        gb = _preorder_defeat_graph(rel)
        assert set(gb.edges()) != {("a", "b"), ("b", "c"), ("c", "a")}
        assert (_preferred_stable(gb)) != (pref_tj, stab_tj)


@pytest.mark.gate
def test_grounded_does_not_separate_is_documented():
    # Honest scope guard: grounded of the bare directed cycle is empty, which the
    # all-indifferent preorder also yields. Separation is credulous-only.
    from fulcrum.semantics import grounded_extension
    af = ArgumentationFramework(
        [A, B, C],
        [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "a"), ("a", "c")],
        reorder=_max_gap,
    )
    assert grounded_extension(af.to_digraph("defeat")) == set()
    all_indiff = _preorder_defeat_graph({(i, i) for i in range(3)})
    assert grounded_extension(all_indiff) == set()  # a baseline matches -> no grounded sep
