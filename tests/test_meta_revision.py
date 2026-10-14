"""Conflict-revision operators (reconcile/supersede) and resolution-operator ordering.

NOTE ON FRAMING (see theory/meta_separation.md): this is NOT a second
"separation theorem". reconcile is an attack-REVISION operator (it deletes an attack),
not an argument-preference operator; two external reviews correctly judged the
"separation" framing unfair. What these tests pin down is the narrow, honest fact
that conflict revision is not reducible to an argument-preorder defeat operator,
plus the (descriptive) non-commutativity of the reconcile_first vs supersede_first mode ordering.
The paper's headline separation is the head-reordering intransitivity.
"""
from __future__ import annotations

import itertools

import networkx as nx

from fulcrum.defeat import ArgumentationFramework
from fulcrum.meta_controller import RECONCILE_FIRST, SUPERSEDE_FIRST
from fulcrum.preference import Preferred
from fulcrum.semantics import (
    grounded_extension,
    preferred_extensions,
    stable_extensions,
)
from fulcrum.structures import Argument, PreferenceHead

_A = Argument("a", "p", head_scores={PreferenceHead.SOURCE: 1.0})
_B = Argument("b", "q", head_scores={PreferenceHead.SOURCE: 2.0})
_MUTUAL = [("a", "b"), ("b", "a")]
_RECON = lambda x, y: True  # noqa: E731
_ABRO = lambda x, y: Preferred.RIGHT if x.aid == "a" else Preferred.LEFT  # noqa: E731  b abrogates a


def _af(order):
    return ArgumentationFramework([_A, _B], _MUTUAL, mode_order=order,
                                  reconcilable=_RECON, abrogates=_ABRO)


def test_reconcile_dissolves_conflict_both_accepted():
    """reconcile revises the attack graph (deletes the edge) -> both accepted."""
    g = _af(RECONCILE_FIRST).to_digraph("defeat")
    assert grounded_extension(g) == {"a", "b"}
    assert [set(e) for e in preferred_extensions(g)] == [{"a", "b"}]


def test_mode_ordering_is_non_commutative():
    """Descriptive: reconcile_first (reconcile first) reconciles; supersede_first (supersede first) abrogates."""
    gs = _af(RECONCILE_FIRST).to_digraph("defeat")
    gh = _af(SUPERSEDE_FIRST).to_digraph("defeat")
    assert grounded_extension(gs) == {"a", "b"}
    assert grounded_extension(gh) == {"b"}


def test_reconcile_not_reducible_to_argument_preorder_defeat():
    """The narrow, honest fact: no argument-preorder defeat operator (one that
    never deletes both directions of a mutual attack -- Phase-3 Prop 3) yields
    joint acceptance. So reconcile (which does) lies outside that class. This is
    'conflict revision exceeds preference-based defeat', NOT a separation of the
    fulcrum *preference* calculus from single-ordering frameworks."""
    names = ["a", "b"]

    def preorders(n):
        e = range(n)
        off = [(i, j) for i in e for j in e if i != j]
        for bits in itertools.product((0, 1), repeat=len(off)):
            rel = {(i, i) for i in e} | {p for p, bt in zip(off, bits) if bt}
            if all(not ((i, j) in rel and (j, k) in rel and (i, k) not in rel)
                   for i in e for j in e for k in e):
                yield rel

    def defeat_graph(rel):
        g = nx.DiGraph()
        g.add_nodes_from(names)
        gij, gji = (0, 1) in rel, (1, 0) in rel
        if gij and not gji:
            g.add_edge("a", "b")
        elif gji and not gij:
            g.add_edge("b", "a")
        else:
            g.add_edge("a", "b")
            g.add_edge("b", "a")
        return g

    both = {"a", "b"}
    for rel in preorders(2):
        g = defeat_graph(rel)
        exts = [set(e) for e in preferred_extensions(g)] + \
               [set(e) for e in stable_extensions(g)]
        assert both not in exts
