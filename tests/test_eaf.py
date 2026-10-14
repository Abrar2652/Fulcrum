"""EAF reification of reconcile/supersede (fulcrum/eaf.py). Rehabilitates the conflict-
revision material as a legitimate second-order encoding (Modgil EAF, 2009):
reconcile/supersede become meta-arguments attacking base attacks; mu is an object-level
preference over them. The reconcile_first (reconcile) outcome -- both a,b accepted -- requires
second-order power and is not reproducible by any FIRST-ORDER single-ordering
framework on the same base AF. See theory/meta_separation.md.
"""
from __future__ import annotations

import itertools

import networkx as nx

from fulcrum.eaf import conflict_witness_eaf
from fulcrum.meta_controller import RECONCILE_FIRST, SUPERSEDE_FIRST
from fulcrum.semantics import (
    grounded_extension,
    preferred_extensions,
    stable_extensions,
)


def test_reconcile_first_reconciles_supersede_first_abrogates():
    gs = conflict_witness_eaf(RECONCILE_FIRST).to_digraph()
    gh = conflict_witness_eaf(SUPERSEDE_FIRST).to_digraph()
    assert grounded_extension(gs) == {"a", "b"}   # reconcile: both retained
    assert grounded_extension(gh) == {"b"}         # supersede: b abrogates a


def test_mode_preference_selects_meta_argument():
    """mu is the object-level meta-preference: it picks which meta-argument
    (reconcile vs supersede) is accepted."""
    assert [m.aid for m in conflict_witness_eaf(RECONCILE_FIRST).accepted_meta()] == ["reconcile"]
    assert [m.aid for m in conflict_witness_eaf(SUPERSEDE_FIRST).accepted_meta()] == ["supersede"]


def test_reconcile_first_outcome_needs_second_order_power():
    """No first-order single preorder over {a,b} on the base mutual-attack AF
    accepts both -> the reconcile outcome genuinely requires the second-order (EAF)
    machinery. This is the legitimate, honest separation (first-order reference
    set vs second-order encoding), NOT a claim of beating EAFs."""
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
