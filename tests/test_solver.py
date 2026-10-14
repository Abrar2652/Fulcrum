"""Solver adapter tests. The clingo cross-validation is gated: it runs only if
clingo is installed (otherwise skipped), so the suite is green with or without
the optional solver dependency.
"""
from __future__ import annotations

import networkx as nx
import pytest

from fulcrum import solver
from fulcrum.semantics import stable_extensions


def _directed_3cycle() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    return g


def _two_cycle_with_source() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "a"), ("d", "a")])
    return g


def test_asp_serialization_wellformed():
    g = _two_cycle_with_source()
    asp = solver.af_to_asp(g)
    assert 'arg("a").' in asp and 'att("d","a").' in asp


def test_solve_raises_or_matches_without_breaking():
    """If clingo is absent, solve() raises a clear message (documented fallback);
    if present, stable extensions must match the built-in enumerator."""
    g = _two_cycle_with_source()
    if not solver.clingo_available():
        with pytest.raises(RuntimeError, match="clingo not installed"):
            solver.solve(g, "stable")
    else:
        got = {frozenset(e) for e in solver.solve(g, "stable")}
        want = {frozenset(e) for e in stable_extensions(g)}
        assert got == want


@pytest.mark.skipif(not solver.clingo_available(), reason="clingo not installed")
@pytest.mark.parametrize("g", [_directed_3cycle(), _two_cycle_with_source()])
def test_clingo_matches_bruteforce_stable(g):
    assert solver.cross_check(g, stable_extensions(g))
