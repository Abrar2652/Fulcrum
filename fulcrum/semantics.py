"""Argumentation semantics over the defeat relation.

We build on STANDARD Dung semantics rather than inventing our own, exactly as
the project design notes Phase 3 requires ("reviewers trust that"). The novelty lives in how
preference shapes the *defeat* relation (defeat.py); the acceptance semantics
applied on top are classical.

Implemented: the grounded extension (unique, always exists, computable by a
simple fixpoint) as the default skeptical semantics. Preferred/stable
enumeration is left as a documented extension point -- for the skeleton we
provide the grounded extension plus admissibility checking, which suffice for
recovery tests and small examples.

For serious experiments, wrap a vetted solver (e.g. an ASPARTIX/clingo
encoding) rather than scaling this pure-Python version. See the project design notes Phase 4.
"""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx


def _attackers(g: nx.DiGraph, node: str) -> set[str]:
    return set(g.predecessors(node))


def is_conflict_free(g: nx.DiGraph, s: set[str]) -> bool:
    for a in s:
        for b in s:
            if g.has_edge(a, b):
                return False
    return True


def defends(g: nx.DiGraph, s: set[str], node: str) -> bool:
    """S defends node iff every attacker of node is attacked by some member
    of S."""

    for attacker in _attackers(g, node):
        if not any(g.has_edge(d, attacker) for d in s):
            return False
    return True


def is_admissible(g: nx.DiGraph, s: set[str]) -> bool:
    if not is_conflict_free(g, s):
        return False
    return all(defends(g, s, node) for node in s)


def grounded_extension(g: nx.DiGraph) -> set[str]:
    """Compute the grounded extension via the characteristic-function
    fixpoint: start from the empty set, repeatedly add all arguments defended
    by the current set, until a fixpoint is reached. Always terminates
    (monotone on a finite lattice) and is unique.
    """

    nodes = set(g.nodes())
    current: set[str] = set()
    while True:
        nxt = {n for n in nodes if defends(g, current, n)}
        # grounded is the least fixpoint of the defense operator; union keeps
        # it monotone and convergent
        nxt |= current
        if nxt == current:
            return current
        current = nxt


def acceptance(g: nx.DiGraph, node: str) -> bool:
    """Skeptical (grounded) acceptance of a single argument."""

    return node in grounded_extension(g)


def _all_subsets(nodes: list[str]) -> Iterator[set[str]]:
    from itertools import combinations

    for r in range(len(nodes) + 1):
        for combo in combinations(nodes, r):
            yield set(combo)


def admissible_sets(g: nx.DiGraph) -> list[set[str]]:
    """Enumerate all admissible sets (brute force; for small AFs / tests only).

    For serious experiments wrap a vetted solver (the project design notes Phase 4). This
    exists so preferred/stable are available in the library rather than being
    reimplemented ad hoc in tests and scripts.
    """

    return [s for s in _all_subsets(list(g.nodes())) if is_admissible(g, s)]


def preferred_extensions(g: nx.DiGraph) -> list[set[str]]:
    """Preferred extensions = maximal (w.r.t. ⊆) admissible sets."""

    adm = admissible_sets(g)
    return [s for s in adm if not any(s < t for t in adm)]


def stable_extensions(g: nx.DiGraph) -> list[set[str]]:
    """Stable extensions = conflict-free sets that attack every outside node."""

    nodes = list(g.nodes())
    out = []
    for s in _all_subsets(nodes):
        if not is_conflict_free(g, s):
            continue
        if all(any(g.has_edge(m, o) for m in s) for o in set(nodes) - s):
            out.append(s)
    return out
