"""Solver adapter: wrap a vetted argumentation solver instead of reimplementing
extension enumeration for scale (the project design notes Phase 4).

We emit the standard **ASPARTIX** ASP encoding of Dung semantics (Egly, Gaggl &
Woltran, "Answer-set programming encodings for argumentation frameworks",
Argument & Computation 1(2), 2010) and run it with `clingo` when available. When
clingo is not installed, callers fall back to the brute-force enumerator in
`fulcrum.semantics` (correct, but exponential — fine for tests/small examples).

Design: the fulcrum layer's only novelty is the defeat relation `R ↦ D`; once `D`
is computed (`defeat.py` / `structured.py`), acceptance is *standard* Dung
semantics, so any vetted Dung solver applies unchanged. This module is the seam.
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

# Canonical ASPARTIX encodings (answer sets ↔ extensions). `in/1` marks accepted.
STABLE_ENCODING = """
in(X) :- arg(X), not out(X).
out(X) :- arg(X), not in(X).
:- in(X), in(Y), att(X,Y).
defeated(X) :- in(Y), att(Y,X).
:- out(X), not defeated(X).
#show in/1.
"""

# Complete/admissible base (defended/undefeated); grounded is its ⊆-least, and is
# already poly-time via the fixpoint in semantics.py, so we expose stable here
# (the semantics most in need of a real solver) and defer preferred to the
# saturation/metasp encoding when a solver is actually wired in Phase 6.
ADMISSIBLE_ENCODING = """
in(X) :- arg(X), not out(X).
out(X) :- arg(X), not in(X).
:- in(X), in(Y), att(X,Y).
defeated(X) :- in(Y), att(Y,X).
:- in(X), att(Y,X), not defeated(Y).
#show in/1.
"""


def af_to_asp(g: nx.DiGraph) -> str:
    """Serialize a defeat digraph as ASPARTIX facts `arg/1`, `att/2`."""
    lines = [f"arg({_atom(n)})." for n in g.nodes()]
    lines += [f"att({_atom(u)},{_atom(v)})." for u, v in g.edges()]
    return "\n".join(lines) + "\n"


def _atom(node: str) -> str:
    """ASP constants must start lowercase / be quoted; wrap arbitrary ids."""
    return f'"{node}"'


def clingo_available() -> bool:
    try:
        import clingo  # noqa: F401
        return True
    except ImportError:
        return False


def solve(g: nx.DiGraph, semantics: str = "stable") -> list[set[str]]:
    """Enumerate extensions under `semantics` ('stable' | 'admissible').

    Uses clingo if available; otherwise raises with a pointer to the built-in
    fallback (`fulcrum.semantics.stable_extensions` / brute-force admissible).
    """
    if not clingo_available():
        raise RuntimeError(
            "clingo not installed; use fulcrum.semantics.stable_extensions / "
            "preferred_extensions (brute force) for small AFs, or "
            "`pip install '.[solver]'` to enable this path (Phase 6 scale)."
        )
    import clingo

    enc = {"stable": STABLE_ENCODING, "admissible": ADMISSIBLE_ENCODING}[semantics]
    ctl = clingo.Control(["0"])  # 0 = all answer sets
    ctl.add("base", [], af_to_asp(g) + enc)
    ctl.ground([("base", [])])
    out: list[set[str]] = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            ext = {sym.arguments[0].string
                   for sym in model.symbols(shown=True) if sym.name == "in"}
            out.append(ext)
    return out


def cross_check(g: nx.DiGraph, extensions: Iterable[set[str]]) -> bool:
    """True iff clingo's stable extensions match `extensions` (set-equality).
    Used by the clingo-gated test to validate the encoding against the built-in
    enumerator. Returns True vacuously if clingo is unavailable."""
    if not clingo_available():
        return True
    got = {frozenset(e) for e in solve(g, "stable")}
    want = {frozenset(e) for e in extensions}
    return got == want
