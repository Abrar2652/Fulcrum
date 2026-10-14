"""Adapters that WRAP established argumentation implementations (not strawmen).

Rationale (addresses the pre-registration's prerequisite (c)): on the synthetic
*resolution* task the calculus's single-ordering baseline is already faithful to
ASPIC+/ABA+ — the separation theorem shows those frameworks reduce a fixed
mutual-conflict instance to a single (pre)order (that is the whole point). Wrapped
implementations matter on the *real* NL benchmark (BoardGameQA), where the full
structured pipeline is the competitor.

SCAFFOLD: guarded imports; NOT exercised in the offline dev environment. Install
the wrapped libraries on a networked box (see experiments/RUNBOOK.md). API call
sites marked `# VERIFY` should be checked against the installed version.

Wrapped / cited:
  - PyArg (py-arg), Utrecht — ASPIC+ / ABA / abstract AF. Prakken et al.
  - clingo + ASPARTIX encodings (Egly, Gaggl & Woltran 2010) — Dung semantics,
    reused via fulcrum.solver.
  - Prioritized default logic / preferred subtheories (Brewka) — ASP encoding.
"""
from __future__ import annotations

from collections.abc import Sequence


class _MissingBackend(RuntimeError):
    pass


class PyArgAspicSolver:
    """Wrap PyArg's ASPIC+ engine. Given a structured theory (axioms, ordinary
    premises, strict/defeasible rules, and a preference over defeasible rules),
    return the grounded/preferred justified conclusions.

    This is the faithful ASPIC+ baseline for the real-data pipeline. The NL→theory
    formalization step (LLM-ASPIC+ style) is a separate module; this adapter solves
    an already-formalized theory.
    """

    def __init__(self, semantics: str = "grounded") -> None:
        self.semantics = semantics

    def _require(self) -> object:
        try:
            import py_arg
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise _MissingBackend(
                "PyArg not installed. `pip install py-arg` (Utrecht) and see RUNBOOK.md."
            ) from e
        return py_arg

    def justified_conclusions(self, theory: object) -> set[str]:  # pragma: no cover
        """`theory` is a PyArg ArgumentationTheory (built by the caller / the NL
        formalizer). Returns the set of justified conclusion strings."""
        self._require()
        # VERIFY against installed py-arg: module paths/class names have shifted
        # across versions. Typical shape:
        #   from py_arg.aspic.classes.argumentation_theory import ArgumentationTheory
        #   from py_arg.abstract_argumentation.semantics import get_grounded_extension
        # 1) generate arguments + attacks from `theory`
        # 2) build the abstract AF, apply `self.semantics`
        # 3) collect Conc(A) for accepted A
        raise NotImplementedError(
            "Fill against the installed PyArg API (see the two import lines above); "
            "kept explicit so a version mismatch fails loudly rather than silently."
        )


class AbaPlusSolver:
    """Wrap an ABA+ implementation (Cyras & Toni 2016). Same contract: a
    formalized ABA+ framework in, justified conclusions out."""

    def __init__(self, semantics: str = "grounded") -> None:
        self.semantics = semantics

    def justified_conclusions(self, framework: object) -> set[str]:  # pragma: no cover
        try:
            import py_arg  # noqa: F401  (PyArg also carries ABA)
        except ImportError as e:
            raise _MissingBackend("PyArg/ABA+ backend not installed; see RUNBOOK.md.") from e
        raise NotImplementedError("Fill against the installed ABA+ API; see RUNBOOK.md.")


def prioritized_default_logic_asp(rules: Sequence[tuple[str, str, int]]) -> str:
    """Emit a clingo (ASP) program for prioritized default logic / preferred
    subtheories (Brewka). `rules` are (name, literal, priority). Run with
    `fulcrum.solver`-style clingo invocation. This is the third baseline named in
    the falsification threshold.

    The encoding: higher-priority defaults are applied first; a default is blocked
    if its conclusion's complement is already derivable from higher priority.
    """
    lines = ["% prioritized default logic (preferred subtheories, Brewka)"]
    for name, lit, prio in rules:
        lines.append(f'default({name},"{lit}",{prio}).')
    lines += [
        "in(N) :- default(N,L,P), not blocked(N).",
        # blocked if a strictly-higher-priority default concludes the complement
        'blocked(N) :- default(N,L,P), default(M,C,Q), Q>P, in(M), complement(L,C).',
        "#show in/1.",
    ]
    return "\n".join(lines) + "\n"


# Dung-semantics baseline (grounded/stable/complete) is provided by
# fulcrum.solver (ASPARTIX/clingo). VAF reduces to a single value-ordering, i.e.
# the LexicographicCollapse baseline already in fulcrum.baselines.single_ordering.
