"""Golden conflict-resolution examples (Phase 4).

Each example is a small, hand-worked conflict that the calculus should resolve
*by a specific criterion* (or by the strength gate, or leave unresolved). They pin
the intended behaviour of the four criteria (source-reliability, specificity,
modal-force, corroboration), the certain/probable strength gate, equipollence, and
the context-dependent (intransitive) witness. Claims are abstract placeholders
(p, q, …). Where a criterion is naturally qualitative (e.g. a fixed asymmetric
preference of one modal reading over another) we currently approximate it by a
scalar head-score and flag the simplification (see the design-decision notes). See
`theory/golden_examples.md` for the criterion ↔ formal-element ↔ role map.
"""
from __future__ import annotations

from dataclasses import dataclass

from fulcrum.preference import HeadOrdering, ReorderPolicy, static_policy
from fulcrum.structured import Literal, Rule, StructuredAF, Theory, build_arguments
from fulcrum.structures import HEAD_ORDER, Argument, PreferenceHead

# index of each head in a head-score vector (HEAD_ORDER = source_reliability, specificity, modal_force, corroboration)
_IDX = {h: i for i, h in enumerate(HEAD_ORDER)}


def _vec(**heads: float) -> tuple[float, ...]:
    v = [0.0] * len(HEAD_ORDER)
    for name, val in heads.items():
        v[_IDX[PreferenceHead[name]]] = float(val)
    return tuple(v)


@dataclass
class Example:
    name: str
    theory: Theory
    saf: StructuredAF
    expect_winner: str | None  # conclusion (str) expected in grounded, or None for equipollence
    source: str                # which criterion this example illustrates


def _two_ruling_conflict(a_vec: tuple[float, ...], b_vec: tuple[float, ...], reorder: ReorderPolicy = static_policy) -> tuple[Theory, StructuredAF]:
    """Two probable arguments for contrary rulings p, q (via defeasible rules from a
    shared premise), differing on the given head-score vectors."""
    base, p, q = Literal("base"), Literal("p"), Literal("q")
    th = Theory(
        axioms={base},
        defeasible=[
            Rule("d_p", frozenset({base}), p, defeasible=True, head_scores=a_vec),
            Rule("d_q", frozenset({base}), q, defeasible=True, head_scores=b_vec),
        ],
        contraries={frozenset({p, q})},
    )
    return th, StructuredAF(build_arguments(th), th, reorder=reorder)


def by_source_reliability() -> Example:
    """Prefer by source reliability: of two conflicting probable reports, prefer the one with
    the more reliable source. p stronger on source_reliability."""
    th, saf = _two_ruling_conflict(_vec(SOURCE=2), _vec(SOURCE=1))
    return Example("by_source_reliability", th, saf, "p", "source-reliability")


def by_specificity() -> Example:
    """Prefer by specificity: prefer the more specific (specific) text to the general
    (general). p is the more specific text (higher specificity), equal on source_reliability."""
    th, saf = _two_ruling_conflict(_vec(SOURCE=1, SPECIFICITY=2), _vec(SOURCE=1, SPECIFICITY=1))
    return Example("by_specificity", th, saf, "p", "specificity (specific > general)")


def by_modal_force() -> Example:
    """Prefer by modal force: prohibition preferred to mere
    permissibility. APPROXIMATION: encoded as a higher modal_force score for the
    prohibition ruling; a real modal preference is qualitative/asymmetric — see
    the design-decision notes and the ❌TODO in fulcrum_calculus.md §2."""
    th, saf = _two_ruling_conflict(_vec(SOURCE=1, SPECIFICITY=1, MODAL=2),
                                   _vec(SOURCE=1, SPECIFICITY=1, MODAL=1))
    return Example("by_modal_force", th, saf, "p", "modal-force (asymmetric preference; scalar approx)")


def by_corroboration() -> Example:
    """Prefer by corroboration: prefer the report corroborated by an external
    indicant. p corroborated (higher corroboration)."""
    th, saf = _two_ruling_conflict(_vec(SOURCE=1, SPECIFICITY=1, MODAL=1, CORROBORATION=2),
                                   _vec(SOURCE=1, SPECIFICITY=1, MODAL=1, CORROBORATION=1))
    return Example("by_corroboration", th, saf, "p", "corroboration")


def strength_gate_certain() -> Example:
    """The certain/probable gate: a certain (certain) indicant is not weighed against a
    probable (probable) one — it dominates. q is certain (axiom → strict-and-firm);
    p is probable with even a very high head-score. q must win regardless."""
    p, q = Literal("p"), Literal("q")
    th = Theory(
        axioms={q},  # certain: strict-and-firm argument for q
        defeasible=[Rule("d_p", frozenset({Literal("base")}), p, defeasible=True,
                         head_scores=_vec(SOURCE=9, SPECIFICITY=9, MODAL=9, CORROBORATION=9))],
        ordinary={Literal("base"): _vec()},
        contraries={frozenset({p, q})},
    )
    return Example("strength_gate_certain", th, StructuredAF(build_arguments(th), th),
                   "q", "certain/probable gate (certain not weighed by preponderance)")


def equipollence() -> Example:
    """equipollence (equipollence): two probable arguments equal on every head — no
    preponderance. Both attacks survive → the conflict is unresolved (grounded
    empty). Illustrates that the calculus does not manufacture a winner."""
    th, saf = _two_ruling_conflict(_vec(SOURCE=1, SPECIFICITY=1), _vec(SOURCE=1, SPECIFICITY=1))
    return Example("equipollence", th, saf, None, "equipollence (equipollence → no preponderance)")


def context_dependent_reordering() -> Example:
    """The NOVEL structural case (the separation witness): three mutually
    exclusive rulings whose deciding head rotates with the conflict, yielding an
    intransitive preponderance no single ordering reproduces. Under MAX-GAP the
    defeat graph is the directed 3-cycle; grounded empty (collective equipollence)."""
    def max_gap(a: Argument, b: Argument, base: HeadOrdering) -> HeadOrdering:
        hs = list(base)
        hs.sort(key=lambda h: (-abs(a.score(h) - b.score(h)), base.index(h)))
        return tuple(hs)

    r0, r1, r2 = Literal("r0"), Literal("r1"), Literal("r2")
    th = Theory(
        ordinary={r0: _vec(SOURCE=0, SPECIFICITY=1, MODAL=2),
                  r1: _vec(SOURCE=1, SPECIFICITY=2, MODAL=0),
                  r2: _vec(SOURCE=2, SPECIFICITY=0, MODAL=1)},
        contraries={frozenset({r0, r1}), frozenset({r1, r2}), frozenset({r0, r2})},
    )
    return Example("context_dependent_reordering", th,
                   StructuredAF(build_arguments(th), th, reorder=max_gap),
                   None, "context-dependent head reordering (intransitive; the separation witness)")


ALL_EXAMPLES = [
    by_source_reliability, by_specificity, by_modal_force, by_corroboration,
    strength_gate_certain, equipollence, context_dependent_reordering,
]
