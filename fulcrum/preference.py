"""The fulcrum preference operator.

This is the formal heart of the contribution. Everything the paper claims as
"new machinery" lives here and in ``meta_controller.py``. Read the project design notes
Phase 2 (THE GATE) before modifying this file.

What is (claimed to be) novel
------------------------------
Mainstream structured-argumentation frameworks reduce preference to a SINGLE
binary ordering over arguments/assumptions (Dung: none at all; ASPIC+: one
<=; ABA+: one ordering over assumptions; VAF: one value-ordering per
audience). Here, preference is:

  1. MULTI-TIER   : an ordered family of heads (source_reliability, specificity, modal_force, corroboration),
                    compared lexicographically by default.
  2. HETEROGENEOUS: each head can carry its own comparison semantics.
  3. CONTEXT-DEPENDENT: the *priority ordering over heads* can be reordered
                    as a function of argument/context features. This dynamic
                    reordering is the primary witness for the strict-extension
                    conjecture (see theory/separation_theorem.md).
  4. STRENGTH-GATED: fulcrum is defined ONLY on pairs of comparable (PROBABLE)
                    strength. A CERTAIN indicant is never weighed against a
                    PROBABLE one by fulcrum (that is handled upstream, by the
                    meta-controller, via supersede/reconcile).

IMPORTANT (honesty gate): it is an OPEN question whether (3) genuinely
escapes existing prioritized/lexicographic frameworks or is encodable in
them. This module implements the operator; it does NOT prove the separation.
The proof-or-refutation is a Phase-2 deliverable in theory/. Do not mistake a
working implementation for a proven contribution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum

from fulcrum.structures import HEAD_ORDER, Argument, EpistemicStrength, PreferenceHead


class Preferred(Enum):
    """Result of comparing two arguments under fulcrum."""

    LEFT = "left"  # a is preferred (arjah) to b
    RIGHT = "right"  # b is preferred to a
    INCOMPARABLE = "incomparable"  # equipollence: equipollence, no preponderance


# A head ordering is a permutation (priority high->low) of the heads.
HeadOrdering = Sequence[PreferenceHead]

# A context reordering policy maps a pair of arguments to the head ordering
# that applies to *that* comparison. The default policy ignores context and
# returns the static ordering (recovering a fixed lexicographic rule).
ReorderPolicy = Callable[[Argument, Argument, HeadOrdering], HeadOrdering]


def static_policy(a: Argument, b: Argument, base: HeadOrdering) -> HeadOrdering:
    """Context-insensitive policy: always use the base ordering.

    Under this policy the operator reduces to a fixed lexicographic
    comparison over heads. The separation conjecture asserts this reduced
    form is strictly weaker than the context-sensitive form; that claim is
    tested (and must be proven/refuted) in Phase 2.
    """

    return base


def _compare_head(a: Argument, b: Argument, head: PreferenceHead) -> Preferred:
    """Default per-head comparison: higher score preferred.

    Heterogeneous semantics (novelty point 2) can be added by dispatching on
    ``head`` here. Kept as a single rule for now; extension points documented
    so a reviewer can see where heterogeneity would enter.
    """

    sa, sb = a.score(head), b.score(head)
    if sa > sb:
        return Preferred.LEFT
    if sb > sa:
        return Preferred.RIGHT
    return Preferred.INCOMPARABLE


def weigh_compare(
    a: Argument,
    b: Argument,
    base_ordering: HeadOrdering = HEAD_ORDER,
    reorder: ReorderPolicy = static_policy,
) -> Preferred:
    """Compare two arguments by preponderance (fulcrum).

    Parameters
    ----------
    a, b:
        The arguments to weigh. Both MUST be of PROBABLE (probable) strength;
        weighing is undefined on certain indicants (raises ValueError). This
        enforces the strength gate faithfully rather than silently.
    base_ordering:
        The default priority ordering over heads (high -> low).
    reorder:
        Context-conditioning policy. ``static_policy`` recovers fixed
        lexicographic order; a non-trivial policy realizes dynamic reordering.

    Returns
    -------
    Preferred.LEFT / RIGHT / INCOMPARABLE.

    Notes
    -----
    Lexicographic resolution: walk the (possibly reordered) heads high->low;
    the first head that discriminates decides. If none discriminates, the
    pair is INCOMPARABLE (equipollence), which upstream triggers a fallback
    resolution mode (see meta_controller.py).
    """

    if a.strength is EpistemicStrength.CERTAIN or b.strength is EpistemicStrength.CERTAIN:
        raise ValueError(
            "weigh_compare is defined only on PROBABLE (probable) indicants; "
            "certain (certain) indicants must be handled by the meta-controller "
            "(supersede/reconcile), not by preponderance. See structures.py docstring."
        )

    ordering = reorder(a, b, base_ordering)

    for head in ordering:
        result = _compare_head(a, b, head)
        if result is not Preferred.INCOMPARABLE:
            return result
    return Preferred.INCOMPARABLE


def is_static(reorder: ReorderPolicy, sample: Sequence[tuple[Argument, Argument]]) -> bool:
    """Empirical check: does ``reorder`` behave as the static policy on a
    sample of pairs? Used by the recovery-to-baseline sanity tests. Not a
    proof of equivalence, only a finite witness of divergence when it returns
    False.
    """

    for a, b in sample:
        if list(reorder(a, b, HEAD_ORDER)) != list(HEAD_ORDER):
            return False
    return True
