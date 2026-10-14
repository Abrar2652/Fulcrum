"""THE GATE (the project design notes Phase 2) -- executable form of the separation conjecture.

These tests encode the central open research question:

    Does the context-dependent, multi-tier fulcrum preference operator produce
    conflict-resolution behavior that NO single fixed ordering
    (LexicographicCollapse, and by extension ASPIC+/ABA+/VAF) can reproduce?

They are marked xfail/skip ON PURPOSE. A green test suite must NOT imply the
conjecture is proven. Do the math in theory/separation_theorem.md; these
tests are where a *machine-checkable* separating example lands once found,
and where a *refutation* (outcome B) is recorded if the collapse baseline
turns out to match.

DO NOT edit these to pass by weakening the baseline. That is the exact
self-deception the project design notes Trap #2 warns against. The baseline must stay strong.
"""

from __future__ import annotations

import itertools

import pytest

from fulcrum.baselines.single_ordering import LexicographicCollapse
from fulcrum.preference import weigh_compare
from fulcrum.structures import HEAD_ORDER, Argument, PreferenceHead


def _context_reorder_policy(a: Argument, b: Argument, base):
    """A candidate context-conditioned policy: if either argument carries the
    'text_critical' context tag, promote SPECIFICITY to the front. This is the kind
    of dynamic reordering that, IF it cannot be matched by any single fixed
    ordering across a set of pairs, witnesses separation.
    """

    if "text_critical" in a.context or "text_critical" in b.context:
        return (PreferenceHead.SPECIFICITY, PreferenceHead.SOURCE,
                PreferenceHead.MODAL, PreferenceHead.CORROBORATION)
    return base


@pytest.mark.gate
@pytest.mark.xfail(
    reason="SEPARATION CONJECTURE UNPROVEN. This test asserts that no single "
    "fixed head-ordering reproduces the context-conditioned operator's verdicts "
    "on the sample. Establishing/refuting this is a Phase-2 deliverable; until "
    "then it is expected to xfail. See theory/separation_theorem.md.",
    strict=False,
)
def test_context_policy_not_matched_by_any_single_ordering():
    """Search all fixed head orderings; assert none matches the
    context-conditioned operator on a crafted sample. If this ever PASSES
    with a genuinely strong construction, that is positive evidence for
    separation -- promote it into the proof. If it cannot be made to pass
    because some fixed ordering always matches, that is evidence for outcome
    (B); record it in theory/self_refutation_log.md.
    """

    # Crafted sample. IMPORTANT: this deliberately uses DISTINCT score
    # vectors across the two pairs, so that a fixed ordering has a genuine
    # chance to reproduce the verdicts. (A degenerate sample where both pairs
    # share identical score vectors would make separation look real for a
    # trivial reason -- a fixed ordering must return the same verdict on
    # identical inputs, so tagging one pair 'context' would 'beat' every
    # fixed ordering by construction. That is an ARTIFACT, not separation, and
    # a reviewer would reject it instantly. We avoid that trap here.)
    p1a = Argument("p1a", "x", head_scores={PreferenceHead.SOURCE: 2.0, PreferenceHead.SPECIFICITY: 1.0})
    p1b = Argument("p1b", "y", head_scores={PreferenceHead.SOURCE: 1.0, PreferenceHead.SPECIFICITY: 3.0})
    p2a = Argument("p2a", "x", head_scores={PreferenceHead.SOURCE: 5.0, PreferenceHead.SPECIFICITY: 4.0},
                   context=frozenset({"text_critical"}))
    p2b = Argument("p2b", "y", head_scores={PreferenceHead.SOURCE: 4.0, PreferenceHead.SPECIFICITY: 1.0},
                   context=frozenset({"text_critical"}))
    sample = [(p1a, p1b), (p2a, p2b)]

    # Verdicts of the context-conditioned operator.
    target = [weigh_compare(a, b, reorder=_context_reorder_policy) for a, b in sample]

    # Try every fixed permutation of heads as a single-ordering baseline.
    matched_by_some_fixed = False
    for perm in itertools.permutations(HEAD_ORDER):
        baseline = LexicographicCollapse(ordering=perm)
        verdicts = [baseline.prefer(a, b) for a, b in sample]
        if verdicts == target:
            matched_by_some_fixed = True
            break

    # The conjecture says NO fixed ordering matches. So we assert that.
    # NOTE: with this particular tiny sample a fixed ordering DOES match
    # (that's why the test xfails). The research task is to determine whether
    # a richer construction escapes ALL fixed orderings, or to prove it
    # cannot.
    assert not matched_by_some_fixed, (
        "A single fixed ordering reproduced the context-conditioned verdicts "
        "on this sample -> no separation witnessed here. Strengthen the "
        "construction or record outcome (B)."
    )


@pytest.mark.gate
@pytest.mark.skip(
    reason="SUPERSEDED. reconcile/supersede are now implemented (meta_controller.py) and "
    "the 'meta-preference separation' idea was DEMOTED to a conflict-revision "
    "case study after external review (theory/meta_separation.md). The honest "
    "residual fact lives in tests/test_meta_revision.py; this is not a second "
    "separation of the same kind as the head-reordering result."
)
def test_meta_ordering_separation():
    """Superseded placeholder; see tests/test_meta_revision.py and
    theory/meta_separation.md for the corrected (demoted) treatment."""

    raise AssertionError("unreachable: skipped")
