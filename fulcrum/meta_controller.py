"""Meta-level conflict-resolution controller.

Implements the ordering over resolution operators for conflict of evidence
(conflict of evidence). Different orderings are possible:

  - reconcile_first : RECONCILE (reconcile) -> SUPERSEDE (abrogate) -> WEIGH (prefer)
  - supersede_first         : SUPERSEDE -> WEIGH -> RECONCILE

That the ordering is a *configurable parameter* (rather than fixed) is a
candidate novelty: a meta-preference over resolution operators. Standard
argumentation frameworks have no analog of selecting *among resolution
strategies* as a function of the case. See the project design notes Phase 2.

This module deliberately implements RECONCILE and SUPERSEDE as STUBS with explicit
NotImplementedError, because faithful definitions depend on modeling choices
that must be fixed with the human and logged in the design-decision notes. Only the
control flow (mode ordering + dispatch + fulcrum fallback) is provided, so the
skeleton runs and is testable without pre-committing contested semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fulcrum.preference import (
    HeadOrdering,
    Preferred,
    ReorderPolicy,
    static_policy,
    weigh_compare,
)
from fulcrum.structures import HEAD_ORDER, Argument, ResolutionMode

# reconcile (reconciliation) semantics: a predicate saying two indicants can be
# harmonized (both retained) rather than one defeating the other.
Reconcilable = Callable[[Argument, Argument], bool]
# supersede (abrogation) semantics: a temporal/authority relation. Returns LEFT if
# a abrogates b, RIGHT if b abrogates a, None if the pair is not so ordered.
Abrogates = Callable[[Argument, Argument], "Preferred | None"]

# Canonical mode orderings.
RECONCILE_FIRST: tuple[ResolutionMode, ...] = (
    ResolutionMode.RECONCILE,
    ResolutionMode.SUPERSEDE,
    ResolutionMode.WEIGH,
)
SUPERSEDE_FIRST: tuple[ResolutionMode, ...] = (
    ResolutionMode.SUPERSEDE,
    ResolutionMode.WEIGH,
    ResolutionMode.RECONCILE,
)
# Meta-controller disabled: preponderance only (used by the A4 ablation).
WEIGH_ONLY: tuple[ResolutionMode, ...] = (ResolutionMode.WEIGH,)


@dataclass
class Resolution:
    """Outcome of resolving a conflict between two arguments."""

    mode: ResolutionMode  # which operator decided
    preferred: Preferred  # LEFT/RIGHT if one prevailed; INCOMPARABLE if none
    note: str = ""  # human-readable audit line (auditability hook)
    dissolve: bool = False  # reconcile: the conflict is spurious -> keep BOTH (no defeat)


def _try_reconcile(a: Argument, b: Argument, reconcilable: Reconcilable | None) -> Resolution | None:
    """Reconciliation (reconcile / harmonization).

    Now IMPLEMENTED given an explicit reconcilability predicate (a modeling
    decision, logged in the design-decision notes). If ``reconcilable(a, b)`` holds, the
    apparent conflict is harmonizable (e.g. by scope restriction / scope_restriction):
    the two indicants are BOTH retained and the attack is DISSOLVED
    (``dissolve=True``) rather than resolved by defeat. This is the capability
    that escapes single-ordering preference (see theory/meta_separation.md).

    If no predicate is supplied the semantics are not fixed for this call, so
    we raise (in strict mode) -- callers must opt in to reconcile explicitly rather
    than silently getting placeholder behaviour.
    """

    if reconcilable is None:
        raise NotImplementedError(
            "reconcile requires an explicit reconcilability predicate; pass "
            "reconcilable=... to resolve(). See the design-decision notes."
        )
    if reconcilable(a, b):
        return Resolution(
            mode=ResolutionMode.RECONCILE,
            preferred=Preferred.INCOMPARABLE,
            note="reconcile: conflict harmonizable -> both indicants retained (attack dissolved)",
            dissolve=True,
        )
    return None  # not reconcilable -> next mode


def _try_supersede(a: Argument, b: Argument, abrogates: Abrogates | None) -> Resolution | None:
    """Abrogation (supersede).

    Now IMPLEMENTED given an explicit temporal/authority relation
    ``abrogates(a, b)`` returning LEFT (a abrogates b), RIGHT (b abrogates a),
    or None (unordered). The later/authoritative indicant defeats the earlier;
    this is a defeat NOT based on preponderance (head-scores), which is why
    ordering it before/after fulcrum in the mode sequence can change outcomes.
    """

    if abrogates is None:
        raise NotImplementedError(
            "supersede requires an explicit abrogation relation; pass abrogates=... "
            "to resolve(). See the design-decision notes."
        )
    verdict = abrogates(a, b)
    if verdict in (Preferred.LEFT, Preferred.RIGHT):
        return Resolution(
            mode=ResolutionMode.SUPERSEDE,
            preferred=verdict,
            note="supersede: later/authoritative indicant abrogates the earlier",
        )
    return None  # not temporally ordered -> next mode


def _try_weigh(
    a: Argument,
    b: Argument,
    base_ordering: HeadOrdering,
    reorder: ReorderPolicy,
) -> Resolution:
    """Preponderance. This one IS implemented (see preference.py)."""

    pref = weigh_compare(a, b, base_ordering=base_ordering, reorder=reorder)
    note = "fulcrum: first discriminating head decided" if pref is not Preferred.INCOMPARABLE \
        else "fulcrum: equipollence (equipollence), no preponderance"
    return Resolution(mode=ResolutionMode.WEIGH, preferred=pref, note=note)


def resolve(
    a: Argument,
    b: Argument,
    mode_order: Sequence[ResolutionMode] = RECONCILE_FIRST,
    base_ordering: HeadOrdering = HEAD_ORDER,
    reorder: ReorderPolicy = static_policy,
    strict: bool = False,
    reconcilable: Reconcilable | None = None,
    abrogates: Abrogates | None = None,
) -> Resolution:
    """Resolve a conflict between two arguments.

    Walks ``mode_order`` and applies the first mode that DECIDES. The mode
    ordering is the *meta-preference* (reconcile_first reconcile->supersede->fulcrum vs supersede_first
    supersede->fulcrum->reconcile); which mode fires first can change the outcome, and
    that configurability is the second candidate novelty (see
    theory/meta_separation.md).

    reconcile/supersede require their semantics to be supplied explicitly
    (``reconcilable`` / ``abrogates``). If a mode is reached without its
    semantics, it raises NotImplementedError; with ``strict=False`` (default)
    that is swallowed and treated as 'not applicable' (falls through), so the
    skeleton runs with fulcrum alone. ``strict=True`` propagates it -- used by
    the guard test asserting a mode is unavailable without supplied semantics.

    A reconcile decision has ``dissolve=True`` (both indicants retained, attack
    removed); the defeat layer honours this. Returns the winning Resolution;
    if every mode declines, INCOMPARABLE under the last mode tried.
    """

    last: Resolution | None = None
    for mode in mode_order:
        try:
            if mode is ResolutionMode.RECONCILE:
                r = _try_reconcile(a, b, reconcilable)
            elif mode is ResolutionMode.SUPERSEDE:
                r = _try_supersede(a, b, abrogates)
            else:
                r = _try_weigh(a, b, base_ordering, reorder)
        except NotImplementedError:
            if strict:
                raise
            continue

        if r is None:
            continue
        last = r
        # A mode DECIDES if it prefers a side OR dissolves the conflict (reconcile).
        if r.preferred is not Preferred.INCOMPARABLE or r.dissolve:
            return r

    if last is not None:
        return last
    return Resolution(
        mode=mode_order[-1],
        preferred=Preferred.INCOMPARABLE,
        note="no resolution mode applied",
    )
