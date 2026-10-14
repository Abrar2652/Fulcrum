"""Single-ordering preference baselines.

Provides:
  - ``SingleOrderingPreference``: the abstract shape of a single binary
    preference ordering over arguments (what Dung-extensions/ASPIC+/ABA+/VAF
    reduce to for our purposes).
  - ``LexicographicCollapse``: the most dangerous baseline for us -- it folds
    the four heads into ONE fixed lexicographic ordering. If this baseline
    matches the full fulcrum calculus on all instances, the separation
    conjecture is FALSE (outcome (B) in the project design notes Phase 2). This class exists
    precisely so we can *try to refute ourselves*.

Faithful adapters to ASPIC+ / ABA+ / VAF should wrap established
implementations rather than be reimplemented here; stubs mark where.
"""

from __future__ import annotations

from collections.abc import Sequence

from fulcrum.preference import Preferred
from fulcrum.structures import HEAD_ORDER, Argument, PreferenceHead


class SingleOrderingPreference:
    """Abstract single binary preference ordering.

    Subclasses implement ``prefer(a, b) -> Preferred`` using a SINGLE global
    ordering with no context-dependence. This is the formal target the
    separation result must beat.
    """

    def prefer(self, a: Argument, b: Argument) -> Preferred:  # pragma: no cover
        raise NotImplementedError


class LexicographicCollapse(SingleOrderingPreference):
    """Fixed lexicographic ordering over heads -- NO context-dependence.

    This is the honest, strong baseline. The whole separation argument turns
    on whether context-dependent reordering (preference.py) can produce a
    comparison behavior that NO instance of this class reproduces.

    Use ``tests/gate/`` to search for a separating example; if none exists,
    report outcome (B).
    """

    def __init__(self, ordering: Sequence[PreferenceHead] = HEAD_ORDER) -> None:
        self.ordering = tuple(ordering)

    def prefer(self, a: Argument, b: Argument) -> Preferred:
        for head in self.ordering:
            sa, sb = a.score(head), b.score(head)
            if sa > sb:
                return Preferred.LEFT
            if sb > sa:
                return Preferred.RIGHT
        return Preferred.INCOMPARABLE


class AspicPlusAdapter(SingleOrderingPreference):
    """STUB. Wrap an established ASPIC+ implementation. Do not reimplement
    ASPIC+ from scratch; cite and adapt. See the project design notes Phase 1 reading."""

    def prefer(self, a: Argument, b: Argument) -> Preferred:
        raise NotImplementedError(
            "Adapt an established ASPIC+ library here (Phase 4/6). A hand-rolled "
            "approximation risks being an unfaithful strawman baseline."
        )


class AbaPlusAdapter(SingleOrderingPreference):
    """STUB. Wrap an established ABA+ implementation (cf. Cyras & Toni 2016)."""

    def prefer(self, a: Argument, b: Argument) -> Preferred:
        raise NotImplementedError("Adapt an established ABA+ library here (Phase 4/6).")
