"""Core data structures for the fulcrum calculus.

This module defines the *vocabulary* of the framework. It is intentionally
small and dependency-light. The formal meaning of these structures is
specified in ``theory/fulcrum_calculus.md``; this file is the executable
counterpart of those definitions.

Mapping to the source_domain source (kept faithful, per the project design notes):
  - EpistemicStrength  <-> certain (certain) vs probable (probable). fulcrum
    (preponderance) operates ONLY among indicants of comparable, i.e.
    probable, strength. CERTAIN indicants do not enter fulcrum.
  - PreferenceHead     <-> the four classical heads of resolution criteria:
      SOURCE   : by the chain/source (e.g. reliability/number of transmitters)
      SPECIFICITY    : by the text (e.g. specific preferred to general)
      MODAL  : by mode of signification (e.g. prohibition over permission)
      CORROBORATION  : by an external corroborating factor (another text, consensus, analogy)
  - Argument.head_scores : the vector <source_reliability, specificity, modal_force, corroboration> in R^4,
    the per-head strength used by the (context-conditioned) preference
    operator. Semantics of the ordering direction are fixed in preference.py.

NOTE (faithfulness): the choice to represent the heads as a fixed 4-vector is
a modeling abstraction of a contested tradition. Departures from the sources
must be logged in the design-decision notes. See the project design notes Trap #4.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum


class EpistemicStrength(IntEnum):
    """certain / probable typing.

    Ordered so that CERTAIN > PROBABLE. The Phase-2 relaxation to a calibrated
    scalar (see differentiable/) will generalize this two-valued type.
    """

    PROBABLE = 0  # probable
    CERTAIN = 1  # certain


class PreferenceHead(Enum):
    """The four principal heads of fulcrum. Order here is the *declaration*
    order only; the *applied* priority ordering is a (context-dependent)
    parameter of the calculus, NOT this enum order. Do not rely on enum
    order for precedence."""

    SOURCE = "source"  # source / chain
    SPECIFICITY = "specificity"  # text
    MODAL = "modal"  # mode of signification
    CORROBORATION = "corroboration"  # external corroboration


# Canonical ordering of heads when a vector<->head correspondence is needed.
HEAD_ORDER: tuple[PreferenceHead, ...] = (
    PreferenceHead.SOURCE,
    PreferenceHead.SPECIFICITY,
    PreferenceHead.MODAL,
    PreferenceHead.CORROBORATION,
)


@dataclass(frozen=True)
class Argument:
    """An argument in the framework.

    Attributes
    ----------
    aid:
        Unique identifier.
    claim:
        The proposition asserted (opaque string; content-agnostic core).
    strength:
        Epistemic-strength type (certain/probable analog).
    head_scores:
        Mapping from PreferenceHead to a real score. Missing heads default
        to 0.0. Direction/meaning of scores is fixed by preference.py.
    context:
        Free-form tags used by the context-conditioned reordering (Phase 2
        candidate novelty). Kept as a frozenset of strings for hashability.
    """

    aid: str
    claim: str
    strength: EpistemicStrength = EpistemicStrength.PROBABLE
    head_scores: Mapping[PreferenceHead, float] = field(default_factory=dict)
    context: frozenset[str] = field(default_factory=frozenset)

    def score(self, head: PreferenceHead) -> float:
        return float(self.head_scores.get(head, 0.0))

    def vector(self) -> tuple[float, ...]:
        """Return head scores in HEAD_ORDER as a tuple."""
        return tuple(self.score(h) for h in HEAD_ORDER)


class ResolutionMode(Enum):
    """The meta-level resolution operators for conflict of evidence (conflict of
    evidence). The *ordering* over these modes is configurable (reconcile_first default
    RECONCILE -> SUPERSEDE -> WEIGH vs supersede_first SUPERSEDE -> WEIGH -> RECONCILE); that
    configurability is itself a candidate novelty (meta-preference). See
    meta_controller.py.
    """

    RECONCILE = "reconcile"  # reconciliation
    SUPERSEDE = "supersede"  # abrogation / override
    WEIGH = "weigh"  # preponderance / preference

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ResolutionMode.{self.name}"
