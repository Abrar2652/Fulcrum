"""EAF reification of reconcile/supersede (rehabilitates theory/meta_separation.md).

Two external reviews demoted the "reconcile deletes an attack" framing as unfair
against first-order preference. Their suggested fix : reify reconcile/supersede
as SECOND-ORDER attacks in an Extended Argumentation Framework (Modgil, "Reasoning
about preferences in argumentation frameworks", AIJ 173, 2009), where an argument
may attack an *attack*. Then:

  - reconcile (reconciliation) = a meta-argument attacking BOTH base attacks (a,b),(b,a)
    -> if accepted, the conflict edge is neutralized (both survive).
  - supersede (abrogation, b over a) = a meta-argument attacking (a,b) -> only (b,a)
    survives, so b defeats a.
  - the mode ordering mu (reconcile_first vs supersede_first) = an OBJECT-LEVEL preference over which
    meta-argument prevails -- the genuine "meta-preference" (preference expressed
    inside the framework over the second-order attackers).

This is a *faithful encoding*, NOT a claim of expressiveness beyond EAFs (EAFs can
of course express it). The legitimate, honest consequence: the fulcrum conflict-of-evidence
machinery requires SECOND-ORDER (attack-on-attack) power, so it is not reproducible
by any FIRST-ORDER single-ordering framework (Dung/ASPIC+/ABA+/VAF, which evaluate
arguments only) on the same base AF -- which is exactly the reference set the paper
separates from. See theory/meta_separation.md.

We implement a minimal EAF sufficient for the taarud witness: meta-arguments
target base attacks, and the applicable meta-argument of the highest-priority mode
in mu is accepted (mirroring the meta-controller's 'first mode that decides'). This
coincides with Modgil's EAF semantics on this acyclic meta-structure; the full
semantics (reinstatement sets) are cited, not reimplemented.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import networkx as nx

from fulcrum.meta_controller import RECONCILE_FIRST
from fulcrum.structures import ResolutionMode

Edge = tuple[str, str]


@dataclass(frozen=True)
class MetaArgument:
    """A second-order argument: if accepted, it neutralizes the base attacks in
    ``targets``. ``mode`` ties it to a resolution operator for the mu-preference."""

    aid: str
    mode: ResolutionMode          # RECONCILE or SUPERSEDE
    targets: frozenset[Edge]      # base attacks this meta-argument attacks


@dataclass
class ExtendedAF:
    base_args: Sequence[str]
    base_attacks: Sequence[Edge]
    meta: Sequence[MetaArgument] = field(default_factory=list)
    mode_order: Sequence[ResolutionMode] = RECONCILE_FIRST  # mu (meta-preference)

    def accepted_meta(self) -> list[MetaArgument]:
        """The applicable meta-arguments of the highest-priority mode in mu are
        accepted; lower-priority ones for the *same* conflict are suppressed.
        Mirrors meta_controller's 'first mode that decides' -> mu is the
        object-level preference over meta-arguments (Modgil EAF: a preference
        realised as attacks among the second-order attackers)."""
        by_mode: dict[ResolutionMode, list[MetaArgument]] = {}
        for m in self.meta:
            by_mode.setdefault(m.mode, []).append(m)
        for mode in self.mode_order:
            if by_mode.get(mode):
                return by_mode[mode]  # first applicable mode wins
        return []

    def active_attacks(self) -> set[Edge]:
        """Base attacks minus those neutralized by accepted meta-arguments."""
        removed: set[Edge] = set()
        for m in self.accepted_meta():
            removed |= set(m.targets)
        return set(self.base_attacks) - removed

    def to_digraph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.base_args)
        g.add_edges_from(self.active_attacks())
        return g


def conflict_witness_eaf(mode_order: Sequence[ResolutionMode] = RECONCILE_FIRST) -> ExtendedAF:
    """The reified conflict: base args a,b mutually attacking; a reconcile meta-arg
    neutralizing both edges; a supersede meta-arg neutralizing (a,b) (b abrogates a).
    mu decides which fires (reconcile_first: reconcile; supersede_first: supersede)."""
    base_attacks: list[Edge] = [("a", "b"), ("b", "a")]
    reconcile = MetaArgument("reconcile", ResolutionMode.RECONCILE, frozenset({("a", "b"), ("b", "a")}))
    supersede = MetaArgument("supersede", ResolutionMode.SUPERSEDE, frozenset({("a", "b")}))
    return ExtendedAF(["a", "b"], base_attacks, [reconcile, supersede], mode_order)
