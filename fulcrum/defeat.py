"""From attack + preference to defeat.

In Dung's abstract framework a conflict is just an attack a -> b. In
preference-based frameworks, whether an attack *succeeds* (becomes a defeat)
depends on preference: an attack from a weaker argument on a stronger one may
be repelled.

Here defeat is computed by resolving the conflict with the fulcrum
meta-controller. The KEY design commitment (for reviewer trust) is the
RECOVERY property: when preference structure is switched off (single trivial
head, static policy, fulcrum-only), the induced defeat relation must coincide
with the plain Dung attack relation. This is asserted by tests in
tests/test_recovery.py and must be proven in theory/semantics_and_properties.md
(conservative-extension result).
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from fulcrum.meta_controller import RECONCILE_FIRST, Abrogates, Reconcilable, resolve
from fulcrum.preference import Preferred, ReorderPolicy, static_policy
from fulcrum.structures import HEAD_ORDER, Argument, ResolutionMode


class ArgumentationFramework:
    """A fulcrum-augmented argumentation framework.

    Parameters
    ----------
    arguments:
        The arguments (nodes).
    attacks:
        Directed attack pairs (aid, aid): first attacks second.
    mode_order, base_ordering, reorder:
        Preference configuration passed through to the meta-controller.
    """

    def __init__(
        self,
        arguments: Sequence[Argument],
        attacks: Sequence[tuple[str, str]],
        mode_order: Sequence[ResolutionMode] = RECONCILE_FIRST,
        base_ordering: Sequence = HEAD_ORDER,
        reorder: ReorderPolicy = static_policy,
        reconcilable: Reconcilable | None = None,
        abrogates: Abrogates | None = None,
    ) -> None:
        self._args: dict[str, Argument] = {a.aid: a for a in arguments}
        if len(self._args) != len(arguments):
            raise ValueError("duplicate argument ids")
        for src, dst in attacks:
            if src not in self._args or dst not in self._args:
                raise ValueError(f"attack references unknown argument: {(src, dst)}")
        self.attacks: list[tuple[str, str]] = list(attacks)
        self.mode_order = tuple(mode_order)
        self.base_ordering = tuple(base_ordering)
        self.reorder = reorder
        # reconcile/supersede semantics (optional). When supplied, the meta-controller
        # can dissolve a conflict (reconcile) or abrogate by authority (supersede); which
        # fires first depends on mode_order (reconcile_first vs supersede_first) -> see
        # theory/meta_separation.md.
        self.reconcilable = reconcilable
        self.abrogates = abrogates

    @property
    def arguments(self) -> dict[str, Argument]:
        return dict(self._args)

    def defeats(self) -> list[tuple[str, str]]:
        """Compute the defeat relation from attacks + preference.

        For each attack (src -> dst) that is *mutual* (both directions
        present), resolution decides which direction survives as a defeat.
        For a one-way attack, we resolve src vs dst and keep the attack as a
        defeat unless dst is strictly preferred (the attack is repelled).

        NOTE: this is ONE reasonable defeat definition; alternatives exist
        (e.g. ASPIC+ distinguishes rebut/undermine/undercut). The choice is a
        modeling decision logged in the design-decision notes. The recovery property is
        what pins it down against Dung in the degenerate case.
        """

        attack_set = set(self.attacks)
        defeats: list[tuple[str, str]] = []
        seen_pairs: set[frozenset[str]] = set()

        for src, dst in self.attacks:
            a, b = self._args[src], self._args[dst]

            # Strength gate: if either indicant is CERTAIN, fulcrum is not the
            # right operator. For the skeleton we keep the attack as a defeat
            # (certain indicants are not repelled by weighing). This is a
            # placeholder pending supersede/reconcile semantics.
            try:
                res = resolve(
                    a,
                    b,
                    mode_order=self.mode_order,
                    base_ordering=self.base_ordering,
                    reorder=self.reorder,
                    reconcilable=self.reconcilable,
                    abrogates=self.abrogates,
                )
            except ValueError:
                # certain-strength pair: default to keeping the attack
                defeats.append((src, dst))
                continue

            # reconcile (reconciliation): the conflict is spurious -> dissolve the
            # attack entirely (both indicants retained, no defeat either way).
            if res.dissolve:
                seen_pairs.add(frozenset({src, dst}))
                continue

            mutual = (dst, src) in attack_set
            if mutual:
                pair = frozenset({src, dst})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if res.preferred is Preferred.LEFT:
                    defeats.append((src, dst))
                elif res.preferred is Preferred.RIGHT:
                    defeats.append((dst, src))
                else:
                    # equipollence: both directions survive as defeats
                    defeats.append((src, dst))
                    defeats.append((dst, src))
            else:
                # one-way attack: repelled only if dst strictly preferred
                if res.preferred is Preferred.RIGHT:
                    continue
                defeats.append((src, dst))

        return defeats

    def to_digraph(self, relation: str = "defeat") -> nx.DiGraph:
        """Return a networkx DiGraph of the chosen relation for the semantics
        layer. ``relation`` in {"attack", "defeat"}."""

        g = nx.DiGraph()
        g.add_nodes_from(self._args.keys())
        edges = self.attacks if relation == "attack" else self.defeats()
        g.add_edges_from(edges)
        return g
