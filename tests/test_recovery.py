"""Recovery property: with preference structure switched off, the induced
defeat relation must coincide with the plain Dung attack relation, and
grounded acceptance must match classical Dung on canonical examples.

This is the conservative-extension sanity check (the project design notes Phase 3). If these
fail, the framework does not degrade gracefully and reviewers will not trust
it.
"""

from __future__ import annotations

from fulcrum.defeat import ArgumentationFramework
from fulcrum.semantics import grounded_extension
from fulcrum.structures import Argument, PreferenceHead


def _neutral(aid: str) -> Argument:
    # all-zero head scores, probable strength -> fulcrum always INCOMPARABLE,
    # so no attack is repelled: defeat == attack
    return Argument(aid, claim=aid, head_scores={h: 0.0 for h in PreferenceHead})


def test_defeat_recovers_attack_when_preference_off():
    args = [_neutral("a"), _neutral("b"), _neutral("c")]
    attacks = [("a", "b"), ("b", "c")]
    af = ArgumentationFramework(args, attacks)
    assert set(af.defeats()) == set(attacks)


def test_grounded_matches_dung_simple_chain():
    # a -> b -> c : grounded = {a, c}
    args = [_neutral("a"), _neutral("b"), _neutral("c")]
    attacks = [("a", "b"), ("b", "c")]
    af = ArgumentationFramework(args, attacks)
    g = af.to_digraph("defeat")
    assert grounded_extension(g) == {"a", "c"}


def test_grounded_two_cycle_is_empty():
    # a <-> b mutual attack, no preference -> both defeats survive -> grounded empty
    args = [_neutral("a"), _neutral("b")]
    attacks = [("a", "b"), ("b", "a")]
    af = ArgumentationFramework(args, attacks)
    g = af.to_digraph("defeat")
    assert grounded_extension(g) == set()


def test_preference_can_break_symmetry():
    # a <-> b but a strictly preferred on SOURCE -> only a defeats b -> grounded {a}
    a = Argument("a", "p", head_scores={PreferenceHead.SOURCE: 2.0})
    b = Argument("b", "q", head_scores={PreferenceHead.SOURCE: 1.0})
    af = ArgumentationFramework([a, b], [("a", "b"), ("b", "a")])
    g = af.to_digraph("defeat")
    assert grounded_extension(g) == {"a"}
