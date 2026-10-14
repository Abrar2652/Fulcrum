"""Golden conflict-resolution examples (Phase 4), structural illustrations of the
four fulcrum heads + the strength gate + equipollence + the intransitive witness.

These assert the calculus resolves each classical conflict *by the head it
should*. They are STRUCTURAL (which head decides), pending domain validation;
see fulcrum/examples.py and theory/golden_examples.md.
"""
from __future__ import annotations

import pytest

from fulcrum import examples as ex
from fulcrum.semantics import grounded_extension


def _contested(theory) -> set[str]:
    return {str(lit) for pair in theory.contraries for lit in pair}


@pytest.mark.parametrize("builder", ex.ALL_EXAMPLES, ids=lambda b: b.__name__)
def test_golden_example_resolves_by_expected_head(builder):
    e = builder()
    cm = {a.aid: str(a.conc) for a in e.saf.args}
    grounded = {cm[x] for x in grounded_extension(e.saf.to_digraph())}
    grounded_rulings = grounded & _contested(e.theory)
    if e.expect_winner is None:
        # equipollence / collective standoff: no contested ruling is accepted
        assert grounded_rulings == set(), (e.name, grounded_rulings)
    else:
        assert grounded_rulings == {e.expect_winner}, (e.name, grounded_rulings)


def test_certain_beats_even_maximal_probable():
    """The strength gate must be absolute: a certain ruling wins over a probable one
    with the maximal head-score on every head."""
    e = ex.strength_gate_certain()
    cm = {a.aid: str(a.conc) for a in e.saf.args}
    defeats = {(cm[s], cm[t]) for s, t in e.saf.defeats()}
    assert ("q", "p") in defeats and ("p", "q") not in defeats


def test_context_reordering_is_the_directed_cycle():
    """The novel case reproduces the separation witness structurally."""
    e = ex.context_dependent_reordering()
    cm = {a.aid: str(a.conc) for a in e.saf.args}
    defeats = {(cm[s], cm[t]) for s, t in e.saf.defeats()}
    assert defeats == {("r0", "r1"), ("r1", "r2"), ("r2", "r0")}


def test_each_head_is_the_deciding_one_in_isolation():
    """Sanity that the four single-head examples each differ on exactly the head
    they claim to illustrate (guards against an example silently deciding on a
    different head)."""
    from fulcrum.structures import HEAD_ORDER
    expected = {"by_source_reliability": HEAD_ORDER[0], "by_specificity": HEAD_ORDER[1],
                "by_modal_force": HEAD_ORDER[2], "by_corroboration": HEAD_ORDER[3]}
    for name, head in expected.items():
        e = getattr(ex, name)()
        # find the two defeasible rules and confirm they differ only on `head`
        rules = e.theory.defeasible
        assert len(rules) == 2
        va, vb = rules[0].head_scores, rules[1].head_scores
        diffs = [i for i in range(len(va)) if va[i] != vb[i]]
        assert diffs == [list(HEAD_ORDER).index(head)], (name, diffs)
