"""Tests for the fulcrum preference operator."""

from __future__ import annotations

import pytest

from fulcrum.preference import Preferred, weigh_compare
from fulcrum.structures import HEAD_ORDER, Argument, PreferenceHead


def test_first_discriminating_head_decides(probable_pair):
    a, b = probable_pair
    # SOURCE is first in HEAD_ORDER; a has higher SOURCE -> a preferred
    assert weigh_compare(a, b) is Preferred.LEFT


def test_reordering_changes_outcome():
    a = Argument("a", "p", head_scores={PreferenceHead.SOURCE: 2.0, PreferenceHead.SPECIFICITY: 1.0})
    b = Argument("b", "q", head_scores={PreferenceHead.SOURCE: 1.0, PreferenceHead.SPECIFICITY: 3.0})
    # default order (SOURCE first): a wins
    assert weigh_compare(a, b) is Preferred.LEFT
    # put SPECIFICITY first: b wins -- demonstrates ordering matters (necessary, not
    # sufficient, for the separation claim)
    matn_first = (PreferenceHead.SPECIFICITY, PreferenceHead.SOURCE,
                  PreferenceHead.MODAL, PreferenceHead.CORROBORATION)
    assert weigh_compare(a, b, base_ordering=matn_first) is Preferred.RIGHT


def test_equipollence_is_incomparable():
    a = Argument("a", "p", head_scores={PreferenceHead.SOURCE: 1.0})
    b = Argument("b", "q", head_scores={PreferenceHead.SOURCE: 1.0})
    assert weigh_compare(a, b) is Preferred.INCOMPARABLE


def test_strength_gate_rejects_certain(probable_pair, certain_arg):
    a, _ = probable_pair
    with pytest.raises(ValueError):
        weigh_compare(a, certain_arg)
    with pytest.raises(ValueError):
        weigh_compare(certain_arg, a)


def test_vector_matches_head_order():
    a = Argument("a", "p", head_scores={PreferenceHead.SPECIFICITY: 5.0})
    vec = a.vector()
    assert len(vec) == len(HEAD_ORDER)
    # SPECIFICITY is index 1 in HEAD_ORDER
    assert vec[1] == 5.0
    assert vec[0] == 0.0  # SOURCE default
