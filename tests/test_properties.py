"""Property-based tests (hypothesis) for well-definedness of the operator.

Required by the project design notes Phase 4. These check structural properties that must
hold regardless of scores, e.g. determinism, antisymmetry of strict
preference, and reflexive incomparability.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from fulcrum.preference import Preferred, weigh_compare
from fulcrum.structures import Argument, PreferenceHead

score_st = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)


@st.composite
def probable_arg(draw, aid: str) -> Argument:
    scores = {h: draw(score_st) for h in PreferenceHead}
    return Argument(aid=aid, claim=aid, head_scores=scores)


@given(a=probable_arg("a"), b=probable_arg("b"))
def test_determinism(a: Argument, b: Argument):
    assert weigh_compare(a, b) == weigh_compare(a, b)


@given(a=probable_arg("a"), b=probable_arg("b"))
def test_antisymmetry(a: Argument, b: Argument):
    ab = weigh_compare(a, b)
    ba = weigh_compare(b, a)
    if ab is Preferred.LEFT:
        assert ba is Preferred.RIGHT
    elif ab is Preferred.RIGHT:
        assert ba is Preferred.LEFT
    else:
        assert ba is Preferred.INCOMPARABLE


@given(a=probable_arg("a"))
def test_reflexive_incomparable(a: Argument):
    # comparing an argument to itself: all heads tie -> INCOMPARABLE
    a2 = Argument(aid="a2", claim=a.claim, head_scores=dict(a.head_scores))
    assert weigh_compare(a, a2) is Preferred.INCOMPARABLE
