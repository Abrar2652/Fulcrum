"""Shared fixtures and small hand-built examples."""

from __future__ import annotations

import pytest

from fulcrum.structures import Argument, EpistemicStrength, PreferenceHead


@pytest.fixture
def probable_pair() -> tuple[Argument, Argument]:
    a = Argument(
        aid="a",
        claim="p",
        strength=EpistemicStrength.PROBABLE,
        head_scores={PreferenceHead.SOURCE: 2.0, PreferenceHead.SPECIFICITY: 1.0},
    )
    b = Argument(
        aid="b",
        claim="not-p",
        strength=EpistemicStrength.PROBABLE,
        head_scores={PreferenceHead.SOURCE: 1.0, PreferenceHead.SPECIFICITY: 3.0},
    )
    return a, b


@pytest.fixture
def certain_arg() -> Argument:
    return Argument(aid="c", claim="q", strength=EpistemicStrength.CERTAIN)
