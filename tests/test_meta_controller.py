"""Meta-controller tests: fulcrum fallback, the now-IMPLEMENTED reconcile/supersede modes
(requiring explicitly supplied semantics), and the meta-preference behaviour
(reconcile_first vs supersede_first mode-ordering) underpinning theory/meta_separation.md.

reconcile/supersede were stubs; they are now implemented but require their semantics
(a reconcilability predicate / abrogation relation) to be supplied at the call
site. Absent those, strict mode still raises -- so placeholder semantics can
never be shipped silently (the original tripwire's intent, preserved).
"""
from __future__ import annotations

import pytest

from fulcrum.meta_controller import RECONCILE_FIRST, SUPERSEDE_FIRST, resolve
from fulcrum.preference import Preferred
from fulcrum.structures import Argument, PreferenceHead, ResolutionMode


def _pair():
    a = Argument("a", "p", head_scores={PreferenceHead.SOURCE: 2.0})
    b = Argument("b", "q", head_scores={PreferenceHead.SOURCE: 1.0})
    return a, b


def test_fulcrum_fallback_decides_non_strict():
    a, b = _pair()
    res = resolve(a, b, mode_order=RECONCILE_FIRST)  # no reconcile/supersede semantics -> fulcrum
    assert res.mode is ResolutionMode.WEIGH
    assert res.preferred is Preferred.LEFT


def test_supersede_first_order_also_reaches_fulcrum_non_strict():
    a, b = _pair()
    res = resolve(a, b, mode_order=SUPERSEDE_FIRST)
    assert res.mode is ResolutionMode.WEIGH


def test_modes_require_explicit_semantics_in_strict_mode():
    """Guard (was the 'stubs unimplemented' tripwire): without supplied
    reconcile/supersede semantics, strict mode raises -- placeholder semantics are never
    shipped silently."""
    a, b = _pair()
    with pytest.raises(NotImplementedError):
        resolve(a, b, mode_order=RECONCILE_FIRST, strict=True)
    with pytest.raises(NotImplementedError):
        resolve(a, b, mode_order=SUPERSEDE_FIRST, strict=True)


def test_reconcile_dissolves_when_reconcilable():
    a, b = _pair()
    res = resolve(a, b, mode_order=RECONCILE_FIRST, reconcilable=lambda x, y: True)
    assert res.mode is ResolutionMode.RECONCILE
    assert res.dissolve is True
    assert res.preferred is Preferred.INCOMPARABLE


def test_supersede_abrogates_by_authority_relation():
    a, b = _pair()
    # b abrogates a
    abro = lambda x, y: Preferred.RIGHT if x.aid == "a" else Preferred.LEFT
    res = resolve(a, b, mode_order=SUPERSEDE_FIRST, abrogates=abro)
    assert res.mode is ResolutionMode.SUPERSEDE
    assert res.preferred is Preferred.RIGHT


def test_mode_ordering_changes_outcome():
    """Meta-preference: on a pair that is BOTH reconcilable AND abrogation-
    ordered, reconcile_first (reconcile first) dissolves while supersede_first (supersede first) abrogates."""
    a, b = _pair()
    recon = lambda x, y: True
    abro = lambda x, y: Preferred.RIGHT if x.aid == "a" else Preferred.LEFT
    reconcile_first = resolve(a, b, RECONCILE_FIRST, reconcilable=recon, abrogates=abro)
    supersede_first = resolve(a, b, SUPERSEDE_FIRST, reconcilable=recon, abrogates=abro)
    assert reconcile_first.mode is ResolutionMode.RECONCILE and reconcile_first.dissolve
    assert supersede_first.mode is ResolutionMode.SUPERSEDE and not supersede_first.dissolve
