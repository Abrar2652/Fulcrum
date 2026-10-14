"""Guard tests for the Phase-6 synthetic experiment (experiments/run.py).

These pin the pre-registered qualitative findings AND act as an anti-p-hacking
guard: the NEGATIVE CONTROL (transitive regime -> all models tie) must hold, so a
change that rigs the generator to make the calculus win spuriously fails here.
"""
from __future__ import annotations

import statistics

from experiments.run import run_regime


def test_context_regime_calculus_beats_single_ordering():
    rows, audit = run_regime("context")
    full = statistics.mean(rows["full (context calculus)"])
    a2 = statistics.mean(rows["A2 single ordering"])
    a1 = statistics.mean(rows["A1 single head"])
    assert full > 0.85                     # reaches the 1-epsilon ceiling
    assert full - a2 > 0.10                # clear, pre-registered gain
    assert full - a1 > 0.10                # multi-head + context both matter
    assert statistics.mean(audit) == 1.0   # H3: auditability, no leakage


def test_transitive_regime_is_a_tie_negative_control():
    rows, _ = run_regime("transitive")
    full = statistics.mean(rows["full (context calculus)"])
    a2 = statistics.mean(rows["A2 single ordering"])
    a1 = statistics.mean(rows["A1 single head"])
    # single-ordering world: the calculus must NOT spuriously beat the baseline
    assert abs(full - a2) < 0.02
    assert abs(full - a1) < 0.02
