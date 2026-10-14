"""Phase-5 differentiable preponderance tests: the learned context model earns its
complexity (beats the fixed-weight ablation on context-dependent data, ties on
single-ordering data), stays auditable, produces calibrated probabilities, and the
strength gate makes a certain (certain) indicant dominate a probable (probable) one.
"""
from __future__ import annotations

import torch

from differentiable.audit import audit_batch, intervention_faithfulness
from differentiable.model import DifferentiablePreponderance
from differentiable.train import run
from experiments.synthetic import N_HEADS


def test_learned_beats_fixed_on_context_ties_on_transitive():
    ctx = run("context", seed=0)
    assert ctx["learned"]["acc"] - ctx["fixed"]["acc"] > 0.10      # earns complexity
    assert ctx["learned"]["brier"] < ctx["fixed"]["brier"]         # better proper score
    trans = run("transitive", seed=0)
    assert abs(trans["learned"]["acc"] - trans["fixed"]["acc"]) < 0.03  # negative control


def test_probabilities_are_calibrated_enough():
    ctx = run("context", seed=0)
    assert ctx["learned"]["ece"] < 0.15         # temperature-scaled, reasonable
    assert 0.0 <= ctx["learned"]["brier"] <= 0.25


def test_auditability_responsible_head_is_causal():
    ctx = run("context", seed=0)
    # from run()'s eval: responsible-head intervention flips far more than others
    assert ctx["learned"]["audit_responsible"] > 0.8
    assert ctx["learned"]["audit_responsible"] > ctx["learned"]["audit_other"] + 0.5


def test_strength_gate_certain_dominates_probable():
    """With the strength gate, a certain (strength=1) indicant beats a probable
    (strength=0) one even when preponderance favours the probable side."""
    m = DifferentiablePreponderance(N_HEADS, n_ctx=1, context_conditioned=False)
    with torch.no_grad():
        m.W.fill_(0.0)
        m.W[0, 0] = 1.0               # preponderance favours b on head 0...
    d = torch.tensor([[-2.0, 0.0, 0.0, 0.0]])   # ...d<0 there -> w.d = -2 (b preferred)
    ctx = torch.tensor([0])
    sa = torch.tensor([1.0])          # a certain
    sb = torch.tensor([0.0])          # b probable
    p_nogate = float(m(d, ctx))                    # ~ sigmoid(-2) < 0.5 -> b
    p_gate = float(m(d, ctx, strength_a=sa, strength_b=sb))
    assert p_nogate < 0.5             # without strength: b wins on preponderance
    assert p_gate > 0.5              # with strength: certain a dominates


def test_h2_strength_typing_reliability_aware_calibration():
    """H2: strength typing yields reliability-aware calibration — confident on
    certain (reliable) conflicts, uncertain on probable (noisy) ones — and much
    better Brier on certain conflicts + overall. Aggregate ECE may tie."""
    from differentiable.h2 import run as h2run
    r = h2run(seed=0)
    typed, non = r["strength-typed"], r["non-typed"]
    assert typed["acc"] - non["acc"] > 0.1                    # uses the certain gate
    assert typed["brier"] < non["brier"]                     # better proper score overall
    assert typed["brier_certain"] < non["brier_certain"] - 0.1   # much better on reliable
    assert typed["conf_certain"] - typed["conf_probable"] > 0.1  # confidence tracks reliability
    assert non["conf_certain"] - non["conf_probable"] < 0.05      # non-typed: flat confidence


def test_audit_report_shape():
    ctx_run = run("context", seed=1)  # smoke that training + audit compose
    m = DifferentiablePreponderance(N_HEADS, n_ctx=N_HEADS, context_conditioned=True)
    d = torch.randn(6, N_HEADS)
    c = torch.zeros(6, dtype=torch.long)
    audits = audit_batch(m, d, c)
    assert len(audits) == 6
    assert all(a.responsible_head in [h for h in a.contributions] for a in audits)
    faith = intervention_faithfulness(m, d, c)
    assert set(faith) == {"flip_when_responsible_negated", "flip_when_other_negated", "faithful"}
    _ = ctx_run
