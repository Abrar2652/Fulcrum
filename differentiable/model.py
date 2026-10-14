"""Differentiable fulcrum preponderance (Phase 5).

The formal core (preference.py) resolves a conflict by a hard, context-conditioned
lexicographic scan over heads. Here we give it a DIFFERENTIABLE, LEARNABLE lift in
the spirit of Potyka's "Interpreting Neural Networks as Gradual Argumentation
Frameworks" (arXiv:2012.05738): the per-head weighting is learned and
context-conditioned, the verdict is a continuous acceptability degree, and the
certain/probable type is relaxed from binary to a calibrated scalar strength.

Crucially it stays AUDITABLE (the selling point unique to this work): the verdict
is a weighted sum of per-head score differences, so for every conflict we can read
off the *responsible head* (the largest-magnitude contribution) — no black box.

Design (deliberately transparent, not an opaque MLP):
  d = s(a) - s(b)  in R^H  (head-score differences)
  logit(a>b | ctx) = w(ctx) . d              # w(ctx): learned per-context weights
  verdict prob     = sigmoid(logit)          # continuous acceptability of "a preferred"
  responsible head = argmax_h | w(ctx)_h * d_h |
  strength gate    : a soft override so a (near-)certain indicant dominates a
                     probable one (relaxes the certain/probable gate to a scalar).

Ablation switch `context_conditioned=False` shares one weight vector across all
contexts = the interpretable fixed-weight baseline the learned version must beat.
"""
from __future__ import annotations

import torch
from torch import nn


class DifferentiablePreponderance(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int = 1, context_conditioned: bool = True,
                 strength_gate: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx if context_conditioned else 1
        self.context_conditioned = context_conditioned
        self.strength_gate = strength_gate
        # per-context head weights (context_conditioned) or one shared row (ablation)
        self.W = nn.Parameter(torch.zeros(self.n_ctx, n_heads))
        # strength-gate strength: how decisively a more-certain (certain) indicant
        # dominates a more-probable (probable) one. Learned, kept >=0.
        self.lam = nn.Parameter(torch.tensor(3.0))
        # temperature for post-hoc calibration (fit separately, frozen during main
        # training via requires_grad_(False) in the calibrator).
        self.register_buffer("temperature", torch.tensor(1.0))

    def _w(self, ctx: torch.Tensor) -> torch.Tensor:
        idx = ctx if self.context_conditioned else torch.zeros_like(ctx)
        return self.W[idx]  # (batch, n_heads)

    def logit(self, d: torch.Tensor, ctx: torch.Tensor,
              strength_a: torch.Tensor | None = None,
              strength_b: torch.Tensor | None = None) -> torch.Tensor:
        """Preponderance logit for 'a preferred to b'. d: (batch, n_heads).

        Strength gate (relaxes certain/probable to a scalar in [0,1], 1=certain):
        add lambda*(strength_a - strength_b). Among equal-strength (probable) pairs
        this term vanishes and preponderance decides; as one side -> certain it
        dominates -- the differentiable analog of "fulcrum only among comparable
        (probable) strength; certain is never outweighed."""
        z = (self._w(ctx) * d).sum(dim=-1)
        if self.strength_gate and strength_a is not None and strength_b is not None:
            z = z + self.lam.abs() * (strength_a - strength_b)
        return z

    def forward(self, d, ctx, strength_a=None, strength_b=None):
        """Continuous acceptability P(a preferred to b) in (0,1)."""
        return torch.sigmoid(self.logit(d, ctx, strength_a, strength_b) / self.temperature)

    @torch.no_grad()
    def responsible_head(self, d: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Auditability: index of the head with the largest-magnitude contribution."""
        contrib = (self._w(ctx) * d).abs()
        return contrib.argmax(dim=-1)

    @torch.no_grad()
    def contributions(self, d: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Per-head signed contribution w_h * d_h (the audit trail)."""
        return self._w(ctx) * d
