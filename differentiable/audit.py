"""Auditability report generator (Phase 5 deliverable).

Auditability is the selling point unique to this work (the project design notes Phase 5): the
learned model is a transparent weighted sum of per-head score differences, so for
every resolved conflict we can emit *which head decided* and *by how much* — no
black box. This module turns a trained DifferentiablePreponderance + a batch of
conflicts into a per-conflict audit trail and a summary intervention metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fulcrum.structures import HEAD_ORDER


@dataclass
class ConflictAudit:
    verdict: str            # 'a' or 'b' (predicted winner)
    confidence: float       # max(p, 1-p)
    responsible_head: str   # head name that decided
    contributions: dict     # head -> signed contribution w_h * d_h


def audit_batch(model, d: torch.Tensor, ctx: torch.Tensor,
                head_names=None) -> list[ConflictAudit]:
    head_names = head_names or [h.value for h in HEAD_ORDER[:model.n_heads]]
    with torch.no_grad():
        p = model(d, ctx)
        contrib = model.contributions(d, ctx)
        rh = model.responsible_head(d, ctx)
    out = []
    for i in range(len(d)):
        pi = float(p[i])
        out.append(ConflictAudit(
            verdict="a" if pi >= 0.5 else "b",
            confidence=max(pi, 1 - pi),
            responsible_head=head_names[int(rh[i])],
            contributions={head_names[h]: round(float(contrib[i, h]), 3)
                           for h in range(model.n_heads)},
        ))
    return out


@torch.no_grad()
def intervention_faithfulness(model, d: torch.Tensor, ctx: torch.Tensor) -> dict:
    """Summary: does negating the RESPONSIBLE head's difference change the verdict
    more than negating a non-responsible head? (Causal-explanation check.)"""
    base = (model(d, ctx) >= 0.5).float()
    rh = model.responsible_head(d, ctx)
    idx = torch.arange(len(d))
    d_r = d.clone(); d_r[idx, rh] *= -1
    flip_resp = float(((model(d_r, ctx) >= 0.5).float() != base).float().mean())
    other = (rh + 1) % model.n_heads
    d_o = d.clone(); d_o[idx, other] *= -1
    flip_other = float(((model(d_o, ctx) >= 0.5).float() != base).float().mean())
    return {"flip_when_responsible_negated": flip_resp,
            "flip_when_other_negated": flip_other,
            "faithful": flip_resp > flip_other}


def format_report(audits: list[ConflictAudit], limit: int = 5) -> str:
    lines = ["conflict | winner | conf | responsible head | contributions"]
    for i, a in enumerate(audits[:limit]):
        lines.append(f"  #{i:<3} |   {a.verdict}    | {a.confidence:.2f} | "
                     f"{a.responsible_head:8} | {a.contributions}")
    if len(audits) > limit:
        lines.append(f"  ... (+{len(audits) - limit} more)")
    return "\n".join(lines)
