"""Attribution-faithfulness diagnostics for criterion-level attributions.

For a model that exposes per-criterion contributions kappa_h (sum_h kappa_h = logit), the
attributed criterion is h* = argmax_h |kappa_h|. 'Largest term' is not faithfulness; the
diagnostics below intervene on the INPUT difference vector d (so gates and every downstream
quantity react) and compare against two controls, a random other criterion and the
least-attributed criterion:

  sign reversal   : d_{h*} <- -d_{h*}   (swap the two arguments' scores on h*)  -> flip rate
  deletion        : d_{h*} <- 0                                                  -> flip rate
  comprehensiveness: mean decrease in probability of the original verdict after deletion
  sufficiency     : keep only d_{h*} (all other criteria <- 0)                    -> agreement rate

For models without built-in contributions (e.g. the pair-MLP), `occlusion_attribution`
defines h* as the criterion whose deletion changes the logit most, so the same diagnostics
can be run for comparison (with the caveat that this attribution is itself occlusion-based).
"""
from __future__ import annotations

import torch

from experiments.zoo import PairData, predict_proba


@torch.no_grad()
def builtin_attribution(model, p: PairData) -> torch.Tensor:
    return model.contributions(p).abs().argmax(-1)


@torch.no_grad()
def occlusion_attribution(model, p: PairData) -> torch.Tensor:
    base = model.logit(p)
    effects = []
    for h in range(p.m):
        q = _with_d(p, p.d.clone())
        q.d[:, h] = 0.0
        effects.append((model.logit(q) - base).abs())
    return torch.stack(effects, -1).argmax(-1)


def _with_d(p: PairData, d: torch.Tensor) -> PairData:
    return PairData(d, p.xa, p.xb, p.y, p.ia, p.ib, p.sa, p.sb, dict(p.meta))


@torch.no_grad()
def diagnostics(model, p: PairData, attribution: torch.Tensor, seed: int = 0) -> dict:
    """Run the four diagnostics for the given attributed criterion and both controls."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, m = p.d.shape
    base_p = predict_proba(model, p)
    base_v = (base_p >= 0.5)
    conf = torch.where(base_v, base_p, 1 - base_p)
    idx = torch.arange(n, device=p.d.device)
    # controls
    rand_other = (attribution.cpu() + torch.randint(1, m, (n,), generator=g)).to(p.d.device) % m
    if hasattr(model, "contributions"):
        least = model.contributions(p).abs().argmin(-1)
    else:
        least = rand_other

    def run(h: torch.Tensor) -> dict:
        out = {}
        d = p.d.clone(); d[idx, h] = -d[idx, h]
        out["flip_sign"] = float(((predict_proba(model, _with_d(p, d)) >= 0.5) != base_v).float().mean())
        d = p.d.clone(); d[idx, h] = 0.0
        pd_ = predict_proba(model, _with_d(p, d))
        out["flip_delete"] = float(((pd_ >= 0.5) != base_v).float().mean())
        conf_del = torch.where(base_v, pd_, 1 - pd_)
        out["comprehensiveness"] = float((conf - conf_del).mean())
        d = torch.zeros_like(p.d); d[idx, h] = p.d[idx, h]
        out["sufficiency"] = float(((predict_proba(model, _with_d(p, d)) >= 0.5) == base_v).float().mean())
        return out
    return {"attributed": run(attribution), "random_other": run(rand_other), "least": run(least)}
