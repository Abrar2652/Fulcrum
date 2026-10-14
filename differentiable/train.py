"""Training + evaluation for the differentiable preponderance (Phase 5).

Trains on the synthetic conflict generator (experiments/synthetic.py), reports
accuracy, CALIBRATION (ECE/Brier -> pre-registered H2), and AUDITABILITY. Compares
the learned context-conditioned model against its fixed-weight ablation (the learned
version must EARN its complexity, the project design notes Phase 5).

Usage: python differentiable/train.py
"""
from __future__ import annotations

import torch

from differentiable.model import DifferentiablePreponderance
from experiments.synthetic import N_HEADS, make_dataset, split


def _tensors(conflicts):
    d = torch.tensor([[c.va[h] - c.vb[h] for h in range(N_HEADS)] for c in conflicts],
                     dtype=torch.float32)
    ctx = torch.tensor([c.ctx for c in conflicts], dtype=torch.long)
    y = torch.tensor([1.0 if c.winner == 1 else 0.0 for c in conflicts])
    return d, ctx, y


def train_model(model, d, ctx, y, epochs=400, lr=0.05, seed=0, weight_decay=1e-2):
    """Train with L2 weight decay -> avoids the overconfidence (weight blow-up)
    that wrecks calibration when logistic weights are trained to convergence."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        p = model(d, ctx).clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(p, y)
        loss.backward()
        opt.step()
    return model


def calibrate_temperature(model, d, ctx, y, steps=200):
    """Post-hoc temperature scaling (Guo et al. 2017): fit a scalar T on a
    validation split to minimize NLL, without touching the learned weights.
    Standard, honest calibration; leaves accuracy unchanged."""
    with torch.no_grad():
        z = model.logit(d, ctx)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(z / logT.exp(), y)
        loss.backward()
        opt.step()
    model.temperature = logT.exp().detach().squeeze()
    return model


@torch.no_grad()
def brier(p, y):
    return float(((p - y) ** 2).mean())


@torch.no_grad()
def ece(p, y, bins=10):
    """Standard ECE: bin by CONFIDENCE = max(p, 1-p) in [0.5,1]; per bin compare
    mean confidence to accuracy of the predicted class."""
    pred = (p >= 0.5).float()
    conf = torch.maximum(p, 1 - p)
    correct = (pred == y).float()
    edges = torch.linspace(0.5, 1.0, bins + 1)
    n = len(p)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += float(m.sum()) / n * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return total


@torch.no_grad()
def accuracy(model, d, ctx, y):
    pred = (model(d, ctx) >= 0.5).float()
    return float((pred == y).float().mean())


@torch.no_grad()
def auditability(model, d, ctx, y):
    """Soft H3: flipping the RESPONSIBLE head's difference changes the predicted
    winner more often than flipping a random other head."""
    base = (model(d, ctx) >= 0.5).float()
    rh = model.responsible_head(d, ctx)
    d_resp = d.clone()
    d_resp[torch.arange(len(d)), rh] *= -1
    flip_resp = ((model(d_resp, ctx) >= 0.5).float() != base).float().mean()
    # random other head
    other = (rh + 1) % N_HEADS
    d_rand = d.clone()
    d_rand[torch.arange(len(d)), other] *= -1
    flip_rand = ((model(d_rand, ctx) >= 0.5).float() != base).float().mean()
    return float(flip_resp), float(flip_rand)


def evaluate(model, d, ctx, y):
    p = model(d, ctx)
    fr, fo = auditability(model, d, ctx, y)
    return {"acc": accuracy(model, d, ctx, y), "brier": brier(p, y),
            "ece": ece(p, y), "audit_responsible": fr, "audit_other": fo}


def run(regime="context", n=1800, seed=0, calibrate=True):
    data, _ = make_dataset(n, seed=seed, regime=regime, n_ctx=N_HEADS)
    trv, te = split(data, 0.6, seed=seed + 1)
    tr, val = split(trv, 0.8, seed=seed + 2)          # train / val (for temp scaling) / test
    dtr, ctr, ytr = _tensors(tr)
    dval, cval, yval = _tensors(val)
    dte, cte, yte = _tensors(te)
    out = {}
    for name, cc in (("learned", True), ("fixed", False)):
        m = train_model(
            DifferentiablePreponderance(N_HEADS, n_ctx=N_HEADS, context_conditioned=cc),
            dtr, ctr, ytr, seed=seed)
        if calibrate:
            calibrate_temperature(m, dval, cval, yval)
        out[name] = evaluate(m, dte, cte, yte)
    return out


def main():
    print("=" * 66)
    print("PHASE 5 — differentiable preponderance: learned vs fixed-weight")
    print("=" * 66)
    for regime, note in [("context", "context-dependent -> learned should win + be calibrated"),
                         ("transitive", "single-ordering -> negative control: tie")]:
        r = run(regime)
        print(f"\n[{regime}]  {note}")
        for name in ("learned", "fixed"):
            m = r[name]
            print(f"  {name:8} acc={m['acc']:.3f}  ECE={m['ece']:.3f}  Brier={m['brier']:.3f}  "
                  f"audit(resp={m['audit_responsible']:.2f} vs other={m['audit_other']:.2f})")


if __name__ == "__main__":
    main()
