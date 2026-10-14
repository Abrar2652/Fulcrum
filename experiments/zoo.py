"""Pairwise-comparison model zoo (transitive and non-transitive baselines, hard- and
soft-gated FULCRUM). All models share one interface so every benchmark script trains
and evaluates them identically.

Data convention (see `PairData`):
  d   : (N, m)  criterion-difference vector for the pair, d = c(a) - c(b). A criterion may
                be *pair-level* (e.g. a type-advantage term), in which case d_h is the
                antisymmetric pair term and it has no per-item counterpart.
  xa,xb: (N, k) per-item feature vectors (may include extra per-item features such as
                one-hots). Transitive models may use ONLY per-item features, because a
                logit of the form u(xa) - u(xb) is what makes them transitive.
  ia,ib: (N,)   item ids (>=0) when items repeat across comparisons, else -1.
  sa,sb: (N,)   optional scalar strength in [0,1] (1 = CERTAIN); zeros if absent.
  y    : (N,)   1 if a preferred to b, else 0.

Every model exposes `logit(batch) -> (N,)` and is antisymmetric in (a,b) by construction
except for the shared position intercept `b0`, which every model receives identically
(it captures a presentation-order effect, not a preference).

Transitive models  : BTLinear, BTMLP, Lexicographic (discrete), MaxGapRule is NOT transitive.
Non-transitive     : Condorcet, LinearPair (when d has pair-level criteria), ContextBTSum,
                     RandomGate, HardGate (FULCRUM-hard), SoftGate (FULCRUM-soft),
                     MixtureUtilities, Bilinear (antisymmetric low-rank), BladeChest
                     (item embeddings), PairMLP (antisymmetrised MLP).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data
@dataclass
class PairData:
    d: torch.Tensor
    xa: torch.Tensor
    xb: torch.Tensor
    y: torch.Tensor
    ia: torch.Tensor | None = None
    ib: torch.Tensor | None = None
    sa: torch.Tensor | None = None
    sb: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def to(self, dev) -> "PairData":
        f = lambda t: None if t is None else t.to(dev)  # noqa: E731
        return PairData(f(self.d), f(self.xa), f(self.xb), f(self.y), f(self.ia), f(self.ib),
                        f(self.sa), f(self.sb), self.meta)

    def subset(self, idx) -> "PairData":
        f = lambda t: None if t is None else t[idx]  # noqa: E731
        return PairData(f(self.d), f(self.xa), f(self.xb), f(self.y), f(self.ia), f(self.ib),
                        f(self.sa), f(self.sb), dict(self.meta))

    def swapped(self) -> "PairData":
        """The same comparisons with the two arguments exchanged (labels flipped)."""
        return PairData(-self.d, self.xb, self.xa, 1 - self.y, self.ib, self.ia,
                        self.sb, self.sa, dict(self.meta))

    @property
    def m(self) -> int:
        return int(self.d.shape[1])

    @property
    def k(self) -> int:
        return int(self.xa.shape[1])


def standardizer(train: PairData):
    """Per-criterion scale fitted on training differences (so |d_h| are comparable
    across criteria for the gates). Returns a function applying it to a PairData."""
    scale = train.d.abs().mean(0).clamp_min(1e-6)
    xscale = train.xa.abs().mean(0).clamp_min(1e-6)

    def apply(p: PairData) -> PairData:
        return PairData(p.d / scale, p.xa / xscale, p.xb / xscale, p.y, p.ia, p.ib,
                        p.sa, p.sb, dict(p.meta))
    return apply


# --------------------------------------------------------------------------- models
class PairModel(nn.Module):
    name = "base"
    transitive = False
    uses_items = False

    def __init__(self):
        super().__init__()
        self.b0 = nn.Parameter(torch.zeros(()))          # shared position intercept
        self.register_buffer("temperature", torch.tensor(1.0))

    def core(self, p: PairData) -> torch.Tensor:       # antisymmetric part
        raise NotImplementedError

    def logit(self, p: PairData) -> torch.Tensor:
        return self.core(p) + self.b0

    def forward(self, p: PairData) -> torch.Tensor:
        return torch.sigmoid(self.logit(p) / self.temperature)

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters() if q.requires_grad)


class BTLinear(PairModel):
    """Single ordering: u(x) = w.x, z = u(xa) - u(xb). Transitive."""
    name = "BT-linear"
    transitive = True

    def __init__(self, m, k, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(k))

    def core(self, p):
        return (p.xa - p.xb) @ self.w


class BTMLP(PairModel):
    """Single ordering with a nonlinear utility u = MLP(x). Transitive but nonlinear:
    the control that separates 'nonlinearity' from 'intransitivity'."""
    name = "BT-MLP"
    transitive = True

    def __init__(self, m, k, hidden=32, **kw):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(k, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                               nn.Tanh(), nn.Linear(hidden, 1))

    def core(self, p):
        return (self.f(p.xa) - self.f(p.xb)).squeeze(-1)


class LinearPair(PairModel):
    """Linear in the criterion differences, z = w.d. Equals BT-linear when every
    criterion is per-item; with a pair-level criterion it is the 'BT + pairwise term'
    model of the earlier draft (non-transitive)."""
    name = "linear-in-d"

    def __init__(self, m, k, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(m))

    def core(self, p):
        return p.d @ self.w


class Condorcet(PairModel):
    """Majority of criteria with a learned polarity per criterion (sign of the
    train correlation between d_h and y) and one learned scale. Non-transitive."""
    name = "Condorcet"

    def __init__(self, m, k, **kw):
        super().__init__()
        self.register_buffer("pol", torch.ones(m))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def fit_polarity(self, p: PairData):
        yc = 2 * p.y - 1
        self.pol = torch.sign((p.d * yc[:, None]).mean(0)).clamp_min(-1)
        self.pol[self.pol == 0] = 1.0

    def core(self, p):
        return self.scale * torch.sign(p.d * self.pol).sum(-1)


class MaxGapRule(PairModel):
    """The discrete MAX-GAP comparator as a predictor: the criterion with the largest
    standardized |d_h| decides with its (learned) polarity; one learned scale.
    'Fixed gate, fixed weights'."""
    name = "MAX-GAP rule"

    def __init__(self, m, k, **kw):
        super().__init__()
        self.register_buffer("pol", torch.ones(m))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def fit_polarity(self, p: PairData):
        yc = 2 * p.y - 1
        self.pol = torch.sign((p.d * yc[:, None]).mean(0))
        self.pol[self.pol == 0] = 1.0

    def core(self, p):
        c = p.d.abs().argmax(-1)
        return self.scale * (p.d * self.pol).gather(1, c[:, None]).squeeze(1)


class _Gated(PairModel):
    """Shared machinery: per-context weight vectors W (n_ctx x m), optional strength gate."""

    def __init__(self, m, k, n_ctx=None, strength_gate=False, **kw):
        super().__init__()
        self.m = m
        self.n_ctx = n_ctx or m
        self.W = nn.Parameter(torch.zeros(self.n_ctx, m))
        self.strength_gate = strength_gate
        if strength_gate:
            self.lam = nn.Parameter(torch.tensor(3.0))

    def _strength(self, p):
        if self.strength_gate and p.sa is not None:
            return self.lam.abs() * (p.sa - p.sb)
        return 0.0

    def contributions(self, p: PairData) -> torch.Tensor:
        """Per-criterion signed contributions kappa_h with sum_h kappa_h = core(p)."""
        raise NotImplementedError

    def core(self, p):
        return self.contributions(p).sum(-1) + self._strength(p)


class HardGate(_Gated):
    """FULCRUM, hard gate: context c = argmax_h |d_h|; z = W[c].d (+ strength gate).
    The only non-differentiable operation is the argmax over contexts; W and lam are
    trained by gradient descent. Contains MAX-GAP (W = I) exactly."""
    name = "FULCRUM-hard"

    def context(self, p):
        return p.d.abs().argmax(-1)

    def gate(self, p):
        return nn.functional.one_hot(self.context(p), self.n_ctx).float()

    def contributions(self, p):
        return self.W[self.context(p)] * p.d


class RandomGate(HardGate):
    """Ablation: the context is a pseudo-random function of the pair (a hash of the
    rounded difference vector), not the gap. Same parameter count as HardGate."""
    name = "random-gate"

    def context(self, p):
        h = (p.d * 1000).round().long()
        key = (h * torch.arange(1, self.m + 1, device=h.device)[None, :]).sum(-1)
        return (key.abs() * 2654435761) % self.n_ctx


class ContextBTSum(HardGate):
    """Context-conditioned BT with a non-gap endogenous context: the criterion on
    which the pair's *combined* (standardized) level |c_h(a)+c_h(b)| is largest."""
    name = "context-BT (sum ctx)"

    def context(self, p):
        return (p.xa[:, :self.m] + p.xb[:, :self.m]).abs().argmax(-1)


class SoftGate(_Gated):
    """FULCRUM, soft gate (Section 5 of the paper):
        g_c(a,b) = softmax_c( beta * |d_c| + b_c )        (symmetric in a,b)
        z(a,b)   = sum_c g_c(a,b) * W[c].d               (antisymmetric)
    kappa_h = (sum_c g_c W[c,h]) d_h are the criterion contributions.
    beta -> inf recovers the hard gate; beta = 0 gives one fixed mixture of the W rows,
    i.e. a single linear utility (transitive). Temperature T is fitted post hoc and is
    not beta."""
    name = "FULCRUM-soft"

    def __init__(self, m, k, n_ctx=None, strength_gate=False, beta_init=2.0,
                 learn_beta=True, gate_bias=True, **kw):
        super().__init__(m, k, n_ctx=n_ctx, strength_gate=strength_gate)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)), requires_grad=learn_beta)
        self.gb = nn.Parameter(torch.zeros(self.n_ctx), requires_grad=gate_bias)

    @property
    def beta(self):
        return self.log_beta.exp()

    def gate(self, p):
        return torch.softmax(self.beta * p.d.abs() + self.gb, dim=-1)

    def contributions(self, p):
        g = self.gate(p)                           # (N, n_ctx)
        w_eff = g @ self.W                         # (N, m): sum_c g_c W[c]
        return w_eff * p.d


class MixtureUtilities(PairModel):
    """Mixture of K linear utilities with constant mixture weights:
    P(a>b) = sum_k pi_k sigma(w_k.d + b0). Can represent cycles (a soft Condorcet
    paradox across components); no criterion structure."""
    name = "mixture-of-utilities"

    def __init__(self, m, k, K=None, **kw):
        super().__init__()
        K = K or m
        self.W = nn.Parameter(0.01 * torch.randn(K, m))
        self.alpha = nn.Parameter(torch.zeros(K))

    def core(self, p):                             # used only for attribution-free paths
        return (torch.softmax(self.alpha, 0)[None, :] * (p.d @ self.W.T)).sum(-1)

    def forward(self, p):
        z = (p.d @ self.W.T + self.b0) / self.temperature
        return (torch.softmax(self.alpha, 0)[None, :] * torch.sigmoid(z)).sum(-1)

    def logit(self, p):
        pr = self.forward(p).clamp(1e-6, 1 - 1e-6)
        return torch.log(pr) - torch.log1p(-pr)


class Bilinear(PairModel):
    """Antisymmetric low-rank matchup model on per-item features:
    z = w.d + xa^T A xb with A = U V^T - V U^T (rank 2r). This is the feature-based
    blade-chest-inner model of Chen & Joachims (2016). Non-transitive."""
    name = "antisym-bilinear"

    def __init__(self, m, k, rank=2, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(m))
        self.U = nn.Parameter(0.05 * torch.randn(k, rank))
        self.V = nn.Parameter(0.05 * torch.randn(k, rank))

    def core(self, p):
        ua, ub = p.xa @ self.U, p.xb @ self.U
        va, vb = p.xa @ self.V, p.xb @ self.V
        return p.d @ self.w + (ua * vb).sum(-1) - (va * ub).sum(-1)


class BladeChest(PairModel):
    """Blade-chest-inner with free item embeddings (Chen & Joachims 2016):
    z = b_i.c_j - b_j.c_i + gamma_i - gamma_j (+ w.d). Requires repeated items."""
    name = "blade-chest"
    uses_items = True

    def __init__(self, m, k, n_items=1, dim=8, with_d=True, **kw):
        super().__init__()
        self.blade = nn.Embedding(n_items, dim)
        self.chest = nn.Embedding(n_items, dim)
        self.gamma = nn.Embedding(n_items, 1)
        for e in (self.blade, self.chest, self.gamma):
            nn.init.normal_(e.weight, std=0.05)
        self.with_d = with_d
        self.w = nn.Parameter(torch.zeros(m)) if with_d else None

    def core(self, p):
        bi, bj = self.blade(p.ia), self.blade(p.ib)
        ci, cj = self.chest(p.ia), self.chest(p.ib)
        z = (bi * cj).sum(-1) - (bj * ci).sum(-1) + (self.gamma(p.ia) - self.gamma(p.ib)).squeeze(-1)
        if self.with_d:
            z = z + p.d @ self.w
        return z


class PairMLP(PairModel):
    """Antisymmetrised MLP comparator: z = f([xa, xb, d]) - f([xb, xa, -d])."""
    name = "pair-MLP"

    def __init__(self, m, k, hidden=64, **kw):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * k + m, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def core(self, p):
        fa = self.f(torch.cat([p.xa, p.xb, p.d], -1))
        fb = self.f(torch.cat([p.xb, p.xa, -p.d], -1))
        return (fa - fb).squeeze(-1)


class Lexicographic:
    """Discrete transitive baseline: one fixed criterion ordering (chosen on the train
    set among all m! permutations for m <= 7, greedy otherwise); the first criterion
    with a nonzero difference decides, with a learned polarity. Not a torch module."""
    name = "lexicographic"
    transitive = True
    uses_items = False

    def __init__(self, m, k, **kw):
        self.m = m
        self.order = tuple(range(m))
        self.pol = torch.ones(m)
        self.temperature = 1.0

    def n_params(self):
        return 0

    def _decide(self, d, order, pol):
        out = torch.zeros(d.shape[0], device=d.device)
        undecided = torch.ones(d.shape[0], dtype=torch.bool, device=d.device)
        for h in order:
            v = torch.sign(d[:, h] * pol[h])
            take = undecided & (v != 0)
            out[take] = v[take]
            undecided &= ~take
        return out

    def fit(self, p: PairData):
        yc = 2 * p.y - 1
        self.pol = torch.sign((p.d * yc[:, None]).mean(0)).to(p.d.device)
        self.pol[self.pol == 0] = 1
        best, best_acc = None, -1.0
        perms = itertools.permutations(range(self.m)) if self.m <= 7 else None
        if perms is None:
            # greedy: pick the best first criterion, then the best second, ...
            remaining = list(range(self.m)); order = []
            while remaining:
                cand = max(remaining, key=lambda h: float(
                    (self._decide(p.d, tuple(order + [h]), self.pol) == yc).float().mean()))
                order.append(cand); remaining.remove(cand)
            self.order = tuple(order)
            return self
        for perm in perms:
            acc = float((self._decide(p.d, perm, self.pol) == yc).float().mean())
            if acc > best_acc:
                best, best_acc = perm, acc
        self.order = best
        return self

    def logit(self, p: PairData):
        return 3.0 * self._decide(p.d, self.order, self.pol.to(p.d.device))

    def forward(self, p: PairData):
        return torch.sigmoid(self.logit(p) / self.temperature)

    __call__ = forward


# --------------------------------------------------------------------------- training
def bce_logits(z, y):
    return nn.functional.binary_cross_entropy_with_logits(z, y)


def _pred_logit(model, p):
    if isinstance(model, MixtureUtilities):
        return model.logit(p)
    return model.logit(p)


def train_model(model, tr: PairData, va: PairData | None = None, epochs=400, lr=0.05,
                weight_decay=1e-4, batch_size=None, seed=0, patience=50, verbose=False):
    """Full-batch (or minibatch) Adam on BCE; early stopping on validation NLL when a
    validation split is given. Returns (model, seconds)."""
    torch.manual_seed(seed)
    t0 = time.time()
    if isinstance(model, Lexicographic):
        model.fit(tr)
        return model, time.time() - t0
    model = model.to(tr.d.device)
    if hasattr(model, "fit_polarity"):
        model.fit_polarity(tr)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    n = len(tr)
    best_state, best_val, bad = None, float("inf"), 0
    for ep in range(epochs):
        model.train()
        if batch_size is None or batch_size >= n:
            opt.zero_grad()
            if isinstance(model, MixtureUtilities):
                loss = nn.functional.binary_cross_entropy(model(tr).clamp(1e-6, 1 - 1e-6), tr.y)
            else:
                loss = bce_logits(model.logit(tr), tr.y)
            loss.backward(); opt.step()
        else:
            perm = torch.randperm(n, device=tr.d.device)
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                b = tr.subset(idx)
                opt.zero_grad()
                if isinstance(model, MixtureUtilities):
                    loss = nn.functional.binary_cross_entropy(model(b).clamp(1e-6, 1 - 1e-6), b.y)
                else:
                    loss = bce_logits(model.logit(b), b.y)
                loss.backward(); opt.step()
        if va is not None and (ep % 5 == 4 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                if isinstance(model, MixtureUtilities):
                    vl = float(nn.functional.binary_cross_entropy(model(va).clamp(1e-6, 1 - 1e-6), va.y))
                else:
                    vl = float(bce_logits(model.logit(va), va.y))
            if vl < best_val - 1e-5:
                best_val, bad = vl, 0
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            else:
                bad += 5
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, time.time() - t0


@torch.no_grad()
def calibrate_temperature(model, va: PairData, steps=300):
    """Post-hoc temperature scaling (Guo et al. 2017) on the validation split."""
    if isinstance(model, Lexicographic):
        z = model.logit(va)
    else:
        z = model.logit(va)
    with torch.enable_grad():
        logT = torch.zeros(1, device=z.device, requires_grad=True)
        opt = torch.optim.Adam([logT], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = bce_logits(z / logT.exp(), va.y)
            loss.backward(); opt.step()
    T = float(logT.exp())
    if isinstance(model, Lexicographic):
        model.temperature = T
    else:
        model.temperature = torch.tensor(T, device=z.device)
    return T


@torch.no_grad()
def predict_proba(model, p: PairData) -> torch.Tensor:
    if isinstance(model, Lexicographic):
        return model(p)
    if isinstance(model, MixtureUtilities):
        return model(p)
    return torch.sigmoid(model.logit(p) / model.temperature)


def predict_proba_core(model, p: PairData) -> torch.Tensor:
    """Probability from the antisymmetric core only (no shared position intercept b0).
    Used for relation-level metrics (cycle orientation): a transitive model's core is
    induced by one utility, so its aggregated relation is acyclic, whereas core + b0
    evaluated on both presentation orders of a repeated pair need not be."""
    if isinstance(model, Lexicographic):
        return model(p)
    if isinstance(model, MixtureUtilities):
        return model(p)
    return torch.sigmoid(model.core(p) / model.temperature)


# --------------------------------------------------------------------------- registry
def make_zoo(m, k, n_items=0, include=None, exclude=(), mlp_small_hidden=None,
             strength_gate=False):
    """Instantiate the standard zoo. `mlp_small_hidden` sets the parameter-matched MLP
    width (defaults to the width that matches FULCRUM-hard's m^2+1 parameters)."""
    if mlp_small_hidden is None:
        # PairMLP params = (2k+m+1)h + (h+1); solve ~= m^2 + 1
        mlp_small_hidden = max(1, round((m * m + 1) / (2 * k + m + 2)))
    zoo = {
        "BT-linear": lambda: BTLinear(m, k),
        "BT-MLP": lambda: BTMLP(m, k, hidden=32),
        "lexicographic": lambda: Lexicographic(m, k),
        "Condorcet": lambda: Condorcet(m, k),
        "MAX-GAP rule": lambda: MaxGapRule(m, k),
        "linear-in-d": lambda: LinearPair(m, k),
        "context-BT (sum ctx)": lambda: ContextBTSum(m, k),
        "random-gate": lambda: RandomGate(m, k),
        "mixture-of-utilities": lambda: MixtureUtilities(m, k, K=m),
        "antisym-bilinear": lambda: Bilinear(m, k, rank=2),
        "pair-MLP (matched)": lambda: PairMLP(m, k, hidden=mlp_small_hidden),
        "pair-MLP (large)": lambda: PairMLP(m, k, hidden=64),
        "FULCRUM-hard": lambda: HardGate(m, k, strength_gate=strength_gate),
        "FULCRUM-soft": lambda: SoftGate(m, k, strength_gate=strength_gate),
    }
    if n_items > 0:
        zoo["blade-chest"] = lambda: BladeChest(m, k, n_items=n_items, dim=8)
    if include is not None:
        zoo = {n_: f for n_, f in zoo.items() if n_ in include}
    return {n_: f for n_, f in zoo.items() if n_ not in exclude}


TRANSITIVE = {"BT-linear", "BT-MLP", "lexicographic"}
