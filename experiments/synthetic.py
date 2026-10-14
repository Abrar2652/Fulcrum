"""Synthetic conflict generator with KNOWN ground truth (Phase 6, pre-registered).

Design (non-circular): a 'true' resolution ORACLE decides each conflict; both the
fulcrum calculus and the baselines are FIT to a training split and evaluated on a
held-out split. The oracle is NOT the calculus's policy — so a win reflects the
calculus's ability to *capture the world's structure*, not the answer being handed
to it.

Two regimes (per the pre-registration notes §2):
  - 'context'   : the deciding head is a function of a per-conflict context tag
                  (a context-dependent world). The calculus CAN condition on the
                  tag; a single fixed ordering cannot. Calculus should win.
  - 'transitive': the deciding head is fixed for all conflicts (a single-ordering
                  world). NEGATIVE CONTROL: calculus and baseline should tie.

Label noise (epsilon) makes neither model perfect, so we measure the GAP.
Deterministic given seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from fulcrum.structures import HEAD_ORDER

N_HEADS = len(HEAD_ORDER)  # 4


@dataclass(frozen=True)
class Conflict:
    va: tuple[int, ...]     # head-score vector of arg a
    vb: tuple[int, ...]     # head-score vector of arg b
    ctx: int                # context tag
    winner: int             # ground-truth winner: +1 (a) or -1 (b)


def _decide_by_head(va, vb, h) -> int:
    """Higher score on head h wins; tie broken toward a (+1)."""
    return 1 if va[h] >= vb[h] else -1


def make_dataset(n: int, seed: int, regime: str, n_ctx: int = 3,
                 epsilon: float = 0.1, vmax: int = 5):
    """Return a list of Conflicts. `regime` in {'context','transitive'}."""
    rng = random.Random(seed)
    # oracle: mapping context-tag -> deciding head
    if regime == "transitive":
        fixed = rng.randrange(N_HEADS)
        head_of_ctx = {c: fixed for c in range(n_ctx)}          # same head for all
    elif regime == "context":
        # distinct deciding heads per context (context-dependent world)
        heads = list(range(N_HEADS))
        rng.shuffle(heads)
        head_of_ctx = {c: heads[c % N_HEADS] for c in range(n_ctx)}
    else:
        raise ValueError(regime)

    data = []
    for _ in range(n):
        ctx = rng.randrange(n_ctx)
        # ensure the deciding head actually discriminates (else the label is noise)
        h = head_of_ctx[ctx]
        while True:
            va = tuple(rng.randint(0, vmax) for _ in range(N_HEADS))
            vb = tuple(rng.randint(0, vmax) for _ in range(N_HEADS))
            if va[h] != vb[h]:
                break
        w = _decide_by_head(va, vb, h)
        if rng.random() < epsilon:      # label noise
            w = -w
        data.append(Conflict(va, vb, ctx, w))
    return data, head_of_ctx


@dataclass(frozen=True)
class StrengthConflict:
    va: tuple[int, ...]
    vb: tuple[int, ...]
    ctx: int
    sa: float          # strength of a in [0,1] (1 = certain/certain, 0 = probable/probable)
    sb: float
    winner: int


def make_strength_dataset(n: int, seed: int, n_ctx: int = 4,
                          eps_certain: float = 0.02, eps_probable: float = 0.25,
                          p_certain: float = 0.35, vmax: int = 5):
    """Conflicts where RELIABILITY depends on strength (the H2 mechanism):

      - one certain (certain) + one probable (probable): the CERTAIN side wins, and the
        label is highly reliable (noise eps_certain ~ 0). The certain gate.
      - both probable (probable): decided by a context-dependent head, but NOISY
        (noise eps_probable, large) -- probable evidence is less reliable.
      - both certain: rare here; decided by a fixed head with low noise.

    A model that USES strength can be confident on certain-involved conflicts and
    uncertain on probable-probable ones -> better calibrated (H2). A model that
    ignores strength cannot distinguish these regimes -> miscalibrated.
    """
    rng = random.Random(seed)
    heads = list(range(N_HEADS)); rng.shuffle(heads)
    head_of_ctx = {c: heads[c % N_HEADS] for c in range(n_ctx)}
    data = []
    for _ in range(n):
        sa = 1.0 if rng.random() < p_certain else 0.0
        sb = 1.0 if rng.random() < p_certain else 0.0
        ctx = rng.randrange(n_ctx)
        h = head_of_ctx[ctx]
        while True:
            va = tuple(rng.randint(0, vmax) for _ in range(N_HEADS))
            vb = tuple(rng.randint(0, vmax) for _ in range(N_HEADS))
            if va[h] != vb[h]:
                break
        if sa != sb:                        # certain vs probable -> certain wins, low noise
            w = 1 if sa > sb else -1
            eps = eps_certain
        elif sa == 0.0:                     # both probable -> preponderance, high noise
            w = _decide_by_head(va, vb, h)
            eps = eps_probable
        else:                               # both certain (rare) -> fixed head, low noise
            w = _decide_by_head(va, vb, 0)
            eps = eps_certain
        if rng.random() < eps:
            w = -w
        data.append(StrengthConflict(va, vb, ctx, sa, sb, w))
    return data


def split(data, frac_train: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    k = int(len(data) * frac_train)
    tr = [data[i] for i in idx[:k]]
    te = [data[i] for i in idx[k:]]
    return tr, te
