"""Real conflicting-criteria benchmark: Pokemon battles (genuine, designed
non-transitive dominance via the type chart).

Two results:
  (1) INTRANSITIVITY IS REAL HERE. Unlike naturally-correlated preference data
      (sushi/Netflix ~0%), the type-advantage relation is intransitive for a large
      fraction of triples -- a genuinely conflicting-criteria domain.
  (2) A single ordering CANNOT capture it. On 50k real battles, a Bradley-Terry
      single ordering (a total order over Pokemon; the ASPIC+/VAF-faithful baseline)
      -- even given type one-hots -- is beaten by a context-dependent model that adds
      the intransitive pairwise type-advantage term (the "deciding head" in a
      cross-type battle). The gain concentrates on cross-type battles.

Data: Kaggle "Weedle's Cave" (combats.csv, 50k battles; pokemon.csv, 800 mons with
6 stats + dual types), id-aligned. Type chart = standard Gen-6 (18 types).
"""
from __future__ import annotations

import csv
import itertools
import math

import torch

DATA = "data/pokemon"
TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
         "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
         "Steel", "Fairy"]
TI = {t: i for i, t in enumerate(TYPES)}

# CHART[attacker] = {defender: multiplier} for non-1.0 entries (standard Gen 6+).
CHART = {
    "Normal": {"Rock": .5, "Ghost": 0, "Steel": .5},
    "Fire": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": .5,
             "Dragon": .5, "Steel": 2},
    "Water": {"Fire": 2, "Water": .5, "Grass": .5, "Ground": 2, "Rock": 2, "Dragon": .5},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Ground": 0, "Flying": 2,
                 "Dragon": .5},
    "Grass": {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5, "Ground": 2,
              "Flying": .5, "Bug": .5, "Rock": 2, "Dragon": .5, "Steel": .5},
    "Ice": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2, "Flying": 2,
            "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": .5, "Flying": .5, "Psychic": .5,
                 "Bug": .5, "Rock": 2, "Ghost": 0, "Dark": 2, "Steel": 2, "Fairy": .5},
    "Poison": {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5, "Ghost": .5,
               "Steel": 0, "Fairy": 2},
    "Ground": {"Fire": 2, "Electric": 2, "Grass": .5, "Poison": 2, "Flying": 0,
               "Bug": .5, "Rock": 2, "Steel": 2},
    "Flying": {"Electric": .5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": .5,
               "Steel": .5},
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": .5, "Dark": 0, "Steel": .5},
    "Bug": {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5, "Flying": .5,
            "Psychic": 2, "Ghost": .5, "Dark": 2, "Steel": .5, "Fairy": .5},
    "Rock": {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5, "Flying": 2, "Bug": 2,
             "Steel": .5},
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon": {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark": {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5, "Fairy": .5},
    "Steel": {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2, "Rock": 2, "Steel": .5,
              "Fairy": 2},
    "Fairy": {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2, "Dark": 2,
              "Steel": .5},
}


def mult(att: str, deff: str) -> float:
    return CHART.get(att, {}).get(deff, 1.0)


def load():
    mons = {}
    with open(f"{DATA}/pokemon.csv") as f:
        for r in csv.DictReader(f):
            i = int(r["#"])
            t = [r["Type 1"].strip()]
            if r["Type 2"].strip():
                t.append(r["Type 2"].strip())
            stats = [float(r[k]) for k in ("HP", "Attack", "Defense", "Sp. Atk",
                                           "Sp. Def", "Speed")]
            mons[i] = {"types": t, "stats": stats}
    battles = []
    with open(f"{DATA}/combats.csv") as f:
        for r in csv.DictReader(f):
            a, b, w = int(r["First_pokemon"]), int(r["Second_pokemon"]), int(r["Winner"])
            if a in mons and b in mons:
                battles.append((a, b, 1.0 if w == a else 0.0))
    return mons, battles


def off(att_types, def_types):
    """Best offensive multiplier of an attacker (its STAB types) vs a defender."""
    best = 0.0
    for at in att_types:
        m = 1.0
        for dt in def_types:
            m *= mult(at, dt)
        best = max(best, m)
    return best


def type_adv(mons, i, j):
    """log-ratio type advantage of i over j (antisymmetric; the intransitive term)."""
    oi = off(mons[i]["types"], mons[j]["types"])
    oj = off(mons[j]["types"], mons[i]["types"])
    return math.log(oi + 1e-6) - math.log(oj + 1e-6)


# ---------- (1) intransitivity of the type-advantage relation ----------
def type_intransitivity():
    dec = cyc = 0
    for a, b, c in itertools.combinations(TYPES, 3):
        def beat(x, y):
            dx, dy = mult(x, y), mult(y, x)
            return 1 if dx > dy else (-1 if dy > dx else 0)
        ab, bc, ca = beat(a, b), beat(b, c), beat(c, a)
        if 0 in (ab, bc, ca):
            continue
        dec += 1
        if ab == bc == ca == 1 or ab == bc == ca == -1:
            cyc += 1
    return cyc, dec


def mon_intransitivity(mons, n_triples=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = list(mons.keys())
    dec = cyc = 0
    for _ in range(n_triples):
        i, j, k = (ids[t] for t in torch.randint(0, len(ids), (3,), generator=g).tolist())
        if len({i, j, k}) < 3:
            continue
        def beat(x, y):
            t = type_adv(mons, x, y)
            return 1 if t > 1e-6 else (-1 if t < -1e-6 else 0)
        ij, jk, ki = beat(i, j), beat(j, k), beat(k, i)
        if 0 in (ij, jk, ki):
            continue
        dec += 1
        if ij == jk == ki == 1 or ij == jk == ki == -1:
            cyc += 1
    return cyc / max(dec, 1), dec


# ---------- (2) single ordering vs context-dependent prediction ----------
def featurize(mons, battles):
    ids = list(mons.keys())
    S = torch.tensor([mons[i]["stats"] for i in ids])
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    idx = {i: k for k, i in enumerate(ids)}
    onehot = torch.zeros(len(ids), len(TYPES))
    for k, i in enumerate(ids):
        for t in mons[i]["types"]:
            onehot[k, TI[t]] = 1.0
    di, dj, y, tadv = [], [], [], []
    for a, b, lab in battles:
        di.append(idx[a]); dj.append(idx[b]); y.append(lab)
        tadv.append(type_adv(mons, a, b))
    return (S, onehot, torch.tensor(di), torch.tensor(dj),
            torch.tensor(y), torch.tensor(tadv))


def fit_eval(feats, model, seed, epochs=250):
    S, OH, di, dj, y, tadv = feats
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    ntr = int(0.8 * len(y)); tr, te = perm[:ntr], perm[ntr:]
    m = S.shape[1]
    w = torch.zeros(m, requires_grad=True)          # stat weights (per-item utility)
    wt = torch.zeros(len(TYPES), requires_grad=True)  # type-onehot weights (single-B)
    beta = torch.zeros(1, requires_grad=True)        # intransitive type-adv term (fulcrum)
    params = {"single_stats": [w], "single_types": [w, wt], "fulcrum": [w, beta]}[model]
    opt = torch.optim.Adam(params, lr=0.05)

    def logit(ix):
        u = S @ w
        z = u[di[ix]] - u[dj[ix]]
        if model == "single_types":
            ut = OH @ wt
            z = z + ut[di[ix]] - ut[dj[ix]]
        if model == "fulcrum":
            z = z + beta * tadv[ix]
        return z
    for _ in range(epochs):
        opt.zero_grad()
        p = torch.sigmoid(logit(tr)).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(p, y[tr])
        loss.backward(); opt.step()
    with torch.no_grad():
        correct = ((torch.sigmoid(logit(te)) >= .5).float() == y[te]).float()
        acc = correct.mean().item()
        cross = tadv[te].abs() > 1e-6   # cross-type decisive battles
        acc_cross = correct[cross].mean().item()
        statgap = (S @ w.detach())[di[te]] - (S @ w.detach())[dj[te]]  # transitive signal
    return {"acc": acc, "acc_cross": acc_cross, "correct": correct,
            "statgap": statgap, "tadv_te": tadv[te].detach(), "te": te}


def ci95(xs):
    n = len(xs); mu = sum(xs) / n
    v = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mu, 1.96 * math.sqrt(v / n)


def main():
    mons, battles = load()
    print(f"loaded {len(mons)} Pokemon, {len(battles)} battles")

    cyc, dec = type_intransitivity()
    print("\n(1) INTRANSITIVITY (genuine, designed conflicting criteria)")
    print(f"  type chart (18 types): {cyc}/{dec} decisive triples cycle "
          f"= {cyc/dec:.3f}")
    rate, mdec = mon_intransitivity(mons)
    print(f"  Pokemon type-advantage: {rate:.3f} of decisive triples cycle "
          f"(n={mdec})   [cf. sushi 0.0002, Netflix 0.000]")

    feats = featurize(mons, battles)
    print("\n(2) SINGLE ORDERING vs CONTEXT-DEPENDENT (held-out battle accuracy, 5 seeds)")
    res = {}
    runs = {m: [fit_eval(feats, m, s) for s in range(5)] for m in
            ("single_stats", "single_types", "fulcrum")}
    for model, rows in runs.items():
        acc = ci95([r["acc"] for r in rows]); accc = ci95([r["acc_cross"] for r in rows])
        res[model] = (acc, accc)
        print(f"  {model:14s} acc={acc[0]:.3f}+-{acc[1]:.3f}   "
              f"cross-type acc={accc[0]:.3f}+-{accc[1]:.3f}")
    d = res["fulcrum"][0][0] - res["single_types"][0][0]
    print(f"\n  fulcrum - best single ordering (overall): +{d:.3f}")

    # Principled slices where the intransitive type structure must decide, with PAIRED
    # per-seed CIs (delta = fulcrum - single_types on the SAME battles, same seed).
    print("\n(3) Where the intransitive structure decides (paired per-seed 95% CI)")

    def slice_report(name, mask_fn):
        s_acc, f_acc, d = [], [], []
        for si in range(5):
            rs, rf = runs["single_types"][si], runs["fulcrum"][si]
            m = mask_fn(si)
            sa = rs["correct"][m].mean().item(); fa = rf["correct"][m].mean().item()
            s_acc.append(sa); f_acc.append(fa); d.append(fa - sa)
        sm, fm, dm = ci95(s_acc)[0], ci95(f_acc)[0], ci95(d)
        sig = "sig" if (dm[0] - dm[1]) > 0 else "ns"
        print(f"  {name:28s} single={sm:.3f} fulcrum={fm:.3f} "
              f"delta=+{dm[0]:.3f}+-{dm[1]:.3f} [{sig}]")

    def statgap(si):
        return runs["single_stats"][si]["statgap"].abs()

    def tadv(si):
        return runs["single_types"][si]["tadv_te"].abs()  # same te across models (same seed)

    # a-priori slice: cross-type decisive battles (types differ) -- not post-hoc
    slice_report("cross-type (a priori)", lambda si: tadv(si) > 1e-6)
    # cross-type AND stat-close: where types must decide (principled intersection)
    slice_report("cross-type & stat-close 30%",
                 lambda si: (tadv(si) > 1e-6) & (statgap(si) <= torch.quantile(statgap(si), 0.3)))
    # threshold robustness for the stat-close slice (show it isn't a cherry-picked cutoff)
    for frac in (0.1, 0.2, 0.3):
        slice_report(f"stat-close {int(frac*100)}% (robustness)",
                     lambda si, f=frac: statgap(si) <= torch.quantile(statgap(si), f))


if __name__ == "__main__":
    main()
