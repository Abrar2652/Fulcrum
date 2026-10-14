"""Transitivity ceilings: the best pairwise accuracy any single ordering can reach on a
given set of comparisons (Theorem 'transitivity ceiling' in the paper).

For a multiset of observed comparisons over items, let n_ij be the number of times i beat
j. A linear order L predicts i>j for every pair it ranks i above j, so its number of
errors is  sum_{pairs} n_{against L}.  Minimising over L is the (weighted) minimum
feedback arc set / Kemeny problem on the majority tournament:

    errors(L) = sum_{pairs} min(n_ij, n_ji)  +  sum_{pairs L disagrees with majority} |n_ij - n_ji|
              =        pair_loss             +        weighted FAS of L on the majority digraph.

Functions
---------
pair_ceiling(counts)           : 1 - pair_loss / N   (best any pairwise predictor can do)
transitive_ceiling_exact(...)  : exact, by ILP (linear-ordering formulation with all triangle
                                 constraints; fine for complete tournaments up to n ~ 60-80).
transitive_ceiling_bounds(...) : [lower, upper] bounds for large sparse graphs: an
                                 achievable ordering (local search) gives the lower bound;
                                 the LP relaxation of the cycle-covering formulation with
                                 lazy cycle constraints gives the upper bound (LP <= ILP).
triangle_bound(...)            : the closed-form bound 1 - f3/3 for complete tournaments.
All three agree with brute force on small instances (see `self_test`).
"""
from __future__ import annotations

import itertools
import math
import random
import time
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import LinearConstraint, milp, linprog, Bounds
    from scipy.sparse import coo_matrix
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


Counts = dict[tuple[int, int], tuple[float, float]]   # (i,j) with i<j -> (n_ij, n_ji)


def counts_from_outcomes(pairs, wins) -> Counts:
    """pairs: iterable of (i,j); wins: 1 if i beat j else 0. Returns Counts keyed i<j."""
    c: dict = defaultdict(lambda: [0.0, 0.0])
    for (i, j), w in zip(pairs, wins):
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if i < j:
            c[key][0 if w else 1] += 1
        else:
            c[key][1 if w else 0] += 1
    return {k: (v[0], v[1]) for k, v in c.items()}


def pair_ceiling(counts: Counts) -> float:
    N = sum(a + b for a, b in counts.values())
    loss = sum(min(a, b) for a, b in counts.values())
    return 1 - loss / N if N else float("nan")


def majority_digraph(counts: Counts):
    """Edges (u -> v, weight |n_uv - n_vu|) for pairs with a strict majority."""
    edges = {}
    for (i, j), (a, b) in counts.items():
        if a > b:
            edges[(i, j)] = a - b
        elif b > a:
            edges[(j, i)] = b - a
    return edges


def _nodes(counts: Counts):
    s = set()
    for i, j in counts:
        s.add(i); s.add(j)
    return sorted(s)


# ------------------------------------------------------------------ exact ILP (linear ordering)
def transitive_ceiling_exact(counts: Counts, time_limit: float = 600.0, verbose=False) -> dict:
    """Exact best-ordering accuracy via the linear-ordering ILP:
    x_ij in {0,1} (i<j) means i ranked above j; for i<j<k:  x_ij + x_jk - x_ik in [0,1].
    Objective: maximise sum n_ij x_ij + n_ji (1 - x_ij)."""
    assert HAVE_SCIPY
    nodes = _nodes(counts)
    idx = {v: t for t, v in enumerate(nodes)}
    n = len(nodes)
    var = {}
    for i in range(n):
        for j in range(i + 1, n):
            var[(i, j)] = len(var)
    nv = len(var)
    c = np.zeros(nv)
    const = 0.0
    N = 0.0
    for (u, v), (a, b) in counts.items():
        i, j = idx[u], idx[v]
        if i > j:
            i, j, a, b = j, i, b, a
        # maximise a*x + b*(1-x)  ->  minimise -(a-b) x ; constant b
        c[var[(i, j)]] -= (a - b)
        const += b
        N += a + b
    rows, cols, vals = [], [], []
    r = 0
    for i, j, k in itertools.combinations(range(n), 3):
        for (p, s) in ((var[(i, j)], 1), (var[(j, k)], 1), (var[(i, k)], -1)):
            rows.append(r); cols.append(p); vals.append(s)
        r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    cons = LinearConstraint(A, lb=np.zeros(r), ub=np.ones(r))
    t0 = time.time()
    res = milp(c, constraints=[cons], integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": verbose})
    x = res.x
    best = const - float(c @ np.round(x))
    status = "optimal" if res.status == 0 else f"status{res.status}"
    # recover the order for reference
    score = np.zeros(n)
    for (i, j), t in var.items():
        if x[t] > 0.5:
            score[i] += 1
        else:
            score[j] += 1
    order = [nodes[t] for t in np.argsort(-score)]
    return {"ceiling": best / N, "errors": N - best, "N": N, "status": status,
            "seconds": time.time() - t0, "order": order, "mip_gap": getattr(res, "mip_gap", None)}


def transitive_ceiling_exact_lazy(counts: Counts, time_limit: float = 900.0, verbose=False,
                                  max_rounds: int = 200) -> dict:
    """Exact best-ordering accuracy by the linear-ordering ILP with LAZY triangle
    constraints: solve with the currently active constraints, add every violated
    triangle (x_ij + x_jk - x_ik outside [0,1]), repeat. Terminates with a provably
    optimal linear order (all constraints satisfied, objective optimal for a relaxation).
    Far faster than the full formulation on near-transitive tournaments."""
    assert HAVE_SCIPY
    nodes = _nodes(counts)
    idx = {v: t for t, v in enumerate(nodes)}
    n = len(nodes)
    var = {}
    for i in range(n):
        for j in range(i + 1, n):
            var[(i, j)] = len(var)
    nv = len(var)
    c = np.zeros(nv); const = 0.0; N = 0.0
    for (u, v), (a, b) in counts.items():
        i, j = idx[u], idx[v]
        if i > j:
            i, j, a, b = j, i, b, a
        c[var[(i, j)]] -= (a - b); const += b; N += a + b
    active: set = set()
    t0 = time.time()
    x = (c < 0).astype(float)          # unconstrained optimum: majority per pair
    x[c == 0] = 1.0
    status = "optimal"
    for rnd in range(max_rounds):
        viol = []
        xr = np.round(x)
        for i, j, k in itertools.combinations(range(n), 3):
            s = xr[var[(i, j)]] + xr[var[(j, k)]] - xr[var[(i, k)]]
            if s < -0.5 or s > 1.5:
                viol.append((i, j, k))
        if not viol:
            break
        active.update(viol)
        rows, cols, vals = [], [], []
        for r, (i, j, k) in enumerate(active):
            for (p, s) in ((var[(i, j)], 1), (var[(j, k)], 1), (var[(i, k)], -1)):
                rows.append(r); cols.append(p); vals.append(s)
        A = coo_matrix((vals, (rows, cols)), shape=(len(active), nv)).tocsr()
        res = milp(c, constraints=[LinearConstraint(A, lb=np.zeros(len(active)), ub=np.ones(len(active)))],
                   integrality=np.ones(nv), bounds=Bounds(0, 1),
                   options={"time_limit": max(5.0, time_limit - (time.time() - t0))})
        if res.x is None:
            status = f"status{res.status}"; break
        x = res.x
        if verbose:
            print(f"  lazy round {rnd}: {len(active)} triangle constraints, obj={const - float(c @ np.round(x)):.1f}")
        if time.time() - t0 > time_limit:
            status = "time_limit"; break
    best = const - float(c @ np.round(x))          # relaxation value: exact if optimal, else an upper bound
    score = np.zeros(n)
    for (i, j), t in var.items():
        if x[t] > 0.5: score[i] += 1
        else: score[j] += 1
    order = [nodes[t] for t in np.argsort(-score)]
    # an achievable ordering (lower bound on the ceiling): local search from the incumbent order
    order_ls = _local_search(counts, order, seed=0)
    pos = {v: t for t, v in enumerate(order_ls)}
    achieved = N - _order_errors(counts, pos)
    out = {"upper": best / N, "lower": achieved / N, "N": N, "status": status,
           "seconds": time.time() - t0, "order": order_ls, "active_constraints": len(active)}
    out["ceiling"] = best / N if status == "optimal" else None
    out["errors"] = N - best
    return out


# ------------------------------------------------------------------ bounds for sparse graphs
def _order_errors(counts: Counts, pos: dict) -> float:
    err = 0.0
    for (i, j), (a, b) in counts.items():
        err += b if pos[i] < pos[j] else a
    return err


def _local_search(counts: Counts, order: list, max_rounds: int = 30, seed: int = 0) -> list:
    """Insertion local search (Kemeny/FAS heuristic): move single items to their best
    position; repeat until no improvement. O(rounds * n * deg log deg)."""
    rng = random.Random(seed)
    adj = defaultdict(dict)
    for (i, j), (a, b) in counts.items():
        adj[i][j] = (a, b); adj[j][i] = (b, a)   # adj[u][v] = (n_uv, n_vu)
    order = list(order)
    improved = True; rounds = 0
    while improved and rounds < max_rounds:
        improved = False; rounds += 1
        items = list(order); rng.shuffle(items)
        for v in items:
            cur = order.index(v)
            rest = order[:cur] + order[cur + 1:]
            pos = {u: t for t, u in enumerate(rest)}
            # cost(slot t) = errors from pairs (v,u) if v is inserted before rest[t]
            # slot 0: v above every neighbour -> error n_uv for each neighbour u
            base = sum(nuv for u, (nvu, nuv) in adj[v].items())
            deltas = [0.0] * (len(rest) + 1)
            for u, (nvu, nuv) in adj[v].items():
                # once v is placed below u (slot > pos[u]), error becomes n_vu instead of n_uv
                deltas[pos[u] + 1] += nvu - nuv
            best_t, best_c, run = 0, base, base
            cur_c = None
            for t in range(len(rest) + 1):
                run += deltas[t]
                if t == cur:
                    cur_c = run
                if run < best_c - 1e-9:
                    best_c, best_t = run, t
            if cur_c is None:
                cur_c = base
            if best_c < cur_c - 1e-9:
                order = rest[:best_t] + [v] + rest[best_t:]
                improved = True
    return order


def transitive_ceiling_bounds(counts: Counts, init_score: dict | None = None, seed: int = 0,
                              max_lp_rounds: int = 200, exact_ilp: bool = True,
                              time_limit: float = 900.0, verbose: bool = False) -> dict:
    """Bounds on the best-ordering accuracy for a (sparse) comparison graph.
    lower: an explicit ordering (initialised from `init_score`, e.g. a fitted BT utility,
           then improved by local search);
    upper: LP relaxation of  min sum_e w_e y_e  s.t. every directed cycle has sum y >= 1
           (cycle constraints added lazily); optionally tightened to the exact ILP value
           when the lazy ILP converges within the time limit.
    Accuracy = 1 - (pair_loss + FAS_w) / N."""
    assert HAVE_SCIPY
    nodes = _nodes(counts)
    N = sum(a + b for a, b in counts.values())
    pair_loss = sum(min(a, b) for a, b in counts.values())
    edges = majority_digraph(counts)                    # u->v : weight
    elist = list(edges.items())
    eidx = {e: t for t, (e, _) in enumerate(elist)}
    w = np.array([wt for _, wt in elist], dtype=float)
    out_adj = defaultdict(list)
    for (u, v), _ in elist:
        out_adj[u].append(v)
    # ---- lower bound (achievable ordering)
    if init_score is None:
        init_score = {v: 0.0 for v in nodes}
        for (u, v), wt in elist:
            init_score[u] += wt; init_score[v] -= wt
    order = sorted(nodes, key=lambda v: -init_score.get(v, 0.0))
    order = _local_search(counts, order, seed=seed)
    pos = {v: t for t, v in enumerate(order)}
    achieved_err = _order_errors(counts, pos)
    lower = 1 - achieved_err / N
    # ---- upper bound: LP with lazy cycle constraints
    t0 = time.time()
    rows: list[list[int]] = []

    def find_violated(y):
        """For each edge (u,v): shortest y-path v ~> u; if dist + y_uv < 1 - eps, cycle."""
        found = []
        import heapq
        # Dijkstra from every v is O(E * (E log V)) worst case; restrict to edges with y<1
        cand = [t for t in range(len(elist)) if y[t] < 1 - 1e-6]
        # group by source v to reuse Dijkstra
        by_src = defaultdict(list)
        for t in cand:
            (u, v) = elist[t][0]
            by_src[v].append(t)
        for v, ts in by_src.items():
            dist = {v: 0.0}; prev = {}
            pq = [(0.0, v)]
            while pq:
                d_, x = heapq.heappop(pq)
                if d_ > dist.get(x, float("inf")):
                    continue
                for z in out_adj[x]:
                    nd = d_ + y[eidx[(x, z)]]
                    if nd < dist.get(z, float("inf")) - 1e-12:
                        dist[z] = nd; prev[z] = x
                        heapq.heappush(pq, (nd, z))
            for t in ts:
                (u, _) = elist[t][0]
                if u in dist and dist[u] + y[t] < 1 - 1e-6:
                    # reconstruct cycle v -> ... -> u -> v
                    cyc = [t]
                    x = u
                    while x != v:
                        p = prev[x]; cyc.append(eidx[(p, x)]); x = p
                    found.append(sorted(set(cyc)))
                    if len(found) >= 2000:
                        return found
        return found

    y = np.zeros(len(elist))
    lp_val = 0.0
    for rnd in range(max_lp_rounds):
        viol = find_violated(y)
        if not viol:
            break
        rows.extend(viol)
        r_, c_, v_ = [], [], []
        for ri, cyc in enumerate(rows):
            for t in cyc:
                r_.append(ri); c_.append(t); v_.append(1.0)
        A = coo_matrix((v_, (r_, c_)), shape=(len(rows), len(elist))).tocsr()
        res = linprog(w, A_ub=-A, b_ub=-np.ones(len(rows)), bounds=(0, 1), method="highs")
        y = res.x; lp_val = float(res.fun)
        if verbose:
            print(f"  LP round {rnd}: {len(rows)} cycle constraints, LP={lp_val:.2f}")
        if time.time() - t0 > time_limit:
            break
    upper = 1 - (pair_loss + lp_val) / N
    result = {"lower": lower, "upper": upper, "pair_ceiling": 1 - pair_loss / N,
              "lp_fas": lp_val, "achieved_fas": achieved_err - pair_loss, "N": N,
              "n_cycle_constraints": len(rows), "seconds_lp": time.time() - t0,
              "exact": None}
    # ---- optional exact ILP on the accumulated cycle constraints (lazy B&B)
    if exact_ilp and rows:
        t1 = time.time()
        yb = y
        for rnd in range(max_lp_rounds):
            r_, c_, v_ = [], [], []
            for ri, cyc in enumerate(rows):
                for t in cyc:
                    r_.append(ri); c_.append(t); v_.append(1.0)
            A = coo_matrix((v_, (r_, c_)), shape=(len(rows), len(elist))).tocsr()
            res = milp(w, constraints=[LinearConstraint(A, lb=np.ones(len(rows)), ub=np.inf)],
                       integrality=np.ones(len(elist)), bounds=Bounds(0, 1),
                       options={"time_limit": max(10.0, time_limit - (time.time() - t1))})
            if res.x is None:
                break
            yb = np.round(res.x)
            viol = find_violated(yb)
            if not viol:
                fas = float(w @ yb)
                result["exact"] = 1 - (pair_loss + fas) / N
                result["exact_fas"] = fas
                # the ILP solution is a feasible FAS -> also a valid achievable ordering
                result["lower"] = max(result["lower"], result["exact"])
                result["upper"] = min(result["upper"], result["exact"])
                break
            rows.extend(viol)
            if time.time() - t1 > time_limit:
                break
        result["seconds_ilp"] = time.time() - t1
    return result


# ------------------------------------------------------------------ closed form
def triangle_bound(counts: Counts) -> dict:
    """For a complete deterministic tournament on n items: cyclic-triple fraction f3 and the
    bound 1 - f3/3 on the accuracy of any linear order (Theorem, corollary)."""
    nodes = _nodes(counts)
    n = len(nodes)
    edges = {}
    for (i, j), (a, b) in counts.items():
        edges[(i, j)] = 1 if a > b else (-1 if b > a else 0)
    win = defaultdict(set)
    for (i, j), s in edges.items():
        if s > 0: win[i].add(j)
        elif s < 0: win[j].add(i)
    cyc = 0
    for i, j, k in itertools.combinations(nodes, 3):
        s = [(i in win[j]) or (j in win[i]), (j in win[k]) or (k in win[j]), (i in win[k]) or (k in win[i])]
        if not all(s):
            continue
        # cyclic iff each node has exactly one win within the triple
        if all(len(win[x] & {i, j, k}) == 1 for x in (i, j, k)):
            cyc += 1
    f3 = cyc / math.comb(n, 3) if n >= 3 else 0.0
    return {"f3": f3, "cyclic_triples": cyc, "bound": 1 - f3 / 3, "n": n}


# ------------------------------------------------------------------ self test
def _brute(counts: Counts) -> float:
    nodes = _nodes(counts)
    N = sum(a + b for a, b in counts.values())
    best = -1
    for perm in itertools.permutations(nodes):
        pos = {v: t for t, v in enumerate(perm)}
        best = max(best, N - _order_errors(counts, pos))
    return best / N


def self_test(seed: int = 0):
    rng = random.Random(seed)
    for trial in range(30):
        n = rng.randint(3, 7)
        counts = {}
        for i, j in itertools.combinations(range(n), 2):
            if rng.random() < 0.85:
                a = rng.randint(0, 3); b = rng.randint(0, 3)
                if a + b == 0:
                    a = 1
                counts[(i, j)] = (float(a), float(b))
        if not counts or len(_nodes(counts)) < 3:
            continue
        bf = _brute(counts)
        ex = transitive_ceiling_exact(counts)["ceiling"]
        bd = transitive_ceiling_bounds(counts, seed=seed)
        assert abs(bf - ex) < 1e-9, (trial, bf, ex)
        assert bd["lower"] <= bf + 1e-9 and bd["upper"] >= bf - 1e-9, (trial, bf, bd)
        if bd["exact"] is not None:
            assert abs(bd["exact"] - bf) < 1e-9, (trial, bf, bd)
        assert pair_ceiling(counts) >= bf - 1e-9
    print("ceiling self-test passed (30 random instances vs brute force)")


if __name__ == "__main__":
    self_test()
