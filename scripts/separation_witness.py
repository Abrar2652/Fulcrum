"""Machine-checked SEPARATION witness for the fulcrum calculus (the project design notes Phase 2).

This script is the executable backbone of theory/separation_theorem.md. It
establishes, by exhaustive finite check, that the context-conditioned fulcrum
preponderance operator produces conflict-resolution behaviour that NO single
preference (pre)order over a FIXED argument set can reproduce -- neither at the
level of the pairwise preponderance relation nor at the level of Dung
acceptance semantics.

The construction closes the standard escape route (gate_search.py's warning):
the reorder policy here is ENDOGENOUS -- a pure function of the two arguments'
head-score vectors, with NO context tags. There is therefore no "context"
variable that a single-ordering framework could push into argument identity by
splitting one argument into several. The separating behaviour lives on three
fixed arguments of fixed identity.

Run:
    python scripts/separation_witness.py
Expected: all checks report SEPARATION (0 single-ordering realizers).
"""
from __future__ import annotations

import itertools

import networkx as nx

from fulcrum.defeat import ArgumentationFramework
from fulcrum.preference import Preferred, weigh_compare
from fulcrum.semantics import grounded_extension, is_admissible, is_conflict_free
from fulcrum.structures import HEAD_ORDER, Argument

HEADS = HEAD_ORDER[:3]  # source_reliability, specificity, modal_force


# --------------------------------------------------------------------------- #
# The endogenous context-conditioned policy: MAX-GAP-FIRST.
# When weighing a against b, apply the heads in order of descending
# |score_h(a) - score_h(b)| (ties broken by the fixed HEAD_ORDER). Under a
# lexicographic scan this means: "the head of greatest disparity decides;
# the higher score on that head wins." It depends ONLY on the pair -> endogenous.
# --------------------------------------------------------------------------- #
def max_gap_policy(a: Argument, b: Argument, base):
    heads = list(base)
    heads.sort(key=lambda h: (-abs(a.score(h) - b.score(h)), base.index(h)))
    return tuple(heads)


def _arg(aid: str, vec) -> Argument:
    return Argument(aid, aid, head_scores={h: float(v) for h, v in zip(HEADS, vec)})


# The witness: three arguments whose score vectors are cyclic shifts of (0,1,2).
# The max-gap head rotates modal_force -> specificity -> source_reliability around the triangle, giving a
# strict preponderance 3-cycle a > b > c > a.
A = _arg("a", (0, 1, 2))
B = _arg("b", (1, 2, 0))
C = _arg("c", (2, 0, 1))
ARGS = [A, B, C]
NAMES = ["a", "b", "c"]


def cmp_fulcrum(x: Argument, y: Argument) -> Preferred:
    return weigh_compare(x, y, base_ordering=HEAD_ORDER, reorder=max_gap_policy)


# --------------------------------------------------------------------------- #
# Baseline space: EVERY preorder over {a,b,c}. A preorder's strict part is the
# most general "single ordering" any of Dung / ASPIC+ / ABA+ / VAF reduces to
# (all four take a single preorder as their preference input). We also confirm
# the enumerator by counting: #preorders on 3 labelled elements = 29 (A000798).
# --------------------------------------------------------------------------- #
def all_preorders(n: int):
    elems = range(n)
    off = [(i, j) for i in elems for j in elems if i != j]
    for bits in itertools.product((0, 1), repeat=len(off)):
        rel = {(i, i) for i in elems} | {p for p, b in zip(off, bits) if b}
        transitive = all(
            not ((i, j) in rel and (j, k) in rel and (i, k) not in rel)
            for i in elems for j in elems for k in elems
        )
        if transitive:
            yield rel


def strict_table_from_preorder(rel):
    tbl = {}
    for i, j in itertools.permutations(range(3), 2):
        ge_ij, ge_ji = (i, j) in rel, (j, i) in rel
        tbl[(i, j)] = (
            Preferred.LEFT if ge_ij and not ge_ji
            else Preferred.RIGHT if ge_ji and not ge_ij
            else Preferred.INCOMPARABLE
        )
    return tbl


def fulcrum_strict_table():
    return {
        (i, j): cmp_fulcrum(ARGS[i], ARGS[j])
        for i, j in itertools.permutations(range(3), 2)
    }


def preorder_defeat_graph(rel) -> nx.DiGraph:
    """Defeat graph induced by (preorder rel) on the all-mutual-attacks AF."""
    g = nx.DiGraph()
    g.add_nodes_from(NAMES)
    for i, j in itertools.combinations(range(3), 2):
        x, y = NAMES[i], NAMES[j]
        ge_ij, ge_ji = (i, j) in rel, (j, i) in rel
        if ge_ij and not ge_ji:
            g.add_edge(x, y)
        elif ge_ji and not ge_ij:
            g.add_edge(y, x)
        else:  # indifference/incomparability -> both attacks survive
            g.add_edge(x, y)
            g.add_edge(y, x)
    return g


def preferred_and_stable(g: nx.DiGraph):
    nodes = list(g.nodes())
    subsets = [set(c) for r in range(len(nodes) + 1)
               for c in itertools.combinations(nodes, r)]
    admissible = [s for s in subsets if is_admissible(g, s)]
    preferred = [s for s in admissible if not any(s < t for t in admissible)]
    stable = []
    for s in subsets:
        if not is_conflict_free(g, s):
            continue
        if all(any(g.has_edge(m, o) for m in s) for o in set(nodes) - s):
            stable.append(s)
    return preferred, stable


def _fmt(exts):
    return "{" + ", ".join("".join(sorted(s)) if s else "0" for s in exts) + "}"


def main() -> int:
    print("=" * 70)
    print("WEIGH SEPARATION WITNESS  (endogenous max-gap policy, no context tags)")
    print("=" * 70)
    print(f"  a={A.vector()[:3]}  b={B.vector()[:3]}  c={C.vector()[:3]}  "
          f"over heads {[h.value for h in HEADS]}")
    for x, y, lbl in [(A, B, "a?b"), (B, C, "b?c"), (C, A, "c?a")]:
        order = max_gap_policy(x, y, HEADS)
        gaps = {h.value: int(abs(x.score(h) - y.score(h))) for h in HEADS}
        print(f"  {lbl}: decides on unique max-gap head "
              f"'{order[0].value}' {gaps} -> {cmp_fulcrum(x, y).value}")

    # ---- Level 1: comparator (preponderance relation) ----
    tj = fulcrum_strict_table()
    preorders = list(all_preorders(3))
    assert len(preorders) == 29, f"enumerator bug: {len(preorders)} != 29"
    realizers = [rel for rel in preorders if strict_table_from_preorder(rel) == tj]
    print("\n[Level 1: preponderance relation]")
    print(f"  preorders on 3 elements: {len(preorders)} (== A000798, check OK)")
    print(f"  preorders realizing the fulcrum preponderance table: {len(realizers)}")

    # ---- Level 2: Dung acceptance (extensions) ----
    af = ArgumentationFramework(
        ARGS,
        [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "a"), ("a", "c")],
        reorder=max_gap_policy,
    )
    g_tj = af.to_digraph("defeat")
    tj_edges = set(g_tj.edges())
    pref_tj, stab_tj = preferred_and_stable(g_tj)
    print("\n[Level 2: Dung acceptance on the all-mutual-attacks AF]")
    print(f"  fulcrum defeat graph: {sorted(tj_edges)}  "
          f"(directed 3-cycle: {tj_edges == {('a','b'),('b','c'),('c','a')}})")
    print(f"  fulcrum grounded={_fmt([grounded_extension(g_tj)])}  "
          f"preferred={_fmt(pref_tj)}  stable={_fmt(stab_tj)}")

    same_def = same_pref = 0
    key_tj = ({frozenset(s) for s in pref_tj}, {frozenset(s) for s in stab_tj})
    for rel in preorders:
        g = preorder_defeat_graph(rel)
        if set(g.edges()) == tj_edges:
            same_def += 1
        pref, stab = preferred_and_stable(g)
        if ({frozenset(s) for s in pref}, {frozenset(s) for s in stab}) == key_tj:
            same_pref += 1  # identical (preferred, stable) profile
    print(f"  preorders inducing the SAME defeat graph:               {same_def}")
    print(f"  preorders with the SAME (preferred,stable) profile:     {same_pref}")

    ok = (len(realizers) == 0 and same_def == 0 and same_pref == 0)
    print("\n" + "=" * 70)
    print("RESULT:", "SEPARATION (outcome A) — 0 single-ordering realizers at every level"
          if ok else "NO SEPARATION — a single ordering matched; investigate (outcome B)")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
