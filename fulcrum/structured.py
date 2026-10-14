"""Minimal STRUCTURED fulcrum layer (Phase 4).

This turns the abstract calculus into an ASPIC+-style structured framework so
the Caminada-Amgoud rationality postulates become stateable and testable, and
so `theory/postulates_structured.md` (Theorem B) can be *empirically*
discharged in the safe regime it characterizes:

  (a) certain (certain) indicants are encoded as STRICT-AND-FIRM arguments and
      are maximal in the ordering  (discharges ASPIC+ Def. 18-1);
  (b) preponderance is computed from the defeasible/evidential core
      (strict continuations inherit it)  (Def. 18-1-iii);
  (c) the rule base has NO multi-antecedent (combining) strict rules
      (in particular no conjunction-introduction) -> Def. 18-2 holds vacuously,
      so an *intransitive* preponderance still yields postulate-satisfying
      extensions. This restriction is enforced by `Theory.check_safe_regime`.

Deliberately minimal (the project design notes: do not reimplement a full ASPIC+/solver).
It is faithful enough to exhibit the witness structurally and to check the
postulates; extension enumeration reuses `fulcrum.semantics`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from fulcrum.preference import Preferred, ReorderPolicy, static_policy, weigh_compare
from fulcrum.structures import HEAD_ORDER, Argument, PreferenceHead


# --------------------------------------------------------------------------- #
# Language: propositional literals with negation as contrariness.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Literal:
    atom: str
    positive: bool = True

    @property
    def negation(self) -> Literal:
        return Literal(self.atom, not self.positive)

    def __str__(self) -> str:
        return ("" if self.positive else "~") + self.atom


def neg_contrary(a: Literal, b: Literal) -> bool:
    """Base symmetric contrariness: p and ~p."""
    return a.atom == b.atom and a.positive != b.positive


# --------------------------------------------------------------------------- #
# Rules and premises.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    name: str
    antecedents: frozenset[Literal]
    consequent: Literal
    defeasible: bool
    # head-score vector carried by a DEFEASIBLE rule (its evidential weight);
    # ignored for strict rules.
    head_scores: tuple[float, ...] = ()  # in HEAD_ORDER; () means all zero


@dataclass
class Theory:
    """A structured argumentation theory.

    axioms:    Kn -- certain (certain) premises -> strict-and-firm.
    ordinary:  Kp -- fallible (probable) premises, each with a head-score vector.
    strict/defeasible: rules.
    """

    axioms: set[Literal] = field(default_factory=set)
    ordinary: dict[Literal, tuple[float, ...]] = field(default_factory=dict)
    strict: list[Rule] = field(default_factory=list)
    defeasible: list[Rule] = field(default_factory=list)
    # extra (non-negation) contrary pairs, e.g. three mutually exclusive rulings.
    contraries: set[frozenset[Literal]] = field(default_factory=set)

    def con(self, a: Literal, b: Literal) -> bool:
        """General contrariness: negation OR a declared contrary pair."""
        return neg_contrary(a, b) or frozenset({a, b}) in self.contraries

    def check_safe_regime(self) -> None:
        """Enforce the load-bearing restriction from postulates_structured.md:
        no multi-antecedent (combining) strict rules. Raising here makes the
        assumption explicit rather than silent (Def. 18-2 vacuity depends on it).
        """
        bad = [r.name for r in self.strict if len(r.antecedents) > 1]
        if bad:
            raise ValueError(
                "multi-antecedent strict rules break Def. 18-2 vacuity "
                f"(indirect consistency at risk under intransitive preference): {bad}. "
                "See theory/postulates_structured.md."
            )


# --------------------------------------------------------------------------- #
# Arguments (built by forward chaining).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SArg:
    """A structured argument."""

    aid: str
    conc: Literal
    subs: frozenset[SArg]
    top_rule: Rule | None                # None => premise argument
    ordinary_prems: frozenset[Literal]   # Kp premises used anywhere in the arg
    defrules: frozenset[str]             # names of defeasible rules used
    head_scores: tuple[float, ...]       # evidential vector (last-link core)

    @property
    def is_firm(self) -> bool:
        return not self.ordinary_prems  # only axiom premises

    @property
    def is_strict(self) -> bool:
        return not self.defrules  # no defeasible rules

    @property
    def is_strict_and_firm(self) -> bool:
        return self.is_strict and self.is_firm

    def __hash__(self) -> int:
        return hash(self.aid)


def _zero_vec() -> tuple[float, ...]:
    return tuple(0.0 for _ in HEAD_ORDER)


def build_arguments(theory: Theory, max_depth: int = 4) -> list[SArg]:
    """Construct arguments by bounded forward chaining. Minimal: one argument
    per derivable (literal, provenance) up to `max_depth` rule applications.
    """
    theory.check_safe_regime()
    args: list[SArg] = []
    seen: set[tuple] = set()

    def add(a: SArg) -> None:
        key = (a.conc, a.top_rule.name if a.top_rule else None,
               frozenset(s.aid for s in a.subs))
        if key not in seen:
            seen.add(key)
            args.append(a)

    # Premise arguments.
    i = 0
    for lit in sorted(theory.axioms, key=str):
        add(SArg(f"A{i}", lit, frozenset(), None, frozenset(), frozenset(), _zero_vec()))
        i += 1
    for lit, vec in sorted(theory.ordinary.items(), key=lambda kv: str(kv[0])):
        v = tuple(vec) if vec else _zero_vec()
        add(SArg(f"A{i}", lit, frozenset(), None, frozenset({lit}), frozenset(), v))
        i += 1

    # Rule applications (fixpoint up to max_depth).
    all_rules = [(r, False) for r in theory.strict] + [(r, True) for r in theory.defeasible]
    for _ in range(max_depth):
        by_conc: dict[Literal, list[SArg]] = {}
        for a in args:
            by_conc.setdefault(a.conc, []).append(a)
        for rule, is_def in all_rules:
            # pick one sub-argument per antecedent (arity <=1 for strict in safe regime)
            ants = list(rule.antecedents)
            choices = [by_conc.get(x, []) for x in ants]
            if any(len(c) == 0 for c in choices):
                continue
            # cartesian product (small); antecedent count is 1 for strict here
            from itertools import product
            for combo in product(*choices):
                subs = frozenset(combo)
                op = frozenset().union(*[s.ordinary_prems for s in combo]) if combo else frozenset()
                dr = frozenset().union(*[s.defrules for s in combo]) if combo else frozenset()
                if is_def:
                    dr = dr | {rule.name}
                    core = tuple(rule.head_scores) if rule.head_scores else _zero_vec()
                else:
                    # strict continuation: inherit the (single) sub's evidential core
                    core = combo[0].head_scores if combo else _zero_vec()
                aid = f"A{len(args) + len(seen)}_{rule.name}"
                add(SArg(aid, rule.consequent, subs, rule, op, dr, core))
    return args


# --------------------------------------------------------------------------- #
# Attacks and the (preference-modified) defeat relation.
# --------------------------------------------------------------------------- #
def _as_abstract(a: SArg) -> Argument:
    """Wrap a structured argument's evidential core as an abstract Argument so
    we can reuse weigh_compare / any ReorderPolicy on head_scores."""
    hs: dict[PreferenceHead, float] = {
        h: (a.head_scores[i] if i < len(a.head_scores) else 0.0)
        for i, h in enumerate(HEAD_ORDER)
    }
    return Argument(a.aid, str(a.conc), head_scores=hs)


def structured_prefer(a: SArg, b: SArg, reorder: ReorderPolicy = static_policy) -> Preferred:
    """Argument ordering with certain-as-strict-firm maximality (Def. 18-1),
    else fulcrum preponderance on the evidential cores.
    Returns which of a,b is *preferred* (LEFT=a, RIGHT=b, INCOMPARABLE=equipollent).
    """
    # Def. 18-1: strict-and-firm (certain) is strictly preferred to any fallible arg.
    if a.is_strict_and_firm and not b.is_strict_and_firm:
        return Preferred.LEFT
    if b.is_strict_and_firm and not a.is_strict_and_firm:
        return Preferred.RIGHT
    if a.is_strict_and_firm and b.is_strict_and_firm:
        return Preferred.INCOMPARABLE  # both infallible, no weighing
    # both fallible: weigh by fulcrum preponderance on evidential cores.
    return weigh_compare(_as_abstract(a), _as_abstract(b), reorder=reorder)


@dataclass
class StructuredAF:
    args: list[SArg]
    theory: Theory
    reorder: ReorderPolicy = static_policy

    def _attacked_subarg(self, a: SArg, b: SArg) -> SArg | None:
        """Return the sub-argument (or premise) of b that a attacks, or None.
        Restricted rebut (defeasible top rule) + undermine (ordinary premise)."""
        for bp in sorted(b.subs | {b}, key=lambda x: x.aid):
            if bp.top_rule is not None and bp.top_rule.defeasible \
                    and self.theory.con(a.conc, bp.conc):
                return bp
            if bp.top_rule is None and any(self.theory.con(a.conc, p)
                                           for p in bp.ordinary_prems):
                return bp
        return None

    def defeats(self) -> list[tuple[str, str]]:
        """ASPIC+-style defeat: a defeats b iff a attacks b on b', and a is not
        strictly less preferred than the attacked point b'. Undercut (none here)
        would be preference-independent."""
        out = []
        for a in self.args:
            for b in self.args:
                if a is b:
                    continue
                bp = self._attacked_subarg(a, b)
                if bp is None:
                    continue
                pref = structured_prefer(a, bp, self.reorder)
                # attack succeeds unless a is STRICTLY less preferred than b'
                if pref is Preferred.RIGHT:  # b' preferred to a => repelled
                    continue
                out.append((a.aid, b.aid))
        return out

    def to_digraph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(a.aid for a in self.args)
        g.add_edges_from(self.defeats())
        return g


# --------------------------------------------------------------------------- #
# Rationality-postulate checks (Caminada-Amgoud) for an extension.
# --------------------------------------------------------------------------- #
def strict_closure(lits: set[Literal], theory: Theory, max_depth: int = 4) -> set[Literal]:
    """Close a set of literals under the (unary/chaining) strict rules."""
    cur = set(lits)
    for _ in range(max_depth):
        nxt = set(cur)
        for r in theory.strict:
            if r.antecedents <= cur:
                nxt.add(r.consequent)
        if nxt == cur:
            break
        cur = nxt
    return cur


def is_consistent(lits: set[Literal], theory: Theory) -> bool:
    return not any(theory.con(x, y) for x in lits for y in lits if x != y)


def conclusions(ext_ids: set[str], args: list[SArg]) -> set[Literal]:
    idx = {a.aid: a for a in args}
    return {idx[i].conc for i in ext_ids}


def check_postulates(ext_ids: set[str], args: list[SArg], theory: Theory) -> dict[str, bool]:
    """Return direct/indirect consistency + closure flags for one extension."""
    concs = conclusions(ext_ids, args)
    direct = is_consistent(concs, theory)
    closure = strict_closure(concs, theory)
    indirect = is_consistent(closure, theory)
    # closure postulate: every strict consequence of the extension's conclusions
    # is itself the conclusion of some argument in the extension.
    idx_concs = concs
    closed = all(c in idx_concs for c in strict_closure(idx_concs, theory)
                 if c in {r.consequent for r in theory.strict})
    return {"direct_consistency": direct,
            "indirect_consistency": indirect,
            "closure_under_strict": closed}
