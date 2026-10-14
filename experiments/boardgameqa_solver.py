"""Symbolic solver for BoardGameQA (real defeasible-reasoning benchmark).

Parses the dataset's structured `theory` field (facts / defeasible rules with
variables + existentials / rule superiority) and computes proved/disproved/unknown
by Defeasible Logic with team defeat (Nute; Maher) — the semantics BoardGameQA's
labels are generated under. Deterministic; no LLM formalization step, so no
extraction noise. This is the symbolic pipeline; on this benchmark fulcrum and
ASPIC+ coincide (conflicts carry a single explicit rule priority), so it stands in
for both — the honest contrast is symbolic vs LLM (see RUNBOOK.md §5).

Usage: python experiments/boardgameqa_solver.py --data data/BoardgameQA/BoardgameQA-Main-depth1/test.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass

# A literal: (subject, verb, object, negated). Variables are single uppercase tokens.
Lit = tuple[str, str, str, bool]


def _parse_lit(s: str) -> tuple[Lit, bool]:
    """Return (literal, existential). Handles '~(a, b, c)' and 'exists X (a, b, c)'."""
    s = s.strip()
    existential = False
    if s.startswith("exists "):
        existential = True
        s = s.split(None, 2)[2].strip()      # drop 'exists X '
    neg = s.startswith("~")
    if neg:
        s = s[1:].strip()
    s = s.strip("()")
    parts = [p.strip() for p in s.split(",", 2)]
    while len(parts) < 3:
        parts.append("")
    return (parts[0], parts[1], parts[2], neg), existential


@dataclass
class Rule:
    name: str
    antecedents: list[tuple[Lit, bool]]      # (literal, existential)
    consequent: Lit


@dataclass
class Theory:
    facts: set[Lit]
    rules: list[Rule]
    superior: set[tuple[str, str]]           # (r, s) means rule r > rule s
    goal: Lit


def parse_theory(theory_text: str, goal_text: str) -> Theory:
    facts: set[Lit] = set()
    rules: list[Rule] = []
    superior: set[tuple[str, str]] = set()
    section = None
    for raw in theory_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.rstrip(":") in ("Facts", "Rules", "Preferences"):
            section = line.rstrip(":")
            continue
        if section == "Facts":
            lit, _ = _parse_lit(line)
            facts.add(lit)
        elif section == "Rules":
            m = re.match(r"(Rule\d+):\s*(.*)", line)
            if not m:
                continue
            name, body = m.group(1), m.group(2)
            ant_str, cons_str = body.split("=>")
            ants = [_parse_lit(a) for a in ant_str.split("^") if a.strip()]
            cons, _ = _parse_lit(cons_str)
            rules.append(Rule(name, ants, cons))
        elif section == "Preferences":
            m = re.match(r"(Rule\d+)\s*>\s*(Rule\d+)", line)
            if m:
                superior.add((m.group(1), m.group(2)))
    goal, _ = _parse_lit(goal_text)
    return Theory(facts, rules, superior, goal)


def _is_var(tok: str) -> bool:
    return len(tok) == 1 and tok.isupper()


def _unify(pat: Lit, fact: Lit, subst: dict) -> dict | None:
    """Unify a (possibly variable-carrying) pattern with a ground fact literal."""
    if pat[3] != fact[3]:
        return None
    out = dict(subst)
    for i in range(3):
        p = pat[i]
        if _is_var(p):
            if p in out and out[p] != fact[i]:
                return None
            out[p] = fact[i]
        elif p != fact[i]:
            return None
    return out


def _apply(pat: Lit, subst: dict) -> Lit:
    return (subst.get(pat[0], pat[0]), subst.get(pat[1], pat[1]),
            subst.get(pat[2], pat[2]), pat[3])


def _rule_instances(rule: Rule, accepted: set[Lit]) -> list[Lit]:
    """All ground consequents obtainable from this rule given the accepted set.
    Non-existential antecedents bind variables; existential ones only need to
    match *some* accepted literal (their variables stay local)."""
    substs = [{}]
    for lit, existential in rule.antecedents:
        nxt = []
        for s in substs:
            matched = False
            for f in accepted:
                u = _unify(lit, f, s if not existential else dict(s))
                if u is not None:
                    matched = True
                    nxt.append(s if existential else u)  # existential: don't bind outward
                    if existential:
                        break
            if not matched:
                pass
        substs = nxt
        if not substs:
            return []
    return [_apply(rule.consequent, s) for s in substs]


def _neg(lit: Lit) -> Lit:
    return (lit[0], lit[1], lit[2], not lit[3])


def solve(theory: Theory, max_iter: int = 50) -> Lit | None:
    """Defeasible-Logic team-defeat fixpoint. Returns the set of accepted literals'
    membership for the goal via the label function below (this returns the accepted
    set indirectly through repeated derivation)."""
    accepted: set[Lit] = set(theory.facts)
    for _ in range(max_iter):
        # collect, per literal, the set of rules currently supporting it
        support: dict[Lit, set[str]] = {}
        for rule in theory.rules:
            for cons in _rule_instances(rule, accepted):
                support.setdefault(cons, set()).add(rule.name)
        newly = set()
        for lit, rules_for in support.items():
            if lit in accepted:
                continue
            rules_against = support.get(_neg(lit), set())
            if not rules_against:
                newly.add(lit)                       # unopposed -> accept
                continue
            # team defeat: lit wins if every opposing rule s is beaten by some rule r for lit with r>s
            if all(any((r, s) in theory.superior for r in rules_for) for s in rules_against):
                newly.add(lit)
        newly -= accepted
        if not newly:
            break
        accepted |= newly
    return None if theory.goal not in accepted and _neg(theory.goal) not in accepted else accepted


def predict(theory: Theory) -> str:
    accepted = solve(theory)
    if accepted is None:
        # neither goal nor its negation derivable
        return "unknown"
    if theory.goal in accepted:
        return "proved"
    if _neg(theory.goal) in accepted:
        return "disproved"
    return "unknown"


def run(path: str, limit: int | None = None) -> dict:
    data = json.load(open(path))[: limit or None]
    preds, golds = [], []
    for ex in data:
        th = parse_theory(ex["theory"], ex["goal"])
        preds.append(predict(th))
        golds.append(ex["label"].strip().lower())
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    labels = ["proved", "disproved", "unknown"]
    f1s = []
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in zip(preds, golds))
        fp = sum(p == lab and g != lab for p, g in zip(preds, golds))
        fn = sum(p != lab and g == lab for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"acc": acc, "macro_f1": sum(f1s) / len(f1s), "n": len(golds),
            "pred_dist": dict(Counter(preds)), "gold_dist": dict(Counter(golds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(f"symbolic (defeasible-logic) solver on {args.data}")
    print(run(args.data, args.limit))


if __name__ == "__main__":
    main()
