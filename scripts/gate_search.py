"""Machine-assisted search for a separating example (the project design notes Phase 2).

Enumerates fixed head-orderings and checks whether a candidate
context-conditioned policy produces verdicts on an instance *family* that NO
single fixed ordering reproduces. A verified separating family on finite
cases is strong (not conclusive) evidence for outcome (A). Wide search
without one is evidence for (B).

This is a research tool, not part of the library. It intentionally makes the
separation question concrete and falsifiable. Extend the instance generator
and the candidate policy; log findings in theory/self_refutation_log.md.

Usage:
    python scripts/gate_search.py --trials 2000 --pairs 3
"""

from __future__ import annotations

import argparse
import itertools
import random

from fulcrum.baselines.single_ordering import LexicographicCollapse
from fulcrum.preference import Preferred, weigh_compare
from fulcrum.structures import HEAD_ORDER, Argument, PreferenceHead


def candidate_policy(a: Argument, b: Argument, base):
    """Same candidate as the gate test: promote SPECIFICITY when 'text_critical' is
    present. Replace/extend this to explore other context rules."""
    if "text_critical" in a.context or "text_critical" in b.context:
        return (PreferenceHead.SPECIFICITY, PreferenceHead.SOURCE,
                PreferenceHead.MODAL, PreferenceHead.CORROBORATION)
    return base


def random_arg(aid: str, rng: random.Random, tagged: bool) -> Argument:
    scores = {h: float(rng.randint(0, 5)) for h in PreferenceHead}
    ctx = frozenset({"text_critical"}) if tagged else frozenset()
    return Argument(aid=aid, claim=aid, head_scores=scores, context=ctx)


def verdicts_context(sample) -> list[Preferred]:
    return [weigh_compare(a, b, reorder=candidate_policy) for a, b in sample]


def matched_by_any_fixed(sample, target) -> bool:
    for perm in itertools.permutations(HEAD_ORDER):
        bl = LexicographicCollapse(ordering=perm)
        if [bl.prefer(a, b) for a, b in sample] == target:
            return True
    return False


def search(trials: int, pairs: int, seed: int) -> None:
    rng = random.Random(seed)
    found = 0
    for t in range(trials):
        sample = []
        for i in range(pairs):
            tagged = rng.random() < 0.5
            a = random_arg(f"a{t}_{i}", rng, tagged)
            b = random_arg(f"b{t}_{i}", rng, tagged)
            sample.append((a, b))
        target = verdicts_context(sample)
        if not matched_by_any_fixed(sample, target):
            found += 1
            print(f"[trial {t}] SEPARATING FAMILY FOUND (no fixed LEXICOGRAPHIC ordering matches):")
            for (a, b), v in zip(sample, target, strict=True):
                tag = "T" if "text_critical" in a.context else "-"
                print(f"    [{tag}] {a.vector()} vs {b.vector()} -> {v.value}")
            print("    >>> THIS IS ONLY NECESSARY, NOT SUFFICIENT, FOR SEPARATION.")
            print("    >>> It defeats fixed *lexicographic* orderings only. Full ASPIC+/ABA+")
            print("    >>> can encode context by using DIFFERENT ARGUMENTS PER CONTEXT; this")
            print("    >>> search does NOT rule that out. Run the self-refutation protocol")
            print("    >>> (theory/separation_theorem.md) against those frameworks before")
            print("    >>> claiming outcome (A). Log the attempt in self_refutation_log.md.\n")
            if found >= 5:
                break
    if found == 0:
        print(f"No separating family found in {trials} trials (pairs={pairs}, seed={seed}).")
        print(">>> Evidence toward outcome (B): the candidate policy may be "
              "expressible by fixed orderings. Try richer policies/instances, "
              "then consider proving expressibility.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    search(args.trials, args.pairs, args.seed)


if __name__ == "__main__":
    main()
