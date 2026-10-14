"""Real defeasible-reasoning benchmark loaders (Phase 6 — the primary external-
validity gap). SCAFFOLD: written to run on an unrestricted machine; NOT executed
in the offline dev environment, so field names are the documented BoardGameQA
schema and should be spot-checked against the downloaded data (see RUNBOOK.md).

BoardGameQA (Kazemi et al., NeurIPS 2023, "BoardgameQA: A Dataset for Natural
Language Reasoning with Contradictory Information") — multi-step defeasible
reasoning where conflicts between rules are resolved by explicit preferences;
labels are {proved, disproved, unknown}. LLM-ASPIC+ reports strong results here,
which is why it is the pre-registered external comparison.

Download (on a networked box):
    git clone https://github.com/google/BoardgameQA      # or the HF mirror
Each split is JSON: a list of records with (roughly) the fields below. If the
mirror you use differs, pass field names to `load_boardgameqa`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DefeasibleExample:
    context: str          # facts + rules + preferences (natural language)
    question: str
    label: str            # normalized to "proved" | "disproved" | "unknown"
    proof: str = ""       # gold proof/explanation, if present
    meta: dict | None = None


_LABEL_MAP = {
    "proved": "proved", "true": "proved", "yes": "proved", "entailment": "proved",
    "disproved": "disproved", "false": "disproved", "no": "disproved",
    "unknown": "unknown", "neutral": "unknown", "cannot": "unknown",
}


def _norm_label(raw: str) -> str:
    return _LABEL_MAP.get(str(raw).strip().lower(), "unknown")


def load_boardgameqa(path: str | Path, *, context_field: str = "example",
                     question_field: str = "question", label_field: str = "label",
                     proof_field: str = "proof", limit: int | None = None
                     ) -> list[DefeasibleExample]:
    """Load one BoardGameQA split (a .json list or .jsonl). Adjust field names if
    the mirror differs (RUNBOOK.md shows how to inspect the schema first)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        records = json.loads(p.read_text())
        if isinstance(records, dict):                       # some mirrors wrap in {"data": [...]}
            records = records.get("data") or next(iter(records.values()))
    out = []
    for r in records[: limit or len(records)]:
        out.append(DefeasibleExample(
            context=str(r.get(context_field, "")),
            question=str(r.get(question_field, "")),
            label=_norm_label(r.get(label_field, "unknown")),
            proof=str(r.get(proof_field, "")),
            meta={k: v for k, v in r.items()
                  if k not in {context_field, question_field, label_field, proof_field}},
        ))
    if not out:
        raise ValueError(f"no records parsed from {path}; check field names (see RUNBOOK.md)")
    return out


# A second real domain (legal/medical) can be added with the same interface.
def load_jsonl_defeasible(path: str | Path, **fields) -> list[DefeasibleExample]:
    """Generic loader for any JSONL defeasible-QA set with the same (context,
    question, label[, proof]) shape. Point it at a legal/medical set for the
    second real-data setting the pre-registration asks for."""
    return load_boardgameqa(path, **fields)
