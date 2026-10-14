# RUNBOOK — closing the empirical gaps on an unrestricted machine

This box was offline / `pip`-locked / no-GPU-solver, so the internet- and
solver-dependent experiments are shipped as **ready-to-run harnesses**, not
results. Below: what is already run here, what to install, and how to run the rest.
**Nothing in this file is claimed as a result until you run it.**

## Status of each piece

| Piece | File | Status |
|---|---|---|
| Synthetic H1/H3 (accuracy, auditability) | `run.py` | ✅ run here |
| Synthetic H2 (calibration) | `../differentiable/h2.py` | ✅ run here |
| Ablations A1–A4 | `ablations.py` | ✅ run here (A3/A4 were the missing ones) |
| Synthetic LLM probe (unfair-by-design) | `llm_local.py` | ✅ run here |
| **BoardGameQA symbolic solver** | `boardgameqa_solver.py` | ✅ RUN here (data downloaded + solved, 15 splits) |
| **BoardGameQA loader / NL→theory** | `datasets.py`, `boardgameqa_solver.py::parse_theory` | ✅ deterministic (dataset ships a structured `theory` field) |
| **Fair LLM on BoardGameQA** | `llm_runner.py` | ✅ runs on GPU — see RESULTS.md |
| Dung/stable/preferred + PDL via clingo | `../fulcrum/solver.py`, `adapters.py` | ✅ clingo installed; solver tests now run |
| Established ASPIC⁺/ABA⁺ *library* (PyArg) | `../fulcrum/baselines/adapters.py` | 🟡 not on PyPI; clingo/ASPARTIX covers the AF semantics |

## 1. Install (networked box)

```bash
pip install -e ".[dev]"
pip install torch transformers datasets           # LLM baselines
pip install py-arg                                  # ASPIC+/ABA+ (Utrecht)
pip install clingo                                  # ASP: Dung semantics + PDL
```

## 2. Get the data

```bash
git clone https://github.com/google/BoardgameQA data/BoardgameQA
# inspect the schema FIRST, then set field names in datasets.load_boardgameqa if needed:
python -c "import json; d=json.load(open('data/BoardgameQA/BoardgameQA-Main-depth1/test.json')); print(list(d[0].keys())[:8]); print(d[0])"
```

## 3. Run

```bash
PYTHONPATH=. python experiments/ablations.py                     # A1–A4 (already reproducible)
# fair LLM baseline on real NL data:
PYTHONPATH=. python experiments/llm_runner.py \
    --data data/BoardgameQA/BoardgameQA-Main-depth1/test.json \
    --model Qwen/Qwen2.5-7B-Instruct --limit 1000
# established symbolic baselines: fill the two `# VERIFY` sites in adapters.py
# against your installed py-arg version, then wire into a benchmark runner.
```

## 4. The one hard TODO: NL → structured theory

To compare **fulcrum vs ASPIC⁺ (not just vs an LLM)** on BoardGameQA, each NL
problem must be formalized into a structured theory (facts, defeasible rules,
preferences), then solved. This is the LLM-ASPIC⁺ recipe (LLM extracts the theory;
a solver computes the answer). Two honest options:
- reuse a released LLM-ASPIC⁺ formalizer if available, feeding its theory to both
  `fulcrum.structured` and `adapters.PyArgAspicSolver`; or
- write a parser for BoardGameQA's (fairly regular) rule syntax.
Either way, keep the formalizer identical across systems so the comparison is fair.

## 5. Honesty notes that must reach the paper

- **BoardGameQA resolves conflicts by single rule-priorities, not multi-criteria
  context-dependent preference.** So it demonstrates the *framework* works on real
  data and enables a *fair LLM comparison*, but it may **not exercise fulcrum's
  distinctive machinery** (the intransitivity/context-dependence novelty). To show
  the *novelty* helps on real data you likely need a benchmark that actually
  contains multi-criteria, context-varying conflicts — which may have to be
  constructed and justified. Do not claim the novelty is validated by BoardGameQA
  alone.
- On the **synthetic** task, the single-ordering baseline *is* faithful to
  ASPIC⁺/ABA⁺/VAF for the resolution question (the separation theorem proves those
  reduce to a single (pre)order there), so it is not a strawman. State this
  explicitly to pre-empt the "you didn't run real ASPIC⁺" objection on synthetic.
- The synthetic accuracy result is a controlled *illustration/ablation*, not
  external-validity evidence. Lead the empirical claims with the real benchmark.
