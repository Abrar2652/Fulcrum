# Reproducing the Phase-6 experiments

Everything is deterministic (fixed seeds) and pure-Python (only `networkx` for the
library; the experiments use no extra deps).

```bash
pip install -e .          # or just ensure networkx is importable
PYTHONPATH=. python experiments/run.py         # H1/H3 synthetic (RESULTS.md top table)
PYTHONPATH=. python differentiable/h2.py       # H2 calibration (strength-typed vs non-typed)
PYTHONPATH=. python experiments/llm_local.py   # H1 plain-LLM baseline (local Qwen2.5-7B; GPU)
pytest -q tests/test_experiments.py tests/test_differentiable.py
```

`h2.py` needs torch (present); `llm_local.py` needs torch + transformers + a GPU and
reads models from a local HuggingFace cache (set `HF_HOME`, or run online). Both
deterministic except the LLM decode (greedy, so stable in practice).

## What runs
- `experiments/synthetic.py` — the conflict generator with known ground truth
  (regimes `context` and `transitive`; label noise ε=0.1; seeded).
- `experiments/run.py` — fits the full context calculus, the A2 single-ordering
  baseline, and the A1 single-head ablation; evaluates held-out accuracy across
  seeds `[11,22,33,44,55,66,77]` with 95% CIs; runs the H3 auditability
  intervention test. Reproduces `experiments/RESULTS.md` exactly.

## Determinism / integrity
- Seeds are fixed in `run.py::SEEDS` and threaded into the generator and split.
- The **negative control** (transitive regime → all models tie) is asserted in
  `tests/test_experiments.py`; if a change ever rigs the generator so the calculus
  wins spuriously, that test fails. This is the anti-p-hacking guard.

## Not reproduced here (deferred, see RESULTS.md §Deferred)
Established-library ABA⁺/EAF/prioritized-default-logic baselines (env-blocked);
H2 calibration (needs Phase 5); BoardGameQA + real data; A3/A4 ablations.
