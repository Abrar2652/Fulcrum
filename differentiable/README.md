# Differentiable fulcrum (Phase 5)

Gate signed (outcome A) — this layer is now ACTIVE. A learnable, **auditable**,
calibrated lift of the preponderance operator, in the spirit of Potyka's
"Interpreting Neural Networks as Gradual Argumentation Frameworks"
(arXiv:2012.05738). The non-differentiable calculus (`fulcrum/`) is the interpretable
baseline; the learned version must *earn* its complexity by beating it (it does, on
context-dependent conflicts).

## What it is (deliberately transparent, not a black box)
- `model.py::DifferentiablePreponderance` — verdict `P(a≻b | ctx) = σ((w(ctx)·d)/T)`,
  where `d = s(a)−s(b)`, `w(ctx)` are learned per-context head weights, `T` a
  calibration temperature. The **strength gate** relaxes certain/probable to a scalar:
  `+λ·(strength_a−strength_b)` makes a certain indicant dominate a probable one,
  vanishing among equal-strength (probable) pairs.
- **Auditable**: the verdict is a weighted sum of per-head differences, so for
  every conflict we read off the *responsible head* = `argmax_h |w_h·d_h|` and the
  full per-head contribution vector (`audit.py`). No opaque MLP.
- Ablation switch `context_conditioned=False` = the fixed-weight baseline.

## Run
```bash
PYTHONPATH=. python differentiable/train.py       # learned vs fixed: acc / ECE / Brier / auditability
pytest -q tests/test_differentiable.py
```

## Result (synthetic, seed 0, temperature-scaled)
| regime | model | acc | ECE | Brier | audit (resp/other) |
|---|---|---|---|---|---|
| context-dependent | **learned** | **0.91** | 0.09 | **0.10** | 0.98 / 0.02 |
| context-dependent | fixed | 0.65 | 0.05 | 0.21 | 0.71 / 0.21 |
| transitive (control) | learned | 0.91 | 0.09 | 0.09 | 1.00 / 0.00 |
| transitive (control) | fixed | 0.91 | 0.09 | 0.09 | 1.00 / 0.00 |

- **Learned earns its complexity**: on context-dependent conflicts it beats the
  fixed-weight model on accuracy (0.91 vs 0.65) and Brier (0.10 vs 0.21).
- **Negative control**: on single-ordering data the two tie — no spurious win.
- **Auditability**: negating the responsible head's difference flips the verdict
  ~98–100% vs ~0–2% for another head — the emitted explanation is causal.
- **Honest nuance**: the learned model is marginally *less* calibrated in ECE than
  the (humble, low-accuracy) fixed model; Brier — a proper scoring rule combining
  calibration and sharpness — favours the learned model.

## Scope / next
Delivers the differentiable operator, auditability report generator, and
calibration machinery (weight decay + temperature scaling). The full **H2** test
(does strength-typing improve calibration vs a non-typed baseline?) needs a
*strength-aware* synthetic generator (certain evidence genuinely more reliable) —
the model supports it (`strength_a/strength_b`); the generator extension is the next
step. Full gradual-AF acceptability propagation (Potyka) is a further extension; the
preponderance operator is where the novelty lives, so it is lifted first.
