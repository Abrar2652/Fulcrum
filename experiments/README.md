# Experiments

Everything reported in the ICLR 2027 manuscript is produced by the scripts here and stored as
JSON under `results/`; `make_tables.py` turns the JSON into the paper's tables and number
macros, so no number is typed by hand. `bash scripts/reproduce_all.sh` runs the whole chain.

## Shared infrastructure
- `zoo.py` — the pairwise model zoo: transitive baselines (BT-linear, BT-MLP, lexicographic),
  non-transitive baselines (Condorcet rule, MAX-GAP rule, linear-in-d, context-BT, random gate,
  mixture of utilities, antisymmetric bilinear, blade-chest, pair-MLP) and FULCRUM-hard /
  FULCRUM-soft; training with early stopping and temperature scaling.
- `metrics.py` — accuracy, NLL, Brier, ECE; cyclic-triple and cycle-orientation metrics;
  conflict levels; paired t and paired bootstrap intervals.
- `ceiling.py` — the transitivity ceiling (best single ordering): exact lazily constrained
  linear-ordering ILP, LP/local-search bounds for large sparse graphs, the 1 − f₃/3 bound;
  `python experiments/ceiling.py` self-tests against brute force.
- `faithfulness.py` — attribution diagnostics (sign reversal, deletion, sufficiency,
  comprehensiveness) with random and least-attributed controls.
- `bench_common.py` — the runner used by every benchmark.

## Benchmarks (10 seeds each)
- `bench_synthetic.py` — World C (Condorcet oracle; r, noise, training-size and shuffle sweeps)
  and World M (MAX-GAP truth) with exact ceilings.
- `bench_pokemon.py` — full seven-criterion comparator on 50,000 Pokémon battle outcomes,
  prespecified slices, ceiling brackets (seeds 0–2).
- `bench_sushi.py` (+ `ceilings_post.py`) — Sushi-B rankings over 100 items with item attributes.
- `bench_helpsteer.py --hs_dir DIR` — HelpSteer2-Preference (five annotated aspects).

## Deterministic checks
- `exhaustive_checks.py` — preorders, k-cycles, grid counts, the 68 cycles under four aggregations,
  the aggregation theorem on random sets, Proposition 1 and its counter-example.
- `capacity_check.py` — the m = n MAX-GAP construction and grid realisability counts.
- `intransitivity_counts.py` — Netflix, Sushi and Pokémon cycle counts from the raw files.

## Earlier-draft studies (appendix)
`synthetic.py`, `run.py`, `ablations.py`, `conflict_experiment.py`, `conflict_ci.py`,
`pokemon_intransitivity.py`, `postulate_combining.py`, `boardgameqa_solver.py`,
`llm_local.py`, `llm_runner.py`; their numbers are recorded in `RESULTS.md` and reproduce exactly.

## Outputs
`make_tables.py` → `output/numbers.tex`, `tab_*.tex`; `make_fig1.py` → Figure 1;
`make_fig_results.py` → Figure 2; `summarize_md.py` prints a markdown summary of every result file.
