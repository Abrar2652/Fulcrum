# FULCRUM

Code, tests and stored per-seed results for the submission "FULCRUM: Pair-Dependent Preferences
Beyond Transitive Rankings". Everything reported in the paper can be regenerated from this
archive plus the public datasets listed below.

## Layout

- `fulcrum/`        the discrete calculus: criterion-ordering policies (incl. MAX-GAP), preponderance,
                    strength gate, attack-to-defeat map, Dung semantics, structured (ASPIC+-style) layer
- `differentiable/` the soft-gated comparator (criterion weights, antisymmetric score, position intercept),
                    training, calibration and attribution utilities
- `experiments/`    benchmarks (`bench_synthetic.py` Worlds C and M, `bench_pokemon.py`, `bench_sushi.py`,
                    `bench_helpsteer.py`), the model zoo (`zoo.py`), metrics, the transitivity-ceiling
                    solver (`ceiling.py`), exhaustive finite checks behind the theorems
                    (`exhaustive_checks.py`, `capacity_check.py`, `postulate_combining.py`,
                    `intransitivity_counts.py`), table/figure generation, and `results/*.json` with the
                    per-seed numbers used in the paper
- `scripts/`        `separation_witness.py` (machine-checked separation), `gate_search.py`,
                    `reproduce_all.sh` (full regeneration)
- `tests/`          unit and property tests (pytest + hypothesis)
- `data/preflib/`   the five PrefLib Netflix profiles used for the cycle counts (PrefLib, open data)

## Setup

Python >= 3.10. `pip install -e ".[dev]"` installs the core library and test tools; the
experiments additionally need `torch`, `numpy` and `matplotlib`; the ceiling solver uses `pulp`
(CBC). Runs were made on one Quadro RTX 8000 and a 32-core CPU; the full regeneration takes about
3-4 hours, dominated by the Pokemon ceiling bracket.

## Datasets (not redistributed here)

- Sushi preference data (Kamishima): download `sushi3-2016.zip` from the dataset page and unpack to
  `data/sushi3-2016/` (its licence forbids redistribution).
- Pokemon battle outcomes: Kaggle dataset `terminus7/pokemon-challenge` (`pokemon.csv`,
  `combats.csv`) into `data/pokemon/`. The dataset author states that the outcomes were generated
  by a simplified battle algorithm, not recorded from play.
- HelpSteer2 / HelpSteer2-Preference (CC-BY-4.0): see `experiments/bench_helpsteer.py` for the
  expected files under `data/helpsteer2/`.
- BoardgameQA (appendix only): `git clone https://github.com/google/BoardgameQA data/BoardgameQA`.

## Reproduction

```bash
export PYTHONPATH=.
pytest -q                                      # tests
python experiments/exhaustive_checks.py        # finite checks behind the theorems
python scripts/separation_witness.py           # 29 / 355 preorders, k-cycles
python experiments/bench_synthetic.py          # Worlds C and M, 10 seeds
python experiments/bench_pokemon.py            # full multi-criterion comparator
python experiments/bench_sushi.py; python experiments/bench_helpsteer.py
python experiments/ceilings_post.py            # ceiling brackets on seeds 0-2
python experiments/make_tables.py              # tables and numbers.tex into output/
python experiments/make_fig1.py; python experiments/make_fig_results.py; python experiments/make_fig_wins.py
```
or `bash scripts/reproduce_all.sh`. All runs are deterministic given the seed.

## Note on the cycle-orientation metric

`zoo.predict_proba_core` and `bench_common.evaluate` now derive relation-level metrics
(cycle-orientation accuracy) from the antisymmetric score alone, without the shared position
intercept. The stored result files were produced with the intercept included, which is the
quantity reported in the paper and explained in its Appendix F; rerunning the benchmarks with
this archive regenerates the metric in the corrected form.
