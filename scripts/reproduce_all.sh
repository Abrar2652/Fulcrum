#!/usr/bin/env bash
# Regenerate every number, table and figure of the ICLR 2027 manuscript from scratch.
# Runtime on one GPU + 32 CPU cores: about 3-4 hours, dominated by the Pokemon ceiling LP.
# Usage: bash scripts/reproduce_all.sh [HS_DIR]
#   HS_DIR: directory holding the HelpSteer2 files (train.jsonl.gz, validation.jsonl.gz,
#           preference/preference.jsonl.gz), e.g. downloaded with huggingface_hub from
#           nvidia/HelpSteer2 (CC-BY-4.0). Default: data/helpsteer2
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
HS_DIR="${1:-data/helpsteer2}"
mkdir -p experiments/results

echo "== deterministic checks behind the theorems"
python3 experiments/exhaustive_checks.py      > experiments/results/exhaustive_checks.log
python3 experiments/capacity_check.py         > experiments/results/capacity_check.log
python3 experiments/ceiling.py                                # self-test vs brute force
python3 experiments/intransitivity_counts.py  > experiments/results/intransitivity_counts.log

echo "== earlier-draft studies (appendix)"
python3 scripts/separation_witness.py         > experiments/results/witness.log
python3 experiments/postulate_combining.py    > experiments/results/postulate_combining.log
python3 experiments/run.py                    > experiments/results/tag_study.log
python3 experiments/ablations.py              > experiments/results/tag_ablations.log
python3 differentiable/h2.py                  > experiments/results/calibration.log
python3 experiments/pokemon_intransitivity.py > experiments/results/pokemon_single_term.log

echo "== benchmarks (10 seeds each)"
python3 experiments/bench_synthetic.py
python3 experiments/bench_pokemon.py
python3 experiments/bench_sushi.py
python3 experiments/ceilings_post.py 1500      # Sushi-B ceiling brackets (seeds 0-2)
if [ -f "$HS_DIR/preference/preference.jsonl.gz" ]; then
  python3 experiments/bench_helpsteer.py --hs_dir "$HS_DIR"
else
  echo "HelpSteer2 files not found in $HS_DIR; skipping (download from nvidia/HelpSteer2)"
fi

echo "== tables, numbers and figures"
python3 experiments/make_tables.py
python3 experiments/make_fig1.py
python3 experiments/make_fig_results.py

echo "== tables and figures written to output/"
