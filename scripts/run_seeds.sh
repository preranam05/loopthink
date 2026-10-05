#!/usr/bin/env bash
# Credibility runs: 3 seeds x {CLINC150, HWU64} x {official, heldout} for the MiniLM router model,
# then the report with confidence intervals. ~25-30 min on an M-series Mac.
#   bash scripts/run_seeds.sh              # MiniLM only
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2}"
[ -f data/hwu64.json ] || python scripts/prepare_hwu64.py
for ds in clinc hwu; do
  if [ $ds = clinc ]; then DATA=data; NH=30; else DATA=data/hwu64.json; NH=16; fi
  for setup in official heldout; do
    for s in $SEEDS; do
      out=runs/minilm_${ds}_${setup}_s$s
      [ -f $out/eval/results.json ] && { echo "skip $out"; continue; }
      save=""; [ $s = 0 ] && save="--save_model"
      python scripts/encoder_baseline.py --setup $setup --clinc $DATA --n_heldout $NH --seed $s --out $out $save
    done
  done
done
python scripts/report.py
