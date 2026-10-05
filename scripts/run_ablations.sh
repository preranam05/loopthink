#!/usr/bin/env bash
# Ablations (200M tokens each). Run on Kaggle, one per session if needed; all runs are resumable.
# Usage: DATA=/kaggle/input/fineweb-tok OUT=/kaggle/working/abl bash scripts/run_ablations.sh
set -euo pipefail
DATA=${DATA:?}; OUT=${OUT:?}; TOK=${TOK:-2e8}
common="--data_dir $DATA --tokens $TOK --precision fp16 --resume"
python -m loopthink.pretrain $common --out $OUT/no_injection  --no_injection
python -m loopthink.pretrain $common --out $OUT/bptt2         --bptt 2
python -m loopthink.pretrain $common --out $OUT/bptt8         --bptt 8
python -m loopthink.pretrain $common --out $OUT/fixed_r6      --fixed_r 6
python -m loopthink.pretrain $common --out $OUT/random_r      # reference arm
for arm in no_injection bptt2 bptt8 fixed_r6 random_r; do
  python -m loopthink.finetune --init $OUT/$arm/final.pt --tokenizer $DATA/tokenizer.json --setup heldout --out $OUT/$arm/ft
  python -m loopthink.evaluate --run $OUT/$arm/ft
done
# deep supervision on/off (fine-tune only, from the reference pretrain)
python -m loopthink.finetune --init $OUT/random_r/final.pt --tokenizer $DATA/tokenizer.json --setup heldout --ds none --out $OUT/random_r/ft_no_ds
python -m loopthink.evaluate --run $OUT/random_r/ft_no_ds
python scripts/summarize.py $OUT/*/ft* > $OUT/ablations.md
