#!/bin/bash
# Ablacao cross-dataset Acne04 x AcneLevel (tratado), as duas direcoes.
cd "$(dirname "$0")"
source venv/bin/activate

for src in acne04 acnelevel; do
  python run_cross_dataset.py --train-source $src --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "CROSS_DATASET DONE"
