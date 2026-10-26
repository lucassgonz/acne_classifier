#!/bin/bash
# Ensemble multi-seed do whole_base + SWA + TTA (receita final do artigo:
# 5 seeds, SWA, mixup 0.2, TTA 7 views). Seed 42 ja esta em
# results/kfold_wholeimage/whole_base_swa_tta/; aqui rodamos 43-46.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
for seed in 43 44 45 46; do
  for fold in 0 1 2 3 4; do
    python run_wholeimage_kfold.py --name "whole_base_swa_tta_seed${seed}" --fold $fold \
      --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
      --seed $seed \
      2>&1 | grep -v -i "warning\|warn("
  done
  echo "SEED $seed DONE"
done
echo "MULTISEED DONE"
