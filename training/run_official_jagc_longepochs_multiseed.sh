#!/bin/bash
# Ensemble multi-seed do official_jagc_longepochs (JAGC fiel, epochs=120
# patience=25). Seed 42 ja esta em
# results/kfold_wholeimage/official_jagc_longepochs/; aqui rodamos 43-46
# pra ter um numero final robusto, igual fizemos pro official_multitask.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth

for seed in 43 44 45 46; do
  for fold in 0 1 2 3 4; do
    python run_wholeimage_jagc.py --name "official_jagc_longepochs_seed${seed}" --fold $fold \
      --epochs 120 --patience 25 \
      --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
      --seed $seed \
      2>&1 | grep -v -i "warning\|warn("
  done
  echo "SEED $seed DONE"
done
echo "JAGC_LONGEPOCHS_MULTISEED DONE"
