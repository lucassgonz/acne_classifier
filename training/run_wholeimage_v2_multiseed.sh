#!/bin/bash
# Ensemble multi-seed da config vencedora do sweep v2 (whole_v2_cbam_swa_tta,
# 77,54% acc / QWK 0,849 com 1 seed, melhor das 3 testadas: base/effnet/cbam).
# Seed 42 ja esta em results/kfold_wholeimage/whole_v2_cbam_swa_tta/; aqui
# rodamos 43-46.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
SPLITS="--splits results/kfold_splits_wholeimage_v2.json --manifest-v2 ../data/final_wholeimage_v2/manifest.json"

for seed in 43 44 45 46; do
  for fold in 0 1 2 3 4; do
    python run_wholeimage_kfold.py --name "whole_v2_cbam_swa_tta_seed${seed}" $SPLITS \
      --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --cbam --swa --swa-epochs 10 --tta-views 7 \
      --seed $seed --fold $fold \
      2>&1 | grep -v -i "warning\|warn("
  done
  echo "SEED $seed DONE"
done
echo "MULTISEED_V2_CBAM DONE"
