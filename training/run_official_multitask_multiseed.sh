#!/bin/bash
# Ensemble multi-seed do official_multitask (severidade+contagem real de
# licoes). Seed 42 ja esta em results/kfold_wholeimage/official_multitask/;
# aqui rodamos 43-46, pra reduzir a variancia de seed unico observada
# (ganho por fold variou de -1,71pp a +6,51pp).
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth

for seed in 43 44 45 46; do
  for fold in 0 1 2 3 4; do
    python run_wholeimage_multitask.py --name "official_multitask_seed${seed}" --fold $fold \
      --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 --count-loss-weight 0.2 \
      --seed $seed \
      2>&1 | grep -v -i "warning\|warn("
  done
  echo "SEED $seed DONE"
done
echo "MULTITASK_MULTISEED DONE"
