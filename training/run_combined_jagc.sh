#!/bin/bash
# JAGC-fiel (ResNet18+CBAM, epochs=120/patience=25) no dataset combinado
# Acne04 + AcneLevel(tratado), mesmo split agrupado do melhor resultado
# anterior (whole_v2_cbam_swa_tta, 79,04% em ensemble de 5 seeds). A contagem
# real de licoes vem do split oficial via identidade canonica da foto.
# Uso: ./run_combined_jagc.sh [seed ...]   (default: 42)
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
SEEDS=${@:-42}

for seed in $SEEDS; do
  if [ "$seed" = "42" ]; then NAME=combined_jagc_longepochs; else NAME=combined_jagc_longepochs_seed${seed}; fi
  for fold in 0 1 2 3 4; do
    python run_wholeimage_jagc.py --name "$NAME" --fold $fold \
      --splits results/kfold_splits_wholeimage_v2.json \
      --manifest-v2 ../data/final_wholeimage_v2/manifest.json \
      --epochs 120 --patience 25 \
      --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
      --seed $seed \
      2>&1 | grep -v -i "warning\|warn("
  done
  echo "COMBINED SEED $seed DONE"
done
echo "COMBINED_JAGC DONE"
