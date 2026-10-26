#!/bin/bash
# Comparacao de referencia: mesma receita do run_combined_jagc.sh (JAGC,
# epochs=120/patience=25, CBAM, mixup, SWA, TTA, dataset combinado, seed 42),
# trocando SO o backbone por ResNet-50 (pesos ImageNet; o pretreino SCIN
# so existe para resnet18/34).
cd "$(dirname "$0")"
source venv/bin/activate
for fold in 0 1 2 3 4; do
  python run_wholeimage_jagc.py --name combined_jagc_longepochs_resnet50 --fold $fold \
    --arch resnet50 \
    --splits results/kfold_splits_wholeimage_v2.json \
    --manifest-v2 ../data/final_wholeimage_v2/manifest.json \
    --epochs 120 --patience 25 \
    --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "COMBINED_JAGC_RESNET50 DONE"
