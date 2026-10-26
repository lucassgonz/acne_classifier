#!/bin/bash
# whole_base + SWA + TTA (7 views), mesma receita do modelo final region-aware.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
for fold in 0 1 2 3 4; do
  python run_wholeimage_kfold.py --name whole_base_swa_tta --fold $fold \
    --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "SWA_TTA DONE"
