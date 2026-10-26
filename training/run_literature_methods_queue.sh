#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7"

echo "=== ACNE04-only (sem CBAM) - RETOMANDO FOLD 4 ==="
for fold in 4; do
  python run_wholeimage_kfold.py --name whole_acne04only $COMMON --acne04-only --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "EXPERIMENTOS FINALIZADOS"