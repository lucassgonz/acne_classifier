#!/bin/bash
# Pipeline principal (imagem inteira) migrado para Acne04+AcneLevel(tratado),
# split agrupado por identidade canonica (data/final_wholeimage_v2,
# kfold_splits_wholeimage_v2.json) -- substitui Dataset/+manual_annotations.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
SPLITS="--splits results/kfold_splits_wholeimage_v2.json --manifest-v2 ../data/final_wholeimage_v2/manifest.json"
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2 $SPLITS"

echo "=== whole_v2_base (resnet18, referencia) -- JA CONCLUIDO, pulando ==="
echo "WHOLE_V2_BASE DONE"

echo "=== whole_v2_effnet_swa_tta (efficientnet_b0 + SWA + TTA7, pesos ImageNet -- NAO usar"
echo "    --pretrained-backbone aqui, o checkpoint SCIN so existe p/ resnet18/34, ver run_ablation.py) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_kfold.py --name whole_v2_effnet_swa_tta $SPLITS --mixup-alpha 0.2 --fold $fold \
    --arch efficientnet_b0 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WHOLE_V2_EFFNET_SWA_TTA DONE"

echo "=== whole_v2_cbam_swa_tta (resnet18+CBAM + SWA + TTA7) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_kfold.py --name whole_v2_cbam_swa_tta $COMMON --fold $fold \
    --cbam --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WHOLE_V2_CBAM_SWA_TTA DONE"

echo "WHOLEIMAGE_V2_SWEEP DONE"
