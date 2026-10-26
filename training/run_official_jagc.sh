#!/bin/bash
# Replica fiel da TECNICA do JAGC (3 saidas, perdas KL sobre distribuicao
# de contagem) com backbone ResNet18+CBAM proprio (nao ResNet50), no
# split oficial do ACNE04 -- ver run_wholeimage_jagc.py pro detalhe.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth

echo "=== official_jagc (ResNet18+CBAM, tecnica JAGC: contagem como distribuicao + consistencia) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_jagc.py --name official_jagc --fold $fold \
    --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "OFFICIAL_JAGC DONE"
