#!/bin/bash
# Mesma tecnica de run_official_jagc.sh, mas com orcamento de treino maior
# (epochs=120, patience=25 em vez do default 40/10) para testar se o gap
# remanescente para o JAGC original (84.11%) vem de treino mais longo.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth

echo "=== official_jagc_longepochs (ResNet18+CBAM, JAGC, epochs=120 patience=25) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_jagc.py --name official_jagc_longepochs --fold $fold \
    --epochs 120 --patience 25 \
    --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "OFFICIAL_JAGC_LONGEPOCHS DONE"
