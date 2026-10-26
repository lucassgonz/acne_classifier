#!/bin/bash
# Treina a receita vencedora (ResNet18+CBAM+SWA+TTA) no split OFICIAL de
# 5-fold do ACNE04 (NNEW_trainval/test, Wu et al. ICCV2019), pra permitir
# comparacao direta com JAGC (84,11% acc) e trabalhos que reusaram esse
# split -- resolve o item 6 do revisor de forma literal, nao aproximada.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
SPLITS="--splits results/kfold_splits_official_acne04.json --official-split"

echo "=== whole_official_cbam_swa_tta (ResNet18+CBAM+SWA+TTA, split oficial ACNE04) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_kfold.py --name whole_official_cbam_swa_tta $SPLITS \
    --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --cbam --swa --swa-epochs 10 --tta-views 7 --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WHOLE_OFFICIAL_CBAM DONE"
