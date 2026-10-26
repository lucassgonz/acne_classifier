#!/bin/bash
# Repete a ablacao embedding v3 (crops filtrados por visibilidade,
# data/final_v3) mas com o MESMO arsenal usado no whole-image (SWA+TTA),
# pedido pelo usuario pra elevar os numeros (58,36%/56,68% bare, baixos
# demais pro artigo) mantendo a comparacao with/without.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2 --data-dir ../data/final_v3 --splits results/kfold_splits_v3.json --swa --swa-epochs 10 --tta-views 7"

echo "=== with_embedding (v3, SWA+TTA) ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_v3_arsenal_with $COMMON --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WITH_EMBED_V3_ARSENAL DONE"

echo "=== no_embedding (v3, SWA+TTA) ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_v3_arsenal_without $COMMON --no-embed --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "NO_EMBED_V3_ARSENAL DONE"
echo "EMBED_ABLATION_V3_ARSENAL DONE"
