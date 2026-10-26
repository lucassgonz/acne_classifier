#!/bin/bash
# Ablacao embedding vs sem, no dataset Acne04+AcneLevel(tratado) (sem
# manual_annotations), split agrupado corretamente (data/final_v2,
# kfold_splits_v2.json).
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2 --data-dir ../data/final_v2 --splits results/kfold_splits_v2.json"

echo "=== with_embedding (v2) ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_v2_with $COMMON --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WITH_EMBED_V2 DONE"

echo "=== no_embedding (v2) ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_v2_without $COMMON --no-embed --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "NO_EMBED_V2 DONE"
echo "EMBED_ABLATION_V2 DONE"
