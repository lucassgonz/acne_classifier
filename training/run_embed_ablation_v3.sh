#!/bin/bash
# Ablacao embedding vs sem, no dataset Acne04+AcneLevel(tratado), com
# FILTRO de confianca de regiao (so gera crop quando a regiao esta de
# fato visivel no quadro, ver utils/build_acne04_acnelevel_crops_v3.py e
# utils/audit_region_visibility.py -- achado: ~50-65% dos crops de
# queixo/bochecha no v2 eram invalidos, cortados na borda).
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2 --data-dir ../data/final_v3 --splits results/kfold_splits_v3.json"

echo "=== with_embedding (v3, filtrado) ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_v3_with $COMMON --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WITH_EMBED_V3 DONE"

echo "=== no_embedding (v3, filtrado) ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_v3_without $COMMON --no-embed --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "NO_EMBED_V3 DONE"
echo "EMBED_ABLATION_V3 DONE"
