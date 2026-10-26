#!/bin/bash
# Ablacao prioritaria do revisor: embedding anatomico vs ResNet18 sem
# embedding, protocolo identico (mesmos hiperparametros, mesmos folds
# limpos), recorte por regiao. 2 configs x 5 folds = 10 rodadas.
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2"

echo "=== with_embedding ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_with $COMMON --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "WITH_EMBED DONE"

echo "=== no_embedding ==="
for fold in 0 1 2 3 4; do
  python run_kfold_clean.py --name embed_ablation_without $COMMON --no-embed --fold $fold \
    2>&1 | grep -v -i "warning\|warn("
done
echo "NO_EMBED DONE"
echo "EMBED_ABLATION DONE"
