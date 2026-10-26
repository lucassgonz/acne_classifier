#!/bin/bash
# Treino multi-task (severidade + contagem real de licoes) no split
# oficial do ACNE04, mesma receita CBAM+SWA+TTA do whole_official_cbam_swa_tta
# (76,64% acc, Secao IV-F do artigo) + cabeca de regressao de contagem,
# pra tentar fechar o gap contra o JAGC (84,11%, mesmo split, usa
# supervisao conjunta grading+counting).
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth

echo "=== official_multitask (CBAM+SWA+TTA+contagem real de licoes) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_multitask.py --name official_multitask --fold $fold \
    --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 --count-loss-weight 0.2 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "OFFICIAL_MULTITASK DONE"
