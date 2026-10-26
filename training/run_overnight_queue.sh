#!/bin/bash
# Espera o multiseed.log terminar e encadeia o proximo teste (EfficientNet-B0)
# sem precisar de intervencao -- o Mac vai ficar ligado a noite toda.
cd "$(dirname "$0")"
source venv/bin/activate

MULTISEED_LOG=/private/tmp/claude-501/-Users-GonLu-Documents-Projetos-pibic-final/927f7222-5870-4ba0-a727-5729557b32fe/scratchpad/multiseed.log
until grep -q "MULTISEED DONE" "$MULTISEED_LOG" 2>/dev/null; do sleep 60; done
echo "=== multiseed terminou, iniciando EfficientNet-B0 ==="

for fold in 0 1 2 3 4; do
  python run_wholeimage_kfold.py --name whole_effnet_b0 --fold $fold \
    --arch efficientnet_b0 --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "EFFNET DONE"
echo "OVERNIGHT QUEUE DONE"
