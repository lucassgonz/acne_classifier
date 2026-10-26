#!/bin/bash
# Testa o CRONOGRAMA de treino original do JAGC (Wu et al., train.py, e
# confirmado pelo AcneGrader/Liu et al. 2022): SGD+momentum=0.9,
# weight_decay=5e-4, LR=0.001 em escada (x0.5 nas epocas 30/60/90),
# 120 epocas fixas, SEM early stopping -- em vez do Adam+ReduceLROnPlateau
# com parada antecipada que usamos nos demais experimentos. Mantem nosso
# backbone ResNet18+CBAM pretreinado, mixup, SWA e TTA (nossa "essencia").
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth

echo "=== official_jagc_fixedschedule (ResNet18+CBAM, JAGC tecnica + cronograma original: SGD, 120 epocas fixas, LR escada 30/60/90) ==="
for fold in 0 1 2 3 4; do
  python run_wholeimage_jagc.py --name official_jagc_fixedschedule --fold $fold \
    --fixed-schedule --epochs 120 --lr-milestones 30,60,90 --lr-decay 0.5 \
    --lr 0.001 --momentum 0.9 --weight-decay 5e-4 --batch-size 32 \
    --pretrained-backbone $BACKBONE --mixup-alpha 0.2 --swa --swa-epochs 10 --tta-views 7 \
    2>&1 | grep -v -i "warning\|warn("
done
echo "OFFICIAL_JAGC_FIXEDSCHEDULE DONE"
