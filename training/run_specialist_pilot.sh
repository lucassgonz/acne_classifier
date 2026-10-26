#!/bin/bash
# Piloto: fold 0 das 5 regioes especialistas (ver run_region_specialist_kfold.py).
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2 --fold 0"

for region in forehead chin nose left_cheek right_cheek; do
  python run_region_specialist_kfold.py --region "$region" $COMMON 2>&1 | grep -v -i "warning\|warn("
done
echo "PILOT DONE"
