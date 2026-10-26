#!/bin/bash
# Mesma receita EXATA da replica do JAGC (SGD, 120 ep, LR 30/60/90, BN congelado,
# imagens 300px, split oficial, sem val), trocando SO o backbone para ResNet-18
# com pesos ImageNet -- isola quanto do resultado vem da receita vs. do ResNet-50.
cd "$(dirname "$0")"
source venv/bin/activate
for fold in 0 1 2 3 4; do
  python run_jagc_original_replica.py --fold $fold --arch resnet18 --name jagc_replica_resnet18 2>&1 | grep -v -i "warn\|F.softmax"
done
echo "JAGC_REPLICA_R18 DONE"
