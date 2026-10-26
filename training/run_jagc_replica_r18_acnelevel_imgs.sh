#!/bin/bash
# Mesma receita/split oficial/ResNet-18, mas com as imagens da versao AcneLevel (Dataset/)
# em vez da versao Acne04 (acne_1024) -- isola o efeito da fonte das imagens.
cd "$(dirname "$0")"
source venv/bin/activate
for fold in 0 1 2 3 4; do
  python run_jagc_original_replica.py --fold $fold --arch resnet18 --name jagc_replica_resnet18_acnelevel_imgs \
    --img-dir ../data/jagc_replica_300_acnelevel 2>&1 | grep -v -i "warn\|F.softmax"
done
echo "R18_ACNELEVEL_IMGS DONE"
