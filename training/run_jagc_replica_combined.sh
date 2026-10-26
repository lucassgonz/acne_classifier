#!/bin/bash
# Receita exata do JAGC (ResNet-18 ImageNet, BN congelado, SGD, 120 ep fixas, 300px)
# no dataset COMBINADO Acne04+AcneLevel, mesmo split agrupado da melhor run anterior.
# Uso: ./run_jagc_replica_combined.sh [seed ...]   (default: 42)
cd "$(dirname "$0")"
source venv/bin/activate
for seed in ${@:-42}; do
  if [ "$seed" = "42" ]; then NAME=jagc_replica_combined_resnet18; else NAME=jagc_replica_combined_resnet18_seed${seed}; fi
  for fold in 0 1 2 3 4; do
    python run_jagc_original_replica.py --fold $fold --arch resnet18 --seed $seed --name $NAME \
      --splits results/kfold_splits_wholeimage_v2.json --manifest-v2 ../data/final_wholeimage_v2/manifest.json \
      --img-dir ../data/jagc_replica_300_combined 2>&1 | grep -v -i "warn\|F.softmax"
  done
  echo "SEED $seed DONE"
done
echo "JAGC_COMBINED DONE"
