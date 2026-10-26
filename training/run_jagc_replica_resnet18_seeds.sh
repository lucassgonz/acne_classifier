#!/bin/bash
# Mais seeds da receita exata do JAGC com ResNet-18 (split oficial), pra firmar o 80,8%.
cd "$(dirname "$0")"
source venv/bin/activate
for seed in "$@"; do
  for fold in 0 1 2 3 4; do
    python run_jagc_original_replica.py --fold $fold --arch resnet18 --seed $seed --name jagc_replica_resnet18_seed${seed} 2>&1 | grep -v -i "warn\|F.softmax"
  done
  echo "SEED $seed DONE"
done
echo "JAGC_R18_SEEDS DONE"
