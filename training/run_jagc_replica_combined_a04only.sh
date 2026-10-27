#!/bin/bash
# Receita exata do JAGC (ResNet-18) nos folds do dataset combinado, mas SO com as imagens
# Acne04 (sem as copias tratadas do AcneLevel) -- isola o efeito de ter as duas versoes
# de cada foto no treino. Uso: ./run_jagc_replica_combined_a04only.sh [seed ...]
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
for seed in ${@:-42}; do
  if [ "$seed" = "42" ]; then NAME=jagc_replica_combined_a04only; else NAME=jagc_replica_combined_a04only_seed${seed}; fi
  for fold in 0 1 2 3 4; do
    python run_jagc_original_replica.py --fold $fold --arch resnet18 --seed $seed --name $NAME \
      --splits results/kfold_splits_wholeimage_v2.json --manifest-v2 ../data/final_wholeimage_v2/manifest.json \
      --img-dir ../data/jagc_replica_300_combined --only-source a04 2>&1 | grep -v -i "warn\|F.softmax"
  done
  echo "SEED $seed DONE"
done
echo "A04ONLY DONE"
