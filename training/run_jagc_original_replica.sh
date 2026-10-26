#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== jagc_original_replica (codigo original do JAGC, ResNet-50 ImageNet+BN congelado, 120 epocas, split oficial) ==="
for fold in 0 1 2 3 4; do
  python run_jagc_original_replica.py --fold $fold 2>&1 | grep -v -i "warn\|F.softmax"
done
echo "JAGC_REPLICA DONE"
