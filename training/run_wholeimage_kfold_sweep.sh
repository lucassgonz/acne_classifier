#!/bin/bash
# Fila de k-fold "imagem inteira" (ver run_wholeimage_kfold.py), split limpo.
# Uso: cd training && bash run_wholeimage_kfold_sweep.sh [config ...]
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2"

CONFIG_NAMES=(whole_base whole_ordinal_ls)
CONFIG_FLAGS=(
  ""
  "--loss ordinal_ls --ordinal-tau 0.7"
)

flags_for() {
  local target="$1"
  local i
  for i in "${!CONFIG_NAMES[@]}"; do
    if [ "${CONFIG_NAMES[$i]}" = "$target" ]; then
      echo "${CONFIG_FLAGS[$i]}"
      return 0
    fi
  done
  echo "ERRO: config desconhecida '$target'" >&2
  exit 1
}

ORDER=("${CONFIG_NAMES[@]}")
[ $# -gt 0 ] && ORDER=("$@")

for name in "${ORDER[@]}"; do
  flags=$(flags_for "$name")
  echo "=== $name: flags=[$flags] ==="
  for fold in 0 1 2 3 4; do
    python run_wholeimage_kfold.py --name "$name" --fold $fold $COMMON $flags 2>&1 | grep -v -i "warning\|warn("
  done
  echo "CONFIG DONE: $name"
done
echo "SWEEP DONE"
