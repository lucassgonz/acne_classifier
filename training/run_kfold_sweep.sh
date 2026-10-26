#!/bin/bash
# Fila de configuracoes de otimizacao avaliadas por k-fold (5 folds, um processo por fold).
# Uso: cd training && bash run_kfold_sweep.sh [config ...]   (sem argumentos roda todas)
#
# NAO use `declare -A` aqui: o /bin/bash padrao do macOS e a 3.2 (2007), sem
# suporte a array associativo -- `declare -A` falha silenciosamente e todo
# CONFIGS[nome]=valor cai no MESMO indice (o ultimo sobrescreve os demais),
# fazendo toda config rodar com os mesmos hiperparametros sem erro visivel.
# Por isso usamos dois arrays indexados em paralelo (nome <-> flags).
cd "$(dirname "$0")"
source venv/bin/activate

BACKBONE=models/backbone_scin_pretrained_v2.pth
COMMON="--pretrained-backbone $BACKBONE --mixup-alpha 0.2"

CONFIG_NAMES=(adam_base adamw adamw_cos_ls sgd_cos ordinal_ls adamw_cos_ls_head)
CONFIG_FLAGS=(
  ""
  "--optimizer adamw --weight-decay 0.01"
  "--optimizer adamw --weight-decay 0.05 --scheduler cosine --label-smoothing 0.1 --lr 2e-4"
  "--optimizer sgd --lr 0.01 --weight-decay 5e-4 --scheduler cosine"
  "--loss ordinal_ls --ordinal-tau 0.7"
  "--optimizer adamw --weight-decay 0.05 --scheduler cosine --label-smoothing 0.1 --lr 1e-4 --head-lr-mult 10"
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
    python run_kfold_clean.py --name "$name" --fold $fold $COMMON $flags 2>&1 | grep -v -i "warning\|warn("
  done
  echo "CONFIG DONE: $name"
  python summarize_kfold.py --baseline adam_base 2>&1 | grep -v -i warning
done
echo "SWEEP DONE"
