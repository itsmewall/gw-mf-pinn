#!/usr/bin/env bash
set -euo pipefail

# raiz do projeto
PROJ="/home/wallace/gw-mf-pinn"
cd "$PROJ"

# ativa venv
source venv/bin/activate

# turbo no WSL
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_MAX_THREADS=12

# CUDA/PyTorch mais estável
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# pasta de logs por rodada
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$PROJ/logs/weekend_$STAMP"
mkdir -p "$LOGDIR"

run() {
  local name="$1"
  shift
  echo "[$(date +%H:%M:%S)] INICIANDO $name..."
  # grava stdout+stderr em arquivo e mantém na tela
  "$@" 2>&1 | tee "$LOGDIR/${name}.log"
  echo "[$(date +%H:%M:%S)] FINALIZADO $name."
}

# 1. dumpa scores atuais
run "dump_scores"          python -m src.exp.dump_scores

# 2. varre FAR do MF2
run "mf2_far_sweep"        python -m src.exp.mf2_far_sweep

# 3. hard negative mining
run "mf2_hardneg"          python -m src.exp.mf2_hardneg

# 4. treino longo do MF2
run "mf2_long_train"       python -m src.exp.mf2_long_train

# 5. multi seed (ou multi series, troca o nome aqui se o arquivo for outro)
run "mf2_multi_seed"       python -m src.exp.mf2_multi_seed

echo "[$(date +%H:%M:%S)] WEEKEND OK. Logs em $LOGDIR"
