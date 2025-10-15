#!/usr/bin/env bash
set -euo pipefail

# Raiz do projeto é a pasta do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# 1) Ativa venv
if [ ! -d "venv" ]; then
  echo "[RUN] Criando venv..."
  python3 -m venv venv
fi
source "venv/bin/activate"

python --version

# 2) FAST pipeline opcional
# 1 = rápido para iteração; 0 = completo para resultados finais
export FAST_PIPELINE=0

# 3) Carrega variáveis de CUDA no shell atual
# use 'source' para persistir
if [ -f scripts/cuda_env.sh ]; then
  source scripts/cuda_env.sh
fi

# 4) Bootstrap GPU. Pode falhar sem abortar, MF tem fallback
bash scripts/bootstrap_gpu.sh "venv" || true

# 5) Roda o pipeline
echo "[RUN] Iniciando pipeline…"
python run.py