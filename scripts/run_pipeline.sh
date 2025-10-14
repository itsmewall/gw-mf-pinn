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

# 2) Exporta libs de CUDA no WSL sem quebrar o sistema
bash "scripts/cuda_env.sh"

# 3) Garante CuPy funcional. Não mata se falhar, o MF tem fallback
bash "scripts/bootstrap_gpu.sh" "venv" || true

# 4) Roda o pipeline
echo "[RUN] Iniciando pipeline…"
python run.py