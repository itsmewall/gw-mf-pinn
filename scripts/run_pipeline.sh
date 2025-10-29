#!/usr/bin/env bash
set -euo pipefail

# pastas
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# venv
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
python --version

# CUDA env opcional
[ -f scripts/cuda_env.sh ] && source scripts/cuda_env.sh || true

# pacotes mínimos
python -m pip install -U pip >/dev/null
pip install -q numpy scipy pandas scikit-learn matplotlib pyarrow joblib tqdm "xgboost>=2.0" cupy-cuda12x pycbc

# ML na GPU
export ML_MODEL=xgb
export ML_XGB_DEVICE=cuda
export ML_STRICT_GPU=1
export ML_THRESH_STRATEGY=target_far
export ML_TARGET_FAR=1e-4

# PYTHONPATH com fallback vazio para não quebrar com set -u
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH-}"

echo "[RUN] Iniciando pipeline…"
python run.py
