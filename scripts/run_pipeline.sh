# scripts/run_pipeline.sh
#!/usr/bin/env bash
set -Eeuo pipefail

# Raiz do projeto = pasta pai de scripts/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Venv
VENV="$ROOT/venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[run] venv não encontrado em $VENV"
  echo "[run] crie com: python3 -m venv \"$VENV\" && source \"$VENV/bin/activate\" && pip install -U pip && pip install -r requirements.txt"
  exit 1
fi

# Ativa venv
source "$VENV/bin/activate"

# CUDA env (usa libs do venv: libcufft/libcurand/cuda_runtime)
source "$ROOT/scripts/cuda_env.sh"

# Sanity: GPU e Python
python -V
echo "[RUN] Iniciando pipeline…"

# PYTHONPATH para importar src/* corretamente
export PYTHONPATH="$ROOT"

# Roda pipeline completo
python "$ROOT/run.py"