#!/usr/bin/env bash
# Configura LD_LIBRARY_PATH com as libs CUDA empacotadas no site-packages do venv.
# Uso:  source venv/bin/activate && source scripts/cuda_env.sh

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[cuda_env] Ative o venv primeiro:  source venv/bin/activate" >&2
  return 1 2>/dev/null || exit 1
fi

SITEPKG="$(python - <<'PY'
import site
cands=[p for p in site.getsitepackages() if 'site-packages' in p]
print(cands[0] if cands else site.getusersitepackages())
PY
)"

# Caminhos das libs empacotadas pelos wheels NVIDIA
CUFFT_DIR="$SITEPKG/nvidia/cufft/lib"
CURAND_DIR="$SITEPKG/nvidia/curand/lib"
CUDART_DIR="$SITEPKG/nvidia/cuda_runtime/lib"

# Verificações rápidas
[[ -e "$CUFFT_DIR/libcufft.so.11" ]] || { echo "[cuda_env] libcufft.so.11 não encontrado em $CUFFT_DIR"; return 1 2>/dev/null || exit 1; }
[[ -e "$CURAND_DIR/libcurand.so.10" ]] || { echo "[cuda_env] libcurand.so.10 não encontrado em $CURAND_DIR"; return 1 2>/dev/null || exit 1; }

export LD_LIBRARY_PATH="$CUFFT_DIR:$CURAND_DIR:$CUDART_DIR:${LD_LIBRARY_PATH:-}"
echo "[cuda_env] LD_LIBRARY_PATH configurado."