# scripts/cuda_env.sh
#!/usr/bin/env bash
set -Eeuo pipefail

# Este script pressupõe que o venv já está ativo (VIRTUAL_ENV definido).
: "${VIRTUAL_ENV:?Ative o venv antes de chamar cuda_env.sh}"

# Descobre o site-packages do venv atual
SITEPKG="$(
  python - <<'PY'
import site, sys
paths=[p for p in site.getsitepackages() if 'site-packages' in p]
print(paths[0] if paths else site.getusersitepackages())
PY
)"

# Pastas das libs CUDA empacotadas via pip (nvidia-*)
CUFFT_DIR="$SITEPKG/nvidia/cufft/lib"
CURAND_DIR="$SITEPKG/nvidia/curand/lib"
CRT_DIR="$SITEPKG/nvidia/cuda_runtime/lib"

# Garante que LD_LIBRARY_PATH exista
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Só adiciona se existir
add_path() {
  local p="$1"
  if [[ -d "$p" ]] && [[ ":$LD_LIBRARY_PATH:" != *":$p:"* ]]; then
    LD_LIBRARY_PATH="$p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

add_path "$CUFFT_DIR"
add_path "$CURAND_DIR"
add_path "$CRT_DIR"

export LD_LIBRARY_PATH
echo "[cuda_env] LD_LIBRARY_PATH configurado."