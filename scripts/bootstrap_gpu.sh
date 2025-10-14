#!/usr/bin/env bash
set -Eeuo pipefail

say(){ printf '%s %s\n' "[bootstrap_gpu]" "$*"; }

# 1) Checa cuFFT via CuPy (suficiente para o MF)
python - <<'PY' 2>/dev/null && exit 0 || true
import cupy as cp
cp.fft.rfft(cp.zeros(1024, dtype=cp.float32))  # usa cuFFT
print("OK: CuPy", cp.__version__, "| GPUs:", cp.cuda.runtime.getDeviceCount())
PY

say "Checando CuPy/cuFFT/NVRTC… falhou na primeira tentativa. Tentando reparar libs CUDA."

# 2) Reparo minimalista de user-space (WSL): instala meta-pacote do toolkit se faltar
if ! dpkg -s cuda-toolkit-12-9 >/dev/null 2>&1 && ! dpkg -s cuda-toolkit-12-8 >/dev/null 2>&1; then
  say "NVIDIA keyring e update do apt"
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
  fi
  sudo apt-get update -y
  # pega a série disponível mais recente
  SERIES=$(apt-cache pkgnames | grep -E '^cuda-toolkit-12-[0-9]+' | sed -E 's/^cuda-toolkit-//' | sort -V | tail -n1)
  [[ -n "${SERIES:-}" ]] || { echo "[bootstrap_gpu] ERRO: não achei cuda-toolkit-12-x no repo"; exit 1; }
  say "Instalando cuda-toolkit-$SERIES (somente user-space)"
  sudo apt-get install -y --no-install-recommends "cuda-toolkit-$SERIES"
fi

# 3) Recarrega cache do linker (ok no WSL)
sudo ldconfig || true

# 4) Validação final: cuFFT (obrigatório) + NVRTC (opcional) sem usar Program
python - <<'PY'
import cupy as cp
from cupy.cuda import runtime
print("CuPy:", cp.__version__, "| GPUs:", runtime.getDeviceCount())
# cuFFT
cp.fft.rfft(cp.zeros(1024, dtype=cp.float32))
print("cuFFT OK")
# NVRTC opcional: só consulta versão se o backend existir
try:
    from cupy_backends.cuda.libs import nvrtc
    v = getattr(nvrtc, 'getVersion', lambda: 'unknown')()
    print("NVRTC backend disponível:", v)
except Exception as e:
    print("NVRTC backend não crítico:", e)
PY

say "Concluído."
