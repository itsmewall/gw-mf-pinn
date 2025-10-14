#!/usr/bin/env bash
set -euo pipefail

echo "[repair] 0) Pré-checagens WSL/GPU"
if [ ! -e /dev/dxg ]; then
  echo "[ERRO] /dev/dxg não existe. Ative GPU no WSL (Windows: driver NVIDIA atualizado) e reinicie: wsl --shutdown"
  exit 1
fi

echo "[repair] 1) Garantindo keyring NVIDIA"
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
fi
sudo apt-get update -y

echo "[repair] 2) Purga de pacotes quebrados/antigos (sem remover drivers do Windows)"
sudo apt-get -y purge 'nvidia-cuda-toolkit' 'cuda-toolkit-*' 'cuda-demo-suite-*' 'cuda-drivers*' || true
sudo apt-get -y purge 'cuda-*' || true
sudo apt-get -y autoremove --purge || true
sudo apt-get -y clean

echo "[repair] 3) Descobrir série CUDA disponível (12-x)"
SERIES="$(apt-cache pkgnames cuda-libraries- | grep -E '12-[0-9]+' | sort -V | tail -n1 || true)"
if [ -z "${SERIES}" ]; then
  echo "[ERRO] Não achei cuda-libraries-12-x no repo. Saída debug:"
  apt-cache pkgnames cuda-libraries- | sort
  exit 1
fi
echo "[repair] Série detectada: ${SERIES}"

echo "[repair] 4) Instalar somente runtime (compat + libs + nvrtc + cudart)"
sudo apt-get install -y \
  "cuda-compat-${SERIES}" \
  "cuda-libraries-${SERIES}" \
  "cuda-nvrtc-${SERIES}" \
  "cuda-cudart-${SERIES}" || {
    echo "[warn] pacote cuda-cudart-${SERIES} pode não existir isolado; tentando seguir com libs padrão"
}

echo "[repair] 5) Atualizar cache de libs e listar o que interessa"
sudo ldconfig
ldconfig -p | grep -E 'libcufft\.so|libnvrtc\.so|libcudart\.so' || true

echo "[repair] 6) Ajustar LD_LIBRARY_PATH (CUDA + WSL)"
# Atualize seu env de runtime do projeto; não grava permanente aqui
append_ld () { local p="$1"; [ -d "$p" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":$p:"* ]] && export LD_LIBRARY_PATH="$p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; }
append_ld "/usr/lib/wsl/lib"
append_ld "/usr/lib/x86_64-linux-gnu"
for p in /usr/local/cuda*/lib64 /usr/local/cuda-*/targets/x86_64-linux/lib /usr/local/cuda/targets/x86_64-linux/lib; do
  append_ld "$p"
done
echo "[repair] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "[repair] 7) Alinhar CuPy ao CUDA 12.x"
source ./venv/bin/activate
pip install -U --force-reinstall "cupy-cuda12x==13.6.0" fastrlock

echo "[repair] 8) Validação rápida CuPy + cuFFT + NVRTC"
python - <<'PY'
import sys
try:
    import cupy as cp
    from cupy.cuda import nvrtc
    import numpy as np
    print("CuPy:", cp.__version__)
    # FFT sanity
    cp.fft.rfft(cp.zeros(1024, dtype=cp.float32))
    # NVRTC sanity
    src = r'extern "C" __global__ void k(float* x){ int i=blockDim.x*blockIdx.x+threadIdx.x; x[i]+=1.0f; }'
    prog = nvrtc.Program(src.encode(), b"k.cu", (), ())
    prog.compile(("-arch=compute_80",))
    print("FFT e NVRTC OK")
    print("GPUs visíveis:", cp.cuda.runtime.getDeviceCount())
except Exception as e:
    print("VALIDATION_FAIL:", repr(e))
    sys.exit(2)
PY

echo "[repair] OK. Rode seu pipeline com GPU."
