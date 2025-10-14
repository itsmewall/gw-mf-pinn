#!/usr/bin/env bash
set -Eeuo pipefail
say(){ printf '%s %s\n' "[repair]" "$*"; }

say "Pré-checagens WSL"
[[ -e /dev/dxg ]] || { echo "[ERRO] /dev/dxg ausente. Atualize o driver NVIDIA no Windows e faça 'wsl --shutdown'."; exit 1; }

say "NVIDIA keyring"
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
fi
sudo apt-get update -y

say "Selecionando série disponível"
SERIES=$(apt-cache pkgnames | grep -E '^cuda-toolkit-12-[0-9]+' | sed -E 's/^cuda-toolkit-//' | sort -V | tail -n1)
[[ -n "${SERIES:-}" ]] || { echo "[ERRO] não achei cuda-toolkit-12-x no repo"; exit 2; }
say "Instalando cuda-toolkit-$SERIES (runtime + cuFFT + NVRTC)"
sudo apt-get install -y --no-install-recommends "cuda-toolkit-$SERIES"

# Exporta caminhos e registra no linker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for p in /usr/local/cuda-"$SERIES"/targets/x86_64-linux/lib /usr/local/cuda-"$SERIES"/lib64 /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib64; do
  [[ -d "$p" ]] && export LD_LIBRARY_PATH="$p:$LD_LIBRARY_PATH"
done
sudo ldconfig || true
echo "[repair] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

say "Validação CuPy"
python - <<'PY'
import cupy as cp
from cupy.cuda import nvrtc
cp.fft.rfft(cp.zeros(1024, dtype=cp.float32))
nvrtc.Program(b'int main(){}', b"t.cu", (), ()).compile(("-arch=compute_80",))
print("OK: CuPy", cp.__version__, "| GPUs:", cp.cuda.runtime.getDeviceCount())
PY

say "Concluído."
