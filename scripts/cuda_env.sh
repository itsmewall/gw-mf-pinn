#!/usr/bin/env bash
set -Eeuo pipefail

add_path(){ local p="$1"; [[ -d "$p" ]] && [[ ":${LD_LIBRARY_PATH-}:" != *":$p:"* ]] && export LD_LIBRARY_PATH="$p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; }

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

# WSL e libs do Ubuntu
add_path "/usr/lib/wsl/lib"
add_path "/usr/lib/x86_64-linux-gnu"
add_path "/usr/local/lib"

# Paths do CUDA Toolkit instalado (ajuste a série se for 12-8 ou 12-6)
add_path "/usr/local/cuda-12.9/targets/x86_64-linux/lib"
add_path "/usr/local/cuda-12.9/lib64"

# Também um fallback genérico, se existir
add_path "/usr/local/cuda/targets/x86_64-linux/lib"
add_path "/usr/local/cuda/lib64"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
echo "[cuda_env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
