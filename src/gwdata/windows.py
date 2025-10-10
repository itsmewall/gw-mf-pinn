# src/gwdata/windows.py
"""
Gera janelas deslizantes a partir dos arquivos whitened de data/interim
e calcula métricas simples de SNR por janela.

Saídas (HDF5 em data/processed):
  /windows/start_idx   (int64)     índice inicial de cada janela no vetor whitened
  /windows/start_gps   (float64)   tempo GPS aproximado para cada janela
  /windows/snr_rms     (float32)   proxy de SNR por RMS
  /windows/snr_peak    (float32)   pico absoluto por janela
  /windows/data        (float32)   [OPCIONAL] janelas (N, T) – grande em disco

Atributos no root:
  fs, window_sec, stride_sec, source_file

Grupo /meta_in (attrs copiados do arquivo de origem):
  xstart (GPS), xspacing (dt), detector etc. quando disponíveis.

Observações:
- Por padrão NÃO salvamos /windows/data (economia de espaço). O baseline
  (corrigido) sabe reconstruir as janelas usando start_idx + fs + window_sec.
- Se quiser depurar/transportar as janelas sem o arquivo original, ligue
  save_windows=True (cuidado com o tamanho!).
"""

from __future__ import annotations
import os
import math
from typing import Tuple, Optional, Dict, Iterable, List

import h5py
import numpy as np
from tqdm import tqdm


# ---------- Utilidades de leitura robusta do whitened ----------

_WHITENED_CANDIDATES = [
    "/strain/StrainWhitened",
    "/strain/whitened",
    "/data/whitened",
    "/whitened",
    # fallback (se pré-processamento gravou por cima do Strain original)
    "/strain/Strain",
]

def _read_hdf5_attrs_as_dict(obj) -> Dict[str, float]:
    out = {}
    try:
        for k, v in obj.attrs.items():
            # Converte h5py-style para python puro
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()
            out[str(k)] = v
    except Exception:
        pass
    return out

def _find_dataset(f: h5py.File, candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path in f:
            dset = f[path]
            if isinstance(dset, h5py.Dataset):
                return path
    return None

def _load_whitened(in_path: str) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Abre um HDF5 "interim" e busca um dataset 'whitened' por heurísticas.
    Retorna: (x, fs, meta)
      x  : np.ndarray (float32)
      fs : float (Hz)
      meta: dict com xstart (GPS), xspacing (dt), detector etc.
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"arquivo não encontrado: {in_path}")

    with h5py.File(in_path, "r") as f:
        meta = _read_hdf5_attrs_as_dict(f)

        # tenta achar o dataset whitened
        dpath = _find_dataset(f, _WHITENED_CANDIDATES)
        if dpath is None:
            raise KeyError(
                f"não encontrei dataset whitened em {in_path} "
                f"(testei {', '.join(_WHITENED_CANDIDATES)})"
            )
        dset = f[dpath]
        x = dset[()]  # carrega tudo na RAM

        # tenta fs por attrs
        d_attrs = _read_hdf5_attrs_as_dict(dset)
        fs = None
        if "fs" in d_attrs:
            fs = float(d_attrs["fs"])
        elif "Xspacing" in d_attrs:
            fs = 1.0 / float(d_attrs["Xspacing"])
        elif "xspacing" in d_attrs:
            fs = 1.0 / float(d_attrs["xspacing"])

        # tenta também no root/meta
        if fs is None:
            if "fs" in meta:
                fs = float(meta["fs"])
            elif "Xspacing" in meta:
                fs = 1.0 / float(meta["Xspacing"])
            elif "xspacing" in meta:
                fs = 1.0 / float(meta["xspacing"])

        if fs is None:
            # último recurso: infere por tamanho vs duração, mas aqui não sabemos duração;
            # então avisa explicitamente.
            raise ValueError("impossível inferir fs; grave 'fs' ou 'Xspacing' no whitened.")

        # completa meta com atributos do dataset
        meta.update(d_attrs)

        # normaliza chaves que usaremos depois
        if "Xstart" in d_attrs and "xstart" not in meta:
            meta["xstart"] = d_attrs["Xstart"]
        if "Xspacing" in d_attrs and "xspacing" not in meta:
            meta["xspacing"] = d_attrs["Xspacing"]

    x = np.asarray(x, dtype=np.float32)
    return x, float(fs), meta


# ---------- Núcleo de janelamento ----------

def _frame_indices(n: int, fs: float, window_sec: float, stride_sec: float) -> Iterable[Tuple[int, int]]:
    w = int(round(window_sec * fs))
    s = int(round(stride_sec * fs))
    if w <= 0 or s <= 0:
        raise ValueError(f"parâmetros inválidos: window_sec={window_sec}, stride_sec={stride_sec}, fs={fs}")
    if n < w:
        return
    for start in range(0, n - w + 1, s):
        yield start, start + w

def _snr_rms(win: np.ndarray) -> float:
    # em whitened ideal, ruído ~N(0,1) => RMS alto sugere evento/transiente
    return float(np.sqrt(float(np.mean(win * win)) + 1e-12))

def _snr_peak(win: np.ndarray) -> float:
    return float(np.max(np.abs(win)))

def process_whitened_file(
    in_path: str,
    out_dir: str,
    window_sec: float = 2.0,
    stride_sec: float = 0.5,
    snr_threshold: Optional[float] = None,
    save_windows: bool = False,
    compress_level: int = 4,
) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    x, fs, meta = _load_whitened(in_path)
    n = len(x)

    base = os.path.basename(in_path).replace("_whitened.hdf5", "")
    out_name = f"{base}_windows.hdf5"
    out_path = os.path.join(out_dir, out_name)

    # NEW: source_path relativo (do processed -> interim)
    # Se processed e interim estão no mesmo nível de data/, usar caminho relativo:
    try:
        rel_source = os.path.relpath(in_path, start=os.path.dirname(out_path))
    except Exception:
        rel_source = os.path.basename(in_path)

    starts, snr_rms_list, snr_peak_list = [], [], []
    selected_windows = []

    # tempo GPS (aprox) por índice, se disponível:
    gps0 = float(meta.get("xstart", 0.0))
    dt   = float(meta.get("xspacing", 1.0 / fs))

    for i0, i1 in tqdm(_frame_indices(n, fs, window_sec, stride_sec),
                       desc=f"Janelando {base}", unit="win"):
        win = x[i0:i1]
        r = _snr_rms(win)
        p = _snr_peak(win)

        if snr_threshold is not None and max(r, p) < snr_threshold:
            continue

        starts.append(i0)
        snr_rms_list.append(r)
        snr_peak_list.append(p)
        if save_windows:
            selected_windows.append(win.copy())

    with h5py.File(out_path, "w") as f:
        # Atributos de alto nível (redundantes, mas úteis)
        f.attrs["fs"] = float(fs)
        f.attrs["window_sec"] = float(window_sec)
        f.attrs["stride_sec"] = float(stride_sec)
        f.attrs["source_file"] = os.path.basename(in_path)

        # NEW: meta_in SEMPRE presente, com ponteiros mínimos
        m = f.create_group("meta_in")
        m.attrs["source_path"] = rel_source  # <<< ESSENCIAL
        # propague o que tiver do whitened
        if "xstart" in meta:   m.attrs["xstart"]   = float(meta["xstart"])
        if "xspacing" in meta: m.attrs["xspacing"] = float(meta["xspacing"])
        # redundância útil:
        m.attrs["fs"] = float(fs)

        # datasets de janelas
        if starts:
            starts = np.asarray(starts, dtype=np.int64)
            snr_rms_arr  = np.asarray(snr_rms_list, dtype=np.float32)
            snr_peak_arr = np.asarray(snr_peak_list, dtype=np.float32)
            start_gps    = gps0 + starts * dt

            g = f.create_group("windows")
            g.create_dataset("start_idx", data=starts, dtype="i8", compression="gzip", compression_opts=compress_level)
            g.create_dataset("start_gps", data=start_gps.astype(np.float64), dtype="f8", compression="gzip", compression_opts=compress_level)
            g.create_dataset("snr_rms",   data=snr_rms_arr, dtype="f4", compression="gzip", compression_opts=compress_level)
            g.create_dataset("snr_peak",  data=snr_peak_arr, dtype="f4", compression="gzip", compression_opts=compress_level)

            if save_windows and selected_windows:
                W = np.vstack(selected_windows).astype(np.float32)
                g.create_dataset("data", data=W, dtype="f4", compression="gzip", compression_opts=compress_level)
        # se não houver starts, ainda assim deixamos meta_in gravado

    return out_path



# ---------- Helper opcional para processar diretórios ----------

def process_dir(
    in_dir: str,
    out_dir: str,
    window_sec: float = 2.0,
    stride_sec: float = 0.5,
    snr_threshold: Optional[float] = None,
    save_windows: bool = False,
    windows_dtype: str = "float32",
    suffix: str = "_whitened.hdf5",
) -> List[str]:
    """
    Percorre 'in_dir' procurando arquivos *whitened* (por sufixo) e gera windows.
    Retorna a lista de arquivos HDF5 de saída criados/atualizados.
    """
    outs: List[str] = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(suffix):
            continue
        in_path = os.path.join(in_dir, name)
        out_path = process_whitened_file(
            in_path=in_path,
            out_dir=out_dir,
            window_sec=window_sec,
            stride_sec=stride_sec,
            snr_threshold=snr_threshold,
            save_windows=save_windows,
            windows_dtype=windows_dtype,
        )
        if out_path:
            outs.append(out_path)
    return outs


# ---------- Execução direta (sem argumentos CLI) ----------

if __name__ == "__main__":
    # Execução padrão (sem CLI): processa tudo de data/interim -> data/processed
    IN_DIR = os.getenv("GW_IN_INTERIM", "data/interim")
    OUT_DIR = os.getenv("GW_OUT_PROCESSED", "data/processed")

    # parâmetros default; ajuste via env vars se quiser
    WINDOW_SEC = float(os.getenv("GW_WINDOW_SEC", "2.0"))
    STRIDE_SEC = float(os.getenv("GW_STRIDE_SEC", "0.5"))
    SNR_TH = os.getenv("GW_SNR_TH", "")
    SNR_TH = float(SNR_TH) if SNR_TH.strip() != "" else None
    SAVE_W = os.getenv("GW_SAVE_WINDOWS", "false").lower() in {"1", "true", "yes"}
    W_DTYPE = os.getenv("GW_WINDOWS_DTYPE", "float32")

    print(f"[windows] in={IN_DIR} out={OUT_DIR} win={WINDOW_SEC}s stride={STRIDE_SEC}s save_windows={SAVE_W} dtype={W_DTYPE} snr_th={SNR_TH}")
    created = process_dir(
        IN_DIR, OUT_DIR,
        window_sec=WINDOW_SEC,
        stride_sec=STRIDE_SEC,
        snr_threshold=SNR_TH,
        save_windows=SAVE_W,
        windows_dtype=W_DTYPE,
    )
    print(f"[windows] arquivos gerados: {len(created)}")
