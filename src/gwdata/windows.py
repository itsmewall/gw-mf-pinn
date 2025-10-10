"""
windows.py
----------
Gera janelas deslizantes a partir dos arquivos whitened de data/interim
e calcula métricas simples de SNR por janela.

Saídas:
 - HDF5 em data/processed com:
     /windows/data         (opcional, janelas completas; pode ser grande)
     /windows/snr_rms      (SNR~RMS por janela)
     /windows/snr_peak     (pico absoluto por janela)
     /windows/start_idx    (índice inicial da janela no sinal whitened)
     /windows/start_gps    (tempo GPS aproximado por janela)
   + atributos com fs, window_sec, stride_sec

Observação:
 - SNR aqui é um "proxy" simples (RMS/peak) no sinal whitened. Para SNR
   ótimo (matched filter), integraremos templates depois.
"""

from __future__ import annotations
import os
import math
import h5py
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm


def _load_whitened(in_path: str) -> Tuple[np.ndarray, float, dict]:
    with h5py.File(in_path, "r") as f:
        x = f["strain"]["StrainWhitened"][()]  # numpy array
        fs = float(f["strain"]["StrainWhitened"].attrs["fs"])
        meta = {}
        if "meta" in f:
            for k, v in f["meta"].attrs.items():
                meta[k] = v
    return x.astype(np.float32, copy=False), fs, meta


def _frame_indices(n: int, fs: float, window_sec: float, stride_sec: float):
    w = int(round(window_sec * fs))
    s = int(round(stride_sec * fs))
    if w <= 0 or s <= 0:
        raise ValueError("window/stride inválidos")
    for start in range(0, n - w + 1, s):
        yield start, start + w


def _snr_rms(win: np.ndarray) -> float:
    # proxy SNR por RMS do whitened (ruído ~N(0,1) após whitening ideal)
    # RMS(alta) sugere presença de evento; normaliza implicitamente
    return float(np.sqrt(np.mean(win * win) + 1e-12))

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
    """
    Gera janelas e salva métricas SNR. Se snr_threshold for definido, salva
    somente janelas acima do limiar (tanto nas métricas quanto nos dados).
    """
    os.makedirs(out_dir, exist_ok=True)
    x, fs, meta = _load_whitened(in_path)
    n = len(x)
    base = os.path.basename(in_path).replace("_whitened.hdf5", "")
    out_name = f"{base}_windows.hdf5"
    out_path = os.path.join(out_dir, out_name)

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

        if snr_threshold is not None:
            if max(r, p) < snr_threshold:
                continue  # descarta janela fraca

        starts.append(i0)
        snr_rms_list.append(r)
        snr_peak_list.append(p)
        if save_windows:
            selected_windows.append(win.copy())

    if not starts:
        # nada selecionado; ainda escrevemos um arquivo "vazio" com meta
        with h5py.File(out_path, "w") as f:
            f.attrs["fs"] = fs
            f.attrs["window_sec"] = window_sec
            f.attrs["stride_sec"] = stride_sec
            f.attrs["source_file"] = os.path.basename(in_path)
            if meta:
                m = f.create_group("meta_in")
                for k, v in meta.items():
                    m.attrs[k] = v
        return out_path

    starts = np.asarray(starts, dtype=np.int64)
    snr_rms_arr = np.asarray(snr_rms_list, dtype=np.float32)
    snr_peak_arr = np.asarray(snr_peak_list, dtype=np.float32)
    start_gps = gps0 + starts * dt

    with h5py.File(out_path, "w") as f:
        f.attrs["fs"] = fs
        f.attrs["window_sec"] = window_sec
        f.attrs["stride_sec"] = stride_sec
        f.attrs["source_file"] = os.path.basename(in_path)

        if meta:
            m = f.create_group("meta_in")
            for k, v in meta.items():
                m.attrs[k] = v

        g = f.create_group("windows")
        g.create_dataset("start_idx", data=starts, dtype="i8", compression="gzip", compression_opts=compress_level)
        g.create_dataset("start_gps", data=start_gps.astype(np.float64), dtype="f8", compression="gzip", compression_opts=compress_level)
        g.create_dataset("snr_rms", data=snr_rms_arr, dtype="f4", compression="gzip", compression_opts=compress_level)
        g.create_dataset("snr_peak", data=snr_peak_arr, dtype="f4", compression="gzip", compression_opts=compress_level)

        if save_windows:
            # (num, T) pode ficar grande — use com parcimônia
            W = np.vstack(selected_windows).astype(np.float32)
            g.create_dataset("data", data=W, dtype="f4", compression="gzip", compression_opts=compress_level)

    return out_path
