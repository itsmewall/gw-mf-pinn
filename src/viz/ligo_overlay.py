# src/viz/ligo_overlay.py
# -----------------------------------------------------------------------------------
# Overlays H1/L1 no estilo LIGO
# -----------------------------------------------------------------------------------

from __future__ import annotations
import os, glob, time
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Raiz do projeto
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS_MF     = os.path.join(ROOT, "reports", "mf_baseline")
OUT_ROOT       = os.path.join(ROOT, "results", "overlays")

# Utilitários do MF
from eval.mf_baseline import (
    CFG as MFCFG,
    build_windows_index,
    cached_start_arrays,
    resolve_whitened_path,
    build_template_bank,
    synth_template,
    load_window_slice,
)

# ------------------------------
# Helpers
# ------------------------------
def _latest_dir(base: str, pattern: str="*") -> Optional[str]:
    cand = sorted(glob.glob(os.path.join(base, pattern)))
    return cand[-1] if cand else None

def _load_dataset_test() -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, "dataset.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset.parquet não encontrado em {path}")
    df = pd.read_parquet(path)
    if "split" not in df.columns and "subset" in df.columns:
        df = df.rename(columns={"subset": "split"})
    need = {"file_id","start_gps","label","split"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"dataset.parquet sem colunas: {miss}")
    return df[df["split"] == "test"].copy()

def _ensure_np_float(arr) -> np.ndarray:
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.number):
        a = a.astype(np.float64, copy=False)
    return a

def _standardize_len(w: np.ndarray, L: int) -> np.ndarray:
    """Trunca ou zero-pad para L."""
    w = np.asarray(w, dtype=np.float32)
    if w.size == L:
        return w
    if w.size > L:
        return w[:L].copy()
    out = np.zeros(L, dtype=np.float32)
    out[:w.size] = w
    return out

def _maybe_load_template_cache(wlen: int) -> List[np.ndarray]:
    latest = _latest_dir(REPORTS_MF)
    waves: List[np.ndarray] = []
    if latest:
        npz = os.path.join(latest, "templates_cache.npz")
        if os.path.exists(npz):
            try:
                with np.load(npz, allow_pickle=True) as z:
                    # Podem vir como object; normaliza para float32 e comprimento wlen
                    if "waves" in z.files:
                        raw = list(z["waves"])
                        for w in raw:
                            w = _ensure_np_float(w).astype(np.float32, copy=False)
                            waves.append(_standardize_len(w, wlen))
                        if len(waves) > 0:
                            return waves
            except Exception:
                pass
    # Fallback: sintetiza um banco pequeno
    bank = build_template_bank(min(256, MFCFG.TEMPLATES_N))
    fs   = MFCFG.FS_TARGET
    for (m1, m2, s1, s2) in bank:
        w = synth_template(m1, m2, s1, s2, fs, wlen, MFCFG.F_LO, MFCFG.HP_WINDOW_TAPER)
        waves.append(_standardize_len(w, wlen))
    return waves

def _shift(arr: np.ndarray, lag: int) -> np.ndarray:
    y = np.zeros_like(arr)
    if lag >= 0:
        if lag < arr.size:
            y[lag:] = arr[:arr.size - lag]
    else:
        k = -lag
        if k < arr.size:
            y[:arr.size - k] = arr[k:]
    return y

def _best_template_and_lag(x: np.ndarray, templates: List[np.ndarray],
                           fs: int, max_shift_sec: float, lag_step: int) -> Tuple[int, int, float]:
    # FFT em float64 para robustez
    x = _ensure_np_float(x).astype(np.float64, copy=False)
    L = x.size
    n_full = 2 * L - 1
    max_shift = max(0, int(round(max_shift_sec * fs)))
    center = n_full // 2
    lag_step = max(1, int(lag_step))

    # Normaliza x
    xz = x - x.mean()
    x_std = xz.std()
    xz = xz / (x_std + 1e-12)

    X = np.fft.fft(xz, n_full)

    lo = max(0, center - max_shift)
    hi = min(n_full, center + max_shift + 1)

    best = -1.0
    best_k, best_lag, best_scale = 0, 0, 1.0

    for k, h in enumerate(templates):
        h = _ensure_np_float(h).astype(np.float64, copy=False)
        # força comprimento correto
        if h.size != L:
            h = _standardize_len(h, L).astype(np.float64, copy=False)

        H = np.fft.fft(h[::-1], n_full)
        c = np.fft.ifft(X * np.conj(H)).real

        denom = (np.linalg.norm(xz) * np.linalg.norm(h)) + 1e-12
        c = c / denom

        seg = c[lo:hi:lag_step]
        j = int(np.argmax(seg))
        val = float(seg[j])
        if val > best:
            best = val
            lag_rel = lo + j - center
            best_k, best_lag = k, int(lag_rel)

            # escala LS com o template alinhado
            if best_lag >= 0:
                h_al = np.zeros_like(xz); h_al[best_lag:] = h[:xz.size-best_lag]
            else:
                h_al = np.zeros_like(xz); h_al[:xz.size+best_lag] = h[-best_lag:]
            num = float(np.dot(xz, h_al))
            den = float(np.dot(h_al, h_al)) + 1e-12
            best_scale = num / den

    return best_k, best_lag, best_scale

def _plot_overlay(fig_path: str,
                  x_H: np.ndarray, x_L: np.ndarray,
                  templates: List[np.ndarray],
                  fs: int, max_shift_sec: float, lag_step: int,
                  t0_sec: float = 0.0):
    # Normalização leve por série
    x_H = _ensure_np_float(x_H).astype(np.float64, copy=False)
    x_L = _ensure_np_float(x_L).astype(np.float64, copy=False)
    x_H = (x_H - x_H.mean()) / (x_H.std() + 1e-12)
    x_L = (x_L - x_L.mean()) / (x_L.std() + 1e-12)

    # Garante templates com mesmo L
    L = x_H.size
    templates = [_standardize_len(t, L) for t in templates]

    kH, lagH, aH = _best_template_and_lag(x_H, templates, fs, max_shift_sec, lag_step)
    kL, lagL, aL = _best_template_and_lag(x_L, templates, fs, max_shift_sec, lag_step)

    hH_al = _shift(templates[kH].astype(np.float64) * aH, lagH)
    hL_al = _shift(templates[kL].astype(np.float64) * aL, lagL)

    t = t0_sec + np.arange(L) / fs

    plt.figure(figsize=(6.8, 8), facecolor="black")

    def style(ax):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        ax.set_ylabel("Strain (1e-21)", color="white")

    # H1
    ax1 = plt.subplot(3,1,1)
    style(ax1)
    ax1.plot(t, x_H, color="#F39C12", lw=2.0, label="LIGO Hanford Data")
    ax1.plot(t, hH_al, color="#F4D03F", lw=1.5, alpha=0.9, label="Predicted")
    ax1.legend(loc="upper left", frameon=False, labelcolor="white")

    # L1
    ax2 = plt.subplot(3,1,2)
    style(ax2)
    ax2.plot(t, x_L, color="#5DADE2", lw=2.0, label="LIGO Livingston Data")
    ax2.plot(t, hL_al, color="#AED6F1", lw=1.5, alpha=0.9, label="Predicted")
    ax2.legend(loc="upper left", frameon=False, labelcolor="white")

    # Coincidência
    ax3 = plt.subplot(3,1,3)
    style(ax3)
    x_H_shift = _shift(x_H, lagL - lagH)
    ax3.plot(t, x_H_shift, color="#F39C12", lw=1.8, label="Hanford shifted")
    ax3.plot(t, x_L,       color="#5DADE2", lw=1.8, label="Livingston")
    ax3.set_xlabel("Time (sec)", color="white")
    ax3.legend(loc="upper left", frameon=False, labelcolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=160, facecolor="black", edgecolor="black")
    plt.close()

def _is_H(fid: str) -> bool:
    s = str(fid)
    return s.startswith("H-") or "H1" in s

def _is_L(fid: str) -> bool:
    s = str(fid)
    return s.startswith("L-") or "L1" in s

def _pick_pair(df: pd.DataFrame) -> Optional[Tuple[int,int]]:
    """Escolhe um par H1/L1 realmente coincidente (mesmo start_gps dentro da tolerância)."""
    def is_H(fid): s = str(fid); return s.startswith("H-") or "H1" in s
    def is_L(fid): s = str(fid); return s.startswith("L-") or "L1" in s

    dfH = df[df["file_id"].apply(is_H)]
    dfL = df[df["file_id"].apply(is_L)]
    if dfH.empty or dfL.empty:
        return None

    tol = 1.0 / MFCFG.FS_TARGET  # 1 amostra de tolerância
    for iH, rH in dfH.iterrows():
        near = dfL[np.abs(dfL["start_gps"] - float(rH["start_gps"])) <= tol]
        if not near.empty:
            return int(iH), int(near.index[0])

    # Sem coincidência? Aborta em vez de pegar “os primeiros”.
    return None

def main():
    t0 = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_dir = os.path.join(OUT_ROOT, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs(out_dir, exist_ok=True)

    print("[overlay] carregando dataset test …")
    df = _load_dataset_test()

    print("[overlay] indexando windows …")
    idx_map = build_windows_index(os.path.join(ROOT, "data", "processed"))
    if not idx_map:
        raise RuntimeError("nenhum *_windows.hdf5 encontrado.")

    # define tamanho de janela em amostras a partir do CFG
    wlen = int(round(MFCFG.WINDOW_SEC * MFCFG.FS_TARGET))
    if wlen <= 0:
        raise ValueError("WINDOW_SEC * FS_TARGET inválido")

    print("[overlay] carregando template cache …")
    templates = _maybe_load_template_cache(wlen)
    if len(templates) == 0:
        raise RuntimeError("nenhum template disponível para overlay")

    pair = _pick_pair(df)
    if not pair:
        print("[overlay] não foi possível escolher par H1/L1. Abortando.")
        print(f"[overlay] saída: {out_dir}")
        return
    iH, iL = pair
    rH, rL = df.loc[iH], df.loc[iL]

    fidH, gpsH = rH["file_id"], float(rH["start_gps"])
    fidL, gpsL = rL["file_id"], float(rL["start_gps"])

    sgpsH, _, metaH = cached_start_arrays(idx_map.get(fidH))
    whtH = resolve_whitened_path(idx_map.get(fidH), metaH)

    sgpsL, _, metaL = cached_start_arrays(idx_map.get(fidL))
    whtL = resolve_whitened_path(idx_map.get(fidL), metaL)

    xH = load_window_slice(idx_map.get(fidH), whtH, gpsH)
    xL = load_window_slice(idx_map.get(fidL), whtL, gpsL)
    if xH is None or xL is None:
        print("[overlay] falha ao carregar H ou L.")
        print(f"[overlay] saída: {out_dir}")
        return

    # força comprimento e tipo das janelas
    xH = _standardize_len(_ensure_np_float(xH).astype(np.float64, copy=False), wlen)
    xL = _standardize_len(_ensure_np_float(xL).astype(np.float64, copy=False), wlen)

    fig_name = f"overlay_{os.path.basename(str(fidH))}__{os.path.basename(str(fidL))}.png"
    fig_path = os.path.join(out_dir, fig_name)

    _plot_overlay(
        fig_path, xH, xL, templates,
        fs=int(MFCFG.FS_TARGET),
        max_shift_sec=float(MFCFG.MAX_SHIFT_SEC),
        lag_step=int(max(1, MFCFG.LAG_STEP)),
        t0_sec=0.0
    )

    print(f"[overlay] salvo: {fig_path}")
    print(f"[overlay] concluído em {time.time()-t0:.1f}s")
    print(f"[overlay] saída: {out_dir}")

if __name__ == "__main__":
    main()
