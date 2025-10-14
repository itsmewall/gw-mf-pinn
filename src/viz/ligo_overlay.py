# src/viz/ligo_overlay.py
# -----------------------------------------------------------------------------------
# Gera figuras no estilo do release do LIGO:
#   1) H1 (laranja) vs template predito alinhado
#   2) L1 (azul) vs template predito alinhado
#   3) H1 deslocado vs L1 (coincidência visual)
# Usa:
#   - dataset.parquet (split=test) para escolher janelas
#   - latest reports/mf_baseline/<tag>/templates_cache.npz (se existir)
#   - funções utilitárias do mf_baseline (leitura de janelas etc.)
# Saída: ./results/overlays/<timestamp>/*.png
# -----------------------------------------------------------------------------------

from __future__ import annotations
import os, glob, time, json
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

# Reuso de utilitários do MF
from eval.mf_baseline import (
    CFG as MFCFG,
    build_windows_index,
    cached_start_arrays,
    resolve_whitened_path,
    _ncc_fft_max_cpu,
    build_template_bank,
    synth_template,
)

# ------------------------------
# Helpers locais
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

def _maybe_load_template_cache() -> Dict[str, np.ndarray]:
    latest = _latest_dir(REPORTS_MF)
    if latest:
        npz = os.path.join(latest, "templates_cache.npz")
        if os.path.exists(npz):
            try:
                with np.load(npz, allow_pickle=True) as z:
                    keys = list(z["keys"]); waves = list(z["waves"])
                return {k: w for k, w in zip(keys, waves)}
            except Exception:
                pass
    # fallback: reconstrói pequeno banco com as configs atuais
    bank = build_template_bank(min(256, MFCFG.TEMPLATES_N))
    wlen = int(round(MFCFG.WINDOW_SEC * MFCFG.FS_TARGET))
    cache = {}
    for (m1, m2, s1, s2) in bank:
        key = f"{m1:.3f}-{m2:.3f}-{s1:.3f}-{s2:.3f}"
        cache[key] = synth_template(m1, m2, s1, s2, MFCFG.FS_TARGET, wlen, MFCFG.F_LO, MFCFG.HP_WINDOW_TAPER)
    return cache

def _shift(arr: np.ndarray, lag: int) -> np.ndarray:
    y = np.zeros_like(arr)
    if lag >= 0:
        y[lag:] = arr[:arr.size - lag]
    else:
        y[:lag] = arr[-lag:]
    return y

def _best_template_and_lag(x: np.ndarray, templates: List[np.ndarray],
                           fs: int, max_shift_sec: float, lag_step: int) -> Tuple[int, int, float]:
    L = x.size
    n_full = 2 * L - 1
    max_shift = int(round(max_shift_sec * fs))
    center = n_full // 2
    lo = max(0, center - max_shift)
    hi = min(n_full, center + max_shift + 1)

    xz = x - x.mean()
    xz = xz / (xz.std() + 1e-12)

    best = -1.0
    best_k, best_lag, best_scale = 0, 0, 1.0
    for k, h in enumerate(templates):
        # correlação normalizada via FFT (CPU)
        c = np.fft.ifft(np.fft.fft(xz, n_full) * np.conj(np.fft.fft(h[::-1], n_full))).real
        denom = (np.linalg.norm(xz) * np.linalg.norm(h)) + 1e-12
        c = c / denom
        seg = c[lo:hi:lag_step]
        j = int(np.argmax(seg))
        val = float(seg[j])
        if val > best:
            best = val
            lag_rel = lo + j - center
            best_k, best_lag = k, int(lag_rel)
            # escala LS
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
    x_H = (x_H - x_H.mean()) / (x_H.std() + 1e-12)
    x_L = (x_L - x_L.mean()) / (x_L.std() + 1e-12)

    kH, lagH, aH = _best_template_and_lag(x_H, templates, fs, max_shift_sec, lag_step)
    kL, lagL, aL = _best_template_and_lag(x_L, templates, fs, max_shift_sec, lag_step)

    hH_al = _shift(templates[kH] * aH, lagH)
    hL_al = _shift(templates[kL] * aL, lagL)

    L = x_H.size
    t = t0_sec + np.arange(L) / fs

    plt.figure(figsize=(6.8, 8), facecolor="black")

    def style(ax):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        ax.set_ylabel("Strain (10$^{-21}$)", color="white")

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

    # Coincidência (H1 deslocado vs L1)
    ax3 = plt.subplot(3,1,3)
    style(ax3)
    x_H_shift = _shift(x_H, lagL - lagH)
    ax3.plot(t, x_H_shift, color="#F39C12", lw=1.8, label="LIGO Hanford Data (shifted)")
    ax3.plot(t, x_L,       color="#5DADE2", lw=1.8, label="LIGO Livingston Data")
    ax3.set_xlabel("Time (sec)", color="white")
    ax3.legend(loc="upper left", frameon=False, labelcolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=160, facecolor="black", edgecolor="black")
    plt.close()

def _pick_pair(df: pd.DataFrame) -> Optional[Tuple[int,int]]:
    """Escolhe um par H1/L1 com mesmo start_gps (ou muito próximo)."""
    def is_H(fid): s = str(fid); return s.startswith("H-") or "H1" in s
    def is_L(fid): s = str(fid); return s.startswith("L-") or "L1" in s
    dfH = df[df["file_id"].apply(is_H)]
    dfL = df[df["file_id"].apply(is_L)]
    if dfH.empty or dfL.empty:
        return None
    tol = 2.0 / MFCFG.FS_TARGET
    # tenta casar pelo mesmo gps
    for iH, rH in dfH.iterrows():
        near = dfL[np.abs(dfL["start_gps"] - float(rH["start_gps"])) <= tol]
        if not near.empty:
            iL = int(near.index[0])
            return int(iH), iL
    # fallback: pega os primeiros
    return int(dfH.index[0]), int(dfL.index[0])

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

    print("[overlay] carregando template cache …")
    cache = _maybe_load_template_cache()
    templates = [cache[k] for k in list(cache.keys())]

    # escolhe um par H1/L1
    pair = _pick_pair(df)
    if not pair:
        print("[overlay] não foi possível escolher par H1/L1. Abortando.")
        print(f"[overlay] saída: {out_dir}")
        return
    iH, iL = pair
    rH, rL = df.loc[iH], df.loc[iL]

    # carrega janelas
    fidH, gpsH = rH["file_id"], float(rH["start_gps"])
    fidL, gpsL = rL["file_id"], float(rL["start_gps"])

    winH = idx_map.get(fidH); sgpsH, _, metaH = cached_start_arrays(winH)
    whtH = resolve_whitened_path(winH, metaH)

    winL = idx_map.get(fidL); sgpsL, _, metaL = cached_start_arrays(winL)
    whtL = resolve_whitened_path(winL, metaL)

    # reusa leitura do mf_baseline via h5py (encapsulada dentro resolve/load)
    from eval.mf_baseline import load_window_slice
    xH = load_window_slice(winH, whtH, gpsH)
    xL = load_window_slice(winL, whtL, gpsL)
    if xH is None or xL is None:
        print("[overlay] falha ao carregar H ou L.")
        print(f"[overlay] saída: {out_dir}")
        return

    # plota
    fig_name = f"overlay_{os.path.basename(str(fidH))}__{os.path.basename(str(fidL))}.png"
    fig_path = os.path.join(out_dir, fig_name)
    _plot_overlay(
        fig_path, xH, xL, templates,
        fs=MFCFG.FS_TARGET,
        max_shift_sec=MFCFG.MAX_SHIFT_SEC,
        lag_step=MFCFG.LAG_STEP,
        t0_sec=0.0
    )

    print(f"[overlay] salvo: {fig_path}")
    print(f"[overlay] concluído em {time.time()-t0:.1f}s")
    print(f"[overlay] saída: {out_dir}")

if __name__ == "__main__":
    main()