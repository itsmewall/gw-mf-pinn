# src/eval/mf_baseline.py
from __future__ import annotations
import os, json, time, math, textwrap
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

try:
    import seaborn as sns
    sns.set_context("talk"); sns.set_style("whitegrid")
except Exception:
    pass

# PyCBC
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.psd import welch

from scipy.signal import resample_poly, fftconvolve
try:
    from scipy.signal.windows import tukey as sp_tukey
    def tukey(N: int, alpha: float = 0.1) -> np.ndarray:
        return sp_tukey(N, alpha=alpha)
except Exception:
    def tukey(N: int, alpha: float = 0.1) -> np.ndarray:
        if N <= 1: return np.ones(N, dtype=float)
        return np.hanning(N).astype(float)

import h5py
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix
)

# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Cfg:
    DATASET: str = os.path.join("data", "processed", "dataset.parquet")
    OUT_DIR: str = os.path.join("reports", "mf_baseline")

    SEARCH_DIRS: Tuple[str, ...] = (
        os.path.join("data", "processed", "windows"),
        os.path.join("data", "processed"),
        os.path.join("data", "interim"),
        os.path.join("data", "raw"),
    )

    APPROXIMANT: str = "IMRPhenomD"
    M1_RANGE: Tuple[float, float] = (5.0, 80.0)
    M2_RANGE: Tuple[float, float] = (5.0, 80.0)
    N_M1: int = 16
    N_M2: int = 16
    SPINZ1: float = 0.0
    SPINZ2: float = 0.0
    F_LOW: float = 30.0

    FS_TARGET: int = 4096
    TAPER_TEMPLATE: bool = True
    TAPER_ALPHA: float = 0.1
    MAX_TEMPLATES: Optional[int] = None

    WELCH_SEG_SEC: float = 1.0

    THRESH_STRATEGY: str = "target_far"   # "best_f1" | "target_far"
    FAR_GRID: Tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05)
    TARGET_FAR: float = 0.01
    EPS_THR: float = 1e-6

    N_JOBS: int = max(os.cpu_count() - 1, 1)
    SEED: int = 42
    VERBOSE: bool = True

CFG = Cfg()

DEBUG = True
DEBUG_N_WINDOWS = 3
DEBUG_N_TEMPLATES = 3
_DBG_FIRST_ERR: List[str] = []
_H5_PRINTED: set[str] = set()

# =========================
# Utils
# =========================
def _log(msg: str):
    if CFG.VERBOSE: print(msg)

def ts_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resolve_path(fname: str) -> str:
    if os.path.isabs(fname) and os.path.exists(fname):
        return fname
    if os.path.exists(fname):
        return fname
    for base in CFG.SEARCH_DIRS:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    base_name = os.path.basename(fname)
    for base in CFG.SEARCH_DIRS:
        p = os.path.join(base, base_name)
        if os.path.exists(p):
            return p
    return fname

def describe_arr(name, a: np.ndarray) -> str:
    a = np.asarray(a, dtype=float)
    return (f"{name}: len={len(a)} min={np.min(a):.3g} max={np.max(a):.3g} "
            f"mean={np.mean(a):.3g} std={np.std(a):.3g} nz={(a!=0).sum()} "
            f"nan={np.isnan(a).sum()}")

# =========================
# Dataset
# =========================
def load_dataset() -> pd.DataFrame:
    assert os.path.exists(CFG.DATASET), f"Dataset não encontrado: {CFG.DATASET}"
    df = pd.read_parquet(CFG.DATASET)
    need = {"file_id","split","label","start_gps","window_sec","stride_sec"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"dataset.parquet sem colunas {miss}")
    return df

def split_sets(df: pd.DataFrame):
    return df[df["split"]=="val"].copy(), df[df["split"]=="test"].copy()

# =========================
# Template bank + cache
# =========================
def build_template_bank() -> List[Dict]:
    m1_vals = np.linspace(CFG.M1_RANGE[0], CFG.M1_RANGE[1], CFG.N_M1)
    m2_vals = np.linspace(CFG.M2_RANGE[0], CFG.M2_RANGE[1], CFG.N_M2)
    bank: List[Dict] = []
    for m1 in m1_vals:
        for m2 in m2_vals:
            if m2 > m1:
                m1, m2 = m2, m1
            bank.append({"mass1": float(m1), "mass2": float(m2),
                         "spin1z": CFG.SPINZ1, "spin2z": CFG.SPINZ2})
    if CFG.MAX_TEMPLATES and len(bank) > CFG.MAX_TEMPLATES:
        rng = np.random.default_rng(CFG.SEED)
        idx = rng.choice(len(bank), size=CFG.MAX_TEMPLATES, replace=False)
        bank = [bank[i] for i in idx]
    return bank

def synth_template(params: Dict, delta_t: float) -> np.ndarray:
    hp, _ = get_td_waveform(approximant=CFG.APPROXIMANT,
                            mass1=params["mass1"], mass2=params["mass2"],
                            spin1z=params["spin1z"], spin2z=params["spin2z"],
                            delta_t=delta_t, f_lower=CFG.F_LOW)
    arr = np.asarray(hp.numpy(), dtype=float)
    if CFG.TAPER_TEMPLATE and arr.size > 8:
        win = tukey(arr.size, alpha=CFG.TAPER_ALPHA)
        arr = arr * win
    return arr

def build_template_cache(bank: List[Dict], fs: float) -> Dict[str, np.ndarray]:
    _log(f"[cache] Gerando {len(bank)} templates em {fs:.0f} Hz…")
    cache: Dict[str, np.ndarray] = {}
    dt = 1.0 / fs
    for p in bank:
        k = f"{p['mass1']:.3f}-{p['mass2']:.3f}-{p['spin1z']:.3f}-{p['spin2z']:.3f}"
        arr = synth_template(p, delta_t=dt)
        cache[k] = arr
    _log("[cache] Pronto.")
    return cache

# =========================
# HDF5 helpers
# =========================
def h5_tree_preview(h5: h5py.File, max_lines=60) -> str:
    rows = []
    def visit(name, obj):
        kind = "DSET" if isinstance(obj, h5py.Dataset) else "GRP "
        shape = ""
        if isinstance(obj, h5py.Dataset):
            try: shape = str(obj.shape)
            except Exception: shape = "?"
        rows.append(f"{kind}  /{name}  {shape}")
    h5.visititems(lambda n,o: visit(n,o))
    preview = "\n".join(rows[:max_lines])
    if len(rows) > max_lines:
        preview += f"\n... (+{len(rows)-max_lines} linhas)"
    return preview

def _pick_windows_dset(h5: h5py.File) -> h5py.Dataset:
    """Escolhe recursivamente um dataset 2D que pareça conter as janelas."""
    cands: List[str] = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.shape[1] >= 128:
            low = name.lower()
            score = 0
            if "window" in low: score -= 4
            if "strain" in low: score -= 2
            if "data"   in low: score -= 1
            cands.append((score, name))
    h5.visititems(lambda n,o: visit(n,o))
    if not cands:
        raise RuntimeError("dataset 2D de janelas não encontrado.")
    cands.sort()
    name = cands[0][1]
    return h5[name]

def _find_starts_gps(h5: h5py.File, dset: h5py.Dataset, n: int, fallback_stride: float, fallback_gps0: float) -> np.ndarray:
    # 1) atributo em file ou no dataset
    if "starts_gps" in h5:
        try:
            return np.array(h5["starts_gps"][:], dtype=float)
        except Exception:
            pass
    for obj in (h5, dset):
        try:
            if "starts_gps" in obj.attrs:
                arr = np.array(obj.attrs["starts_gps"], dtype=float)
                if arr.size == n: return arr
        except Exception:
            pass
    # 2) procurar recursivamente datasets com "start" e "gps"
    cands: List[str] = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
            low = name.lower()
            if "start" in low and "gps" in low:
                cands.append(name)
    h5.visititems(lambda n,o: visit(n,o))
    for name in cands:
        try:
            arr = np.array(h5[name][:], dtype=float)
            if arr.size == n: return arr
        except Exception:
            pass
    # 3) fallback: aritmética
    return fallback_gps0 + np.arange(n, dtype=float) * float(fallback_stride)

# =========================
# Load janela (HDF5 / NPZ)
# =========================
def load_window_from_hdf5(path: str, start_gps: float) -> Tuple[np.ndarray,float,str,bool,float]:
    with h5py.File(path, "r") as f:
        # Pega dataset 2D válido
        try:
            dset = _pick_windows_dset(f)
        except Exception as e:
            # imprime árvore uma única vez por arquivo
            base = os.path.basename(path)
            if base not in _H5_PRINTED:
                _H5_PRINTED.add(base)
                print("\n[DEBUG] Estrutura HDF5 (preview):", base)
                print(h5_tree_preview(f))
            raise

        # metadados
        fs = (f.attrs.get("fs") or dset.attrs.get("fs") or
              (1.0/float(dset.attrs.get("dt"))) if "dt" in dset.attrs else CFG.FS_TARGET)
        fs = float(fs)
        det = str(f.attrs.get("detector") or dset.attrs.get("detector") or "H1")
        whitened = bool(f.attrs.get("whitened", True))
        window_sec = float(f.attrs.get("window_sec") or dset.attrs.get("window_sec") or (dset.shape[1]/fs))
        gps0 = float(f.attrs.get("gps_start") or dset.attrs.get("gps_start") or 0.0)
        stride = float(f.attrs.get("stride_sec") or dset.attrs.get("stride_sec") or window_sec)

        n = int(dset.shape[0])
        starts = _find_starts_gps(f, dset, n=n, fallback_stride=stride, fallback_gps0=gps0)
        idx = int(np.argmin(np.abs(starts - float(start_gps))))
        h = np.array(dset[idx, :], dtype=float)

        if DEBUG:
            print(f"[DEBUG] HDF5 pick -> dset=/{dset.name} shape={dset.shape} fs={fs:.0f} det={det} win={window_sec}s")

    return h, fs, det, whitened, window_sec

def load_window_npz(path: str) -> Tuple[np.ndarray,float,str,bool,float]:
    with np.load(path, allow_pickle=False) as npz:
        key = "h" if "h" in npz.files else ("strain" if "strain" in npz.files else "data")
        if key is None:
            raise RuntimeError(f"{path}: sem chave de strain esperada.")
        h = np.array(npz[key], dtype=float)
        fs = float(npz["fs"]) if "fs" in npz.files else CFG.FS_TARGET
        det = str(npz["detector"]) if "detector" in npz.files else "H1"
        whitened = bool(npz["whitened"]) if "whitened" in npz.files else True
        window_sec = float(npz["window_sec"]) if "window_sec" in npz.files else (len(h)/fs)
    return h, fs, det, whitened, window_sec

def load_row_signal(abs_path: str, start_gps: float) -> Tuple[np.ndarray,float,str,float,bool]:
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in (".hdf5", ".h5"):
        h, fs, det, whitened, window_sec = load_window_from_hdf5(abs_path, start_gps)
    else:
        h, fs, det, whitened, window_sec = load_window_npz(abs_path)

    if abs(fs - CFG.FS_TARGET) > 1e-6:
        up = CFG.FS_TARGET; down = int(round(fs))
        g = math.gcd(up, down) if down > 0 else 1
        h = resample_poly(h, up//g, max(down//g,1))
        fs = CFG.FS_TARGET
        window_sec = len(h)/fs

    if not whitened:
        ts = TimeSeries(h, delta_t=1.0/fs)
        psd = welch(ts, seg_len=int(max(1, CFG.WELCH_SEG_SEC * fs)))
        ts = ts.whiten(psd, low_frequency_cutoff=CFG.F_LOW)
        h = ts.numpy()

    return h.astype(float, copy=False), float(fs), det, float(window_sec), bool(whitened)

# =========================
# NCC em todos os lags
# =========================
def normxcorr1d_max(x: np.ndarray, t: np.ndarray) -> float:
    nx, nt = len(x), len(t)
    if nx < 8 or nt < 8:
        return 0.0
    if nt > nx:
        start = (nt - nx) // 2
        t = t[start:start+nx]
        nt = len(t)
        if nt < 8:
            return 0.0

    x0 = x.astype(float, copy=False) - float(np.mean(x))
    t0 = t.astype(float, copy=False) - float(np.mean(t))

    num = fftconvolve(x0, t0[::-1], mode="valid")  # len = nx-nt+1

    xx = x0 * x0
    csum = np.cumsum(np.r_[0.0, xx])
    win_energy = csum[nt:] - csum[:-nt]
    denom_x = np.sqrt(np.maximum(win_energy, 1e-18))
    denom_t = np.sqrt(np.maximum(np.sum(t0*t0), 1e-18))
    r = num / (denom_x * denom_t)

    return float(np.max(np.abs(r)))

def score_window(abs_path: str, start_gps: float, tmpl_cache: Dict[str, np.ndarray]) -> float:
    global _DBG_FIRST_ERR
    try:
        h, fs, _, window_sec, _ = load_row_signal(abs_path, start_gps)
        if len(h) < 16:
            return 0.0
        best = 0.0
        L = len(h)
        for _, t_raw in tmpl_cache.items():
            t = t_raw
            if len(t) > 2 * L:
                t = t[-2*L:]
            m = normxcorr1d_max(h, t)
            if m > best:
                best = m
        return best
    except Exception as e:
        if len(_DBG_FIRST_ERR) < 3:
            _DBG_FIRST_ERR.append(f"{os.path.basename(abs_path)} @ {start_gps}: {e}")
        return 0.0

# =========================
# Thresholds / Plots
# =========================
def find_threshold_best_f1(y_true, y_score):
    thr = np.unique(np.clip(y_score, CFG.EPS_THR, None))
    thr = np.concatenate(([CFG.EPS_THR], thr, [y_score.max() if len(y_score) else 1.0]))
    best = {"thr": CFG.EPS_THR, "f1": -1.0, "precision": 0.0, "recall": 0.0, "tn":0,"fp":0,"fn":0,"tp":0}
    for t in thr:
        yp = (y_score >= t).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-12)
        if f1 > best["f1"]:
            best = {"thr": float(t), "f1": float(f1), "precision": float(prec), "recall": float(rec),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return best

def find_threshold_by_far(y_true, y_score, far: float):
    order = np.argsort(-y_score)
    y = y_true[order]; s = y_score[order]
    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())
    fp = 0; tp = 0
    if len(s) == 0:
        return {"threshold": CFG.EPS_THR, "fpr":0.0, "tpr":0.0, "precision":1.0, "recall":0.0}
    best = {"threshold": float(max(s.max(), CFG.EPS_THR)), "fpr":0.0, "tpr":0.0, "precision":1.0, "recall":0.0}
    for i in range(len(s)):
        thr = s[i]
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / max(n0, 1)
        tpr = tp / max(n1, 1)
        prec = tp / max(tp+fp, 1)
        rec  = tpr
        if fpr <= far:
            best = {"threshold": float(max(thr, CFG.EPS_THR)), "fpr": float(fpr), "tpr": float(tpr),
                    "precision": float(prec), "recall": float(rec)}
            break
    return best

def plot_roc_pr(y_true, y_score, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_ = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={auc_:.4f}")
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {tag}"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"), dpi=160); plt.close()
    except Exception:
        pass
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(rec, prec, label=f"AP={ap:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {tag}"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"pr_{tag}.png"), dpi=160); plt.close()
    except Exception:
        pass

def plot_conf_mat(tn, fp, fn, tp, out_dir: str, tag: str):
    cm = np.array([[tn, fp],[fn, tp]], dtype=int)
    plt.figure(figsize=(4.6,4.2)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="black", fontsize=12)
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    plt.title(f"Confusion — {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cm_{tag}.png"), dpi=160); plt.close()

# =========================
# DEBUG HELPERS
# =========================
def debug_sample_windows(df_part: pd.DataFrame):
    if len(df_part) == 0:
        print("\n[DEBUG] Sem janelas para inspecionar.")
        return
    print("\n[DEBUG] Inspecionando algumas janelas ...")
    sample_df = df_part.sample(min(DEBUG_N_WINDOWS, len(df_part)), random_state=CFG.SEED)
    for r in sample_df.itertuples(index=False):
        try:
            abs_path = resolve_path(r.file_id)
            h, fs, det, win_sec, whit = load_row_signal(abs_path, float(r.start_gps))
            print("  file:", os.path.basename(abs_path), "| label:", r.label)
            print("   ", describe_arr("h", h), "| fs=", fs, "det=", det, "win_sec=", win_sec, "whitened=", whit)
        except Exception as e:
            print("  EXC ao ler janela:", e)

def debug_templates_and_selftest(tmpl_cache: Dict[str, np.ndarray]):
    print("\n[DEBUG] Inspecionando alguns templates ...")
    keys = list(tmpl_cache.keys())[:max(1, DEBUG_N_TEMPLATES)]
    for k in keys:
        t = tmpl_cache[k]
        print("  tmpl:", k, "|", describe_arr("t", t))

    print("\n[DEBUG] Self-test NCC (injeção sintética) ...")
    fs = CFG.FS_TARGET
    L = fs * 4  # 4 s
    rng = np.random.default_rng(CFG.SEED)
    noise = rng.normal(0, 1, int(L)).astype(float)
    t = tmpl_cache[keys[0]]
    if len(t) > len(noise): t = t[:len(noise)]
    x = noise.copy()
    t0 = t - np.mean(t)
    t0 /= max(np.sqrt(np.sum(t0*t0)), 1e-12)
    snr = 8.0
    start = len(x)//2 - len(t0)//2
    x[start:start+len(t0)] += snr * t0
    score = normxcorr1d_max(x, t)
    print(f"  NCC(injection): {score:.6f}  (esperado > ~0.05–0.2)")

# =========================
# Main
# =========================
def main():
    t_all = time.time()
    df = load_dataset()
    val_df, test_df = split_sets(df)

    val_df = val_df.assign(abs_path=val_df["file_id"].apply(resolve_path))
    test_df = test_df.assign(abs_path=test_df["file_id"].apply(resolve_path))

    miss_val = val_df.loc[~val_df["abs_path"].apply(os.path.exists)]
    miss_tst = test_df.loc[~test_df["abs_path"].apply(os.path.exists)]
    if len(miss_val) or len(miss_tst):
        ex = (miss_val["abs_path"].iloc[0] if len(miss_val) else miss_tst["abs_path"].iloc[0])
        raise RuntimeError(f"Arquivos ausentes. Exemplo: {ex}")

    _log(f"[1/6] Dataset OK: val={len(val_df):,} | test={len(test_df):,}")

    if DEBUG:
        debug_sample_windows(val_df)

    _log("[2/6] Construindo banco e cache de templates…")
    bank = build_template_bank()
    tmpl_cache = build_template_cache(bank, fs=CFG.FS_TARGET)
    _log(f"      Total templates: {len(bank)} ({CFG.APPROXIMANT}) em cache")

    if DEBUG:
        debug_templates_and_selftest(tmpl_cache)

    def batch_score(df_part: pd.DataFrame) -> np.ndarray:
        return np.asarray(Parallel(n_jobs=CFG.N_JOBS, batch_size=1, prefer="threads")(
            delayed(score_window)(r.abs_path, float(r.start_gps), tmpl_cache)
            for r in df_part.itertuples(index=False)
        ), dtype=float)

    _log("[3/6] Scoring VAL… (NCC em todos os lags)")
    t0 = time.time()
    val_scores = batch_score(val_df)
    t_val = time.time() - t0
    yv = val_df["label"].to_numpy(np.uint8)
    _log(f"      tempo={t_val:.2f}s | {len(val_scores)} janelas")
    _log(f"      val score stats: min={val_scores.min():.4g} med={np.median(val_scores):.4g} max={val_scores.max():.4g}")
    if _DBG_FIRST_ERR:
        _log("[dbg] exemplos de exceções no score_window:")
        for msg in _DBG_FIRST_ERR: _log("      " + msg)

    _log("[4/6] Thresholds no VAL…")
    best = find_threshold_best_f1(yv, val_scores)
    far_grid = {str(far): find_threshold_by_far(yv, val_scores, far) for far in CFG.FAR_GRID}
    thr_main = best["thr"] if CFG.THRESH_STRATEGY == "best_f1" else far_grid[str(CFG.TARGET_FAR)]["threshold"]

    _log("[5/6] Scoring TEST…")
    t1 = time.time()
    test_scores = batch_score(test_df)
    t_test = time.time() - t1
    yt = test_df["label"].to_numpy(np.uint8)
    yhat = (test_scores >= thr_main).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0,1]).ravel()

    metrics = {
        "templates": {
            "approximant": CFG.APPROXIMANT,
            "m1_range": CFG.M1_RANGE, "m2_range": CFG.M2_RANGE,
            "n_m1": CFG.N_M1, "n_m2": CFG.N_M2,
            "spin1z": CFG.SPINZ1, "spin2z": CFG.SPINZ2,
            "f_low": CFG.F_LOW, "fs_target": CFG.FS_TARGET,
            "n_total": len(bank)
        },
        "timing_sec": {
            "val_scoring": round(t_val, 3),
            "test_scoring": round(t_test, 3),
            "total": round(time.time()-t_all, 3)
        },
        "threshold_strategy": CFG.THRESH_STRATEGY,
        "threshold_used": float(thr_main),
        "val": {
            "roc_auc": float(roc_auc_score(yv, val_scores)) if len(np.unique(yv))>1 else None,
            "pr_ap":  float(average_precision_score(yv, val_scores)) if len(np.unique(yv))>1 else None,
            "best_f1": best,
            "far_grid": far_grid
        },
        "test": {
            "roc_auc": float(roc_auc_score(yt, test_scores)) if len(np.unique(yt))>1 else None,
            "pr_ap":  float(average_precision_score(yt, test_scores)) if len(np.unique(yt))>1 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(tp / max(tp+fp, 1)),
            "recall":    float(tp / max(tp+fn, 1)),
            "fpr":       float(fp / max((test_df['label']==0).sum(), 1))
        }
    }

    out_dir = os.path.join(CFG.OUT_DIR, ts_now()); ensure_dir(out_dir)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({
            "strategy_used": CFG.THRESH_STRATEGY,
            "threshold_used": float(thr_main),
            "best_f1_val": best,
            "far_grid_val": far_grid
        }, f, indent=2)

    plot_roc_pr(yv, val_scores, out_dir, tag="val")
    plot_roc_pr(yt, test_scores, out_dir, tag="test")
    plot_conf_mat(tn, fp, fn, tp, out_dir, tag="test")

    vdump = val_df[["file_id","start_gps","window_sec","stride_sec","label"]].copy()
    vdump["abs_path"] = val_df["abs_path"]; vdump["score"] = val_scores
    vdump.to_parquet(os.path.join(out_dir, "scores_val.parquet"), index=False)

    tdump = test_df[["file_id","start_gps","window_sec","stride_sec","label"]].copy()
    tdump["abs_path"] = test_df["abs_path"]; tdump["score"] = test_scores; tdump["pred"] = yhat
    tdump.to_parquet(os.path.join(out_dir, "scores_test.parquet"), index=False)

    print("\n[OK] Matched Filtering (NCC FFT all-shifts) concluído.")
    print(f" Saída: {out_dir}")
    print(f" Templates: {len(bank)} ({CFG.APPROXIMANT})  | fs_target={CFG.FS_TARGET} Hz")
    print(f" Val AUC: {metrics['val']['roc_auc']}  | Val AP: {metrics['val']['pr_ap']}")
    print(f" Test AUC: {metrics['test']['roc_auc']} | Test AP: {metrics['test']['pr_ap']}")
    print(f" Confusion (test): tn={tn} fp={fp} fn={fn} tp={tp}")
    print(f" Threshold usado: {metrics['threshold_used']:.6f} ({CFG.THRESH_STRATEGY})")
    print(f" Tempo total: {metrics['timing_sec']['total']} s")

if __name__ == "__main__":
    main()
