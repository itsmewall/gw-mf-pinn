# src/eval/mf_baseline.py
# --------------------------------------------------------------------------------------
# Baseline de "matched filtering" (estilo) via correlação cruzada normalizada (NCC)
# usando um banco simples de templates IMRPhenomD (pycbc) reamostrados para fs alvo.
#
# NÃO depende de /windows/data. Lê apenas os metadados das janelas e puxa o trecho
# bruto a partir do arquivo *_whitened.hdf5 original (via meta_in.source_path).
#
# Saída: reports/mf_baseline/YYYYmmdd-HHMMSS/ com CSVs, PNGs e cache de templates.
# --------------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import time
import math
from glob import glob
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from typing import Dict, Tuple, Optional, List

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from scipy.signal import resample_poly, fftconvolve, get_window
from pycbc.waveform import get_td_waveform

# -----------------------------------
# CONFIG
# -----------------------------------
@dataclass
class CFG:
    PARQUET_PATH: str = "data/processed/dataset.parquet"   # preferido
    CSV_FALLBACK: str = "data/processed/dataset_preview.csv"  # fallback
    WINDOWS_DIR: str   = "data/processed"                  # onde estão *_windows.hdf5

    # quais subsets usar; se 'subset' não existir, criaremos val/test estratificados
    SUBSETS: Tuple[str, ...] = ("val", "test")

    # janela alvo
    WINDOW_SEC: float = 2.0
    FS_TARGET: int    = 4096

    # banco de templates
    TEMPLATES_N: int  = 256
    M1_RANGE: Tuple[float, float] = (5.0, 40.0)
    M2_RANGE: Tuple[float, float] = (5.0, 40.0)
    SPIN1: float = 0.0
    SPIN2: float = 0.0
    F_LO: float  = 20.0
    HP_WINDOW_TAPER: float = 0.1

    # threshold selection
    MODE: str = "target_far"   # "target_far" ou "best_f1"
    TARGET_FAR: float = 1e-6

    # relatórios
    OUT_DIR_ROOT: str = "reports/mf_baseline"
    CACHE_FILE: str   = "templates_cache.npz"

    # debug/limites (None = tudo)
    MAX_VAL_ROWS: Optional[int] = None
    MAX_TEST_ROWS: Optional[int] = None

CFG = CFG()

# -----------------------------------
# Utils
# -----------------------------------
def read_attrs(obj) -> Dict[str, float | str]:
    out = {}
    try:
        for k, v in obj.attrs.items():
            if hasattr(v, "item"):
                try: v = v.item()
                except Exception: pass
            out[str(k)] = v
    except Exception:
        pass
    return out

def resolve_whitened_path(win_path: str, meta: Dict[str, str]) -> Optional[str]:
    src = meta.get("source_path") or meta.get("source_file")
    if src:
        p1 = os.path.join(os.path.dirname(win_path), src)
        if os.path.exists(p1): return p1
        p2 = os.path.join("data/interim", os.path.basename(src))
        if os.path.exists(p2): return p2

    bn = os.path.basename(win_path).replace("_windows.hdf5", "_whitened.hdf5")
    for p in [
        os.path.join(os.path.dirname(win_path), bn),
        os.path.join("data/interim", bn),
        os.path.join("data/processed", bn)
    ]:
        if os.path.exists(p):
            return p

    hits = glob(f"data/**/{bn}", recursive=True)
    if hits:
        return hits[0]
    return None

def find_window_file(file_id: str) -> Optional[str]:
    hits = glob(os.path.join(CFG.WINDOWS_DIR, "**", file_id), recursive=True)
    return hits[0] if hits else None

def tukey_window(N: int, alpha: float = 0.1) -> np.ndarray:
    try:
        return get_window(("tukey", alpha), N, fftbins=False).astype(np.float32)
    except Exception:
        return get_window("hann", N, fftbins=False).astype(np.float32)

# -----------------------------------
# Templates
# -----------------------------------
def synth_template(m1: float, m2: float, spin1: float, spin2: float,
                   fs: int, wlen: int, f_low: float, alpha: float) -> np.ndarray:
    hp, _ = get_td_waveform(approximant="IMRPhenomD",
                            mass1=m1, mass2=m2,
                            spin1z=spin1, spin2z=spin2,
                            delta_t=1.0/fs,
                            f_lower=f_low)
    hp = hp.numpy().astype(np.float32)

    if hp.size > wlen:
        hp = hp[-wlen:]
    elif hp.size < wlen:
        vec = np.zeros(wlen, np.float32)
        vec[-hp.size:] = hp
        hp = vec

    w = tukey_window(wlen, alpha=max(0.0, min(1.0, alpha)))
    hp *= w

    hp = hp - hp.mean()
    std = float(hp.std()) + 1e-12
    hp /= std
    return hp

def build_template_bank(n: int) -> List[Tuple[float, float, float, float]]:
    side = int(math.sqrt(n))
    m1s = np.linspace(CFG.M1_RANGE[0], CFG.M1_RANGE[1], side)
    m2s = np.linspace(CFG.M2_RANGE[0], CFG.M2_RANGE[1], side)
    bank = []
    for m1 in m1s:
        for m2 in m2s:
            if len(bank) >= n: break
            bank.append((float(m1), float(m2), CFG.SPIN1, CFG.SPIN2))
        if len(bank) >= n: break
    return bank

def build_template_cache(bank: List[Tuple[float, float, float, float]],
                         fs: int, window_sec: float) -> Dict[str, np.ndarray]:
    wlen = int(round(window_sec * fs))
    cache = {}
    for (m1, m2, s1, s2) in bank:
        key = f"{m1:.3f}-{m2:.3f}-{s1:.3f}-{s2:.3f}"
        cache[key] = synth_template(m1, m2, s1, s2, fs, wlen, CFG.F_LO, CFG.HP_WINDOW_TAPER)
    return cache

# -----------------------------------
# Leitura da janela a partir do *_whitened
# -----------------------------------
def load_window_slice(win_file: str, start_gps: float) -> Optional[np.ndarray]:
    with h5py.File(win_file, "r") as f:
        if "/windows/start_gps" not in f:
            raise ValueError("arquivo de janelas sem /windows/start_gps")
        arr_gps = f["/windows/start_gps"][()]
        idx_candidates = np.where(np.isclose(arr_gps, start_gps, atol=5e-4))[0]
        if idx_candidates.size == 0:
            return None
        idx0 = int(idx_candidates[0])

        meta_in = read_attrs(f["/meta_in"]) if "/meta_in" in f else {}
        src = resolve_whitened_path(win_file, meta_in)
        if not src or not os.path.exists(src):
            raise FileNotFoundError(f"whitened não encontrado (source_path): {src}")

    with h5py.File(src, "r") as g:
        cand_paths = ["/strain/StrainWhitened", "/strain/whitened", "/data/whitened",
                      "/whitened", "/strain/Strain"]
        dset = None
        for c in cand_paths:
            if c in g and isinstance(g[c], h5py.Dataset):
                dset = g[c]; break
        if dset is None:
            raise ValueError("dataset de whitened não encontrado no HDF5")

        a_g = read_attrs(g) | read_attrs(dset)
        fs_in = a_g.get("fs", None)
        if not fs_in:
            xs = a_g.get("xspacing") or a_g.get("Xspacing")
            if not xs:
                raise ValueError("impossível inferir fs; xspacing ausente")
            fs_in = 1.0 / float(xs)
        fs_in = int(round(float(fs_in)))

        with h5py.File(win_file, "r") as f:
            starts = f["/windows/start_idx"][()]
            i0 = int(starts[idx0])

        wlen_in = int(round(CFG.WINDOW_SEC * fs_in))
        sl = dset[i0:i0 + wlen_in].astype(np.float32, copy=True)

    if fs_in != CFG.FS_TARGET:
        frac = Fraction(CFG.FS_TARGET, fs_in).limit_denominator(64)
        sl = resample_poly(sl, frac.numerator, frac.denominator).astype(np.float32, copy=False)

    wlen_tgt = int(round(CFG.WINDOW_SEC * CFG.FS_TARGET))
    if sl.size > wlen_tgt:
        sl = sl[-wlen_tgt:]
    elif sl.size < wlen_tgt:
        buf = np.zeros(wlen_tgt, np.float32)
        buf[-sl.size:] = sl
        sl = buf

    sl = sl - sl.mean()
    std = float(sl.std()) + 1e-12
    sl /= std
    return sl

# -----------------------------------
# NCC por FFT (máximo nos lags)
# -----------------------------------
def ncc_fft_max(x: np.ndarray, h: np.ndarray) -> float:
    if x.size != h.size:
        w = min(x.size, h.size)
        x = x[-w:]; h = h[-w:]
    c = fftconvolve(x, h[::-1], mode="full")
    denom = (np.linalg.norm(x) * np.linalg.norm(h)) + 1e-12
    c /= denom
    return float(np.max(c))

def score_window(file_id: str, start_gps: float, tmpl_cache: Dict[str, np.ndarray]) -> float:
    try:
        p = find_window_file(file_id)
        if not p:
            return 0.0
        x = load_window_slice(p, float(start_gps))
        if x is None or not np.any(np.isfinite(x)):
            return 0.0
        best = 0.0
        for hp in tmpl_cache.values():
            s = ncc_fft_max(x, hp)
            if s > best: best = s
        return best
    except Exception:
        return 0.0

# -----------------------------------
# Thresholding
# -----------------------------------
def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if CFG.MODE == "target_far":
        neg = scores[y_true == 0]
        if neg.size == 0:
            return 0.0, {"mode": "target_far", "far": CFG.TARGET_FAR}
        q = 1.0 - CFG.TARGET_FAR
        thr = float(np.quantile(neg, q)) if neg.size > 10 else float(np.max(neg))
        return thr, {"mode": "target_far", "far": CFG.TARGET_FAR}

    thr_candidates = np.unique(scores)
    best_f1, best_thr = -1.0, 0.0
    for thr in thr_candidates:
        yhat = (scores >= thr).astype(int)
        tp = int(np.sum((yhat == 1) & (y_true == 1)))
        fp = int(np.sum((yhat == 1) & (y_true == 0)))
        fn = int(np.sum((yhat == 0) & (y_true == 1)))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, {"mode": "best_f1", "f1": best_f1}

# -----------------------------------
# Métricas e plots
# -----------------------------------
def summarize_and_plot(y_true: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(4.5,4.5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--',alpha=.4)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
        plt.legend(); plt.grid(alpha=.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{prefix}.png"), dpi=130)
        plt.close()
    except Exception:
        pass

    # PR
    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(4.5,4.5))
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
        plt.legend(); plt.grid(alpha=.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_{prefix}.png"), dpi=130)
        plt.close()
    except Exception:
        pass

    # hist score por classe
    try:
        plt.figure(figsize=(5.8,3.4))
        plt.hist(scores[y_true==0], bins=60, alpha=.7, label="neg", density=True)
        plt.hist(scores[y_true==1], bins=60, alpha=.7, label="pos", density=True)
        plt.legend(); plt.grid(alpha=.2); plt.title(f"Score dist - {prefix}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{prefix}.png"), dpi=130)
        plt.close()
    except Exception:
        pass

    return auc, ap

def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Tuple[int,int,int,int]:
    yhat = (scores >= thr).astype(int)
    tn = int(np.sum((yhat==0) & (y_true==0)))
    fp = int(np.sum((yhat==1) & (y_true==0)))
    fn = int(np.sum((yhat==0) & (y_true==1)))
    tp = int(np.sum((yhat==1) & (y_true==1)))
    return tn, fp, fn, tp

# -----------------------------------
# Loader robusto do dataset
# -----------------------------------
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # possíveis aliases -> alvo
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}

    # file_id
    for cand in ["file_id", "file", "fid", "hdf5"]:
        if cand in cols:
            rename_map[cols[cand]] = "file_id"; break
    # start_gps
    for cand in ["start_gps", "gps", "tstart", "start"]:
        if cand in cols:
            rename_map[cols[cand]] = "start_gps"; break
    # label
    for cand in ["label", "y", "target", "class"]:
        if cand in cols:
            rename_map[cols[cand]] = "label"; break
    # subset
    for cand in ["subset", "split", "partition", "set"]:
        if cand in cols:
            rename_map[cols[cand]] = "subset"; break

    if rename_map:
        df = df.rename(columns=rename_map)

    # checagens mínimas
    needed = ["file_id", "start_gps", "label"]
    for k in needed:
        if k not in df.columns:
            raise KeyError(f"coluna obrigatória ausente no dataset: {k}")

    return df

def _ensure_subset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if "subset" in df.columns:
        return df
    # criar split estratificado val/test (50/50) por label
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["subset"] = "val"
    for y in df["label"].unique():
        idx = df.index[df["label"] == y].to_numpy()
        rng.shuffle(idx)
        half = len(idx) // 2
        df.loc[idx[half:], "subset"] = "test"
    return df

def load_dataset_frame() -> pd.DataFrame:
    if os.path.exists(CFG.PARQUET_PATH):
        df = pd.read_parquet(CFG.PARQUET_PATH)
    elif os.path.exists(CFG.CSV_FALLBACK):
        df = pd.read_csv(CFG.CSV_FALLBACK)
    else:
        raise FileNotFoundError(
            f"Nem {CFG.PARQUET_PATH} nem {CFG.CSV_FALLBACK} existem."
        )
    df = _standardize_columns(df)
    df = _ensure_subset(df)
    # manter só colunas necessárias
    keep_cols = ["subset", "file_id", "start_gps", "label"]
    extra = [c for c in keep_cols if c not in df.columns]
    if extra:
        raise KeyError(f"faltando colunas após normalização: {extra}")
    return df[keep_cols].copy()

# -----------------------------------
# MAIN
# -----------------------------------
def main():
    t0 = time.time()

    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(CFG.OUT_DIR_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Carregar dataset (robusto)
    df = load_dataset_frame()
    keep = df["subset"].isin(CFG.SUBSETS)
    df = df.loc[keep].copy()

    if CFG.MAX_VAL_ROWS:
        df_val = df[df["subset"]=="val"].head(CFG.MAX_VAL_ROWS).copy()
    else:
        df_val = df[df["subset"]=="val"].copy()

    if CFG.MAX_TEST_ROWS:
        df_test = df[df["subset"]=="test"].head(CFG.MAX_TEST_ROWS).copy()
    else:
        df_test = df[df["subset"]=="test"].copy()

    print(f"[1/6] Dataset OK: val={len(df_val):,} | test={len(df_test):,}")

    # 2) Banco de templates + cache (por execução)
    cache_path = os.path.join(out_dir, CFG.CACHE_FILE)
    bank = build_template_bank(CFG.TEMPLATES_N)
    tmpl_cache = None

    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as npz:
                keys = list(npz["keys"])
                waves = list(npz["waves"])
            tmpl_cache = {k: w for k, w in zip(keys, waves)}
        except Exception:
            tmpl_cache = None

    if tmpl_cache is None:
        print("[2/6] Construindo banco e cache de templates…")
        tmpl_cache = build_template_cache(bank, CFG.FS_TARGET, CFG.WINDOW_SEC)
        np.savez(cache_path, keys=np.array(list(tmpl_cache.keys()), dtype=object),
                 waves=np.array(list(tmpl_cache.values()), dtype=object))
        print(f"      Total templates: {len(tmpl_cache)} (IMRPhenomD) em cache")

    # Self-test rápido
    def self_test_ncc(cache: Dict[str, np.ndarray]) -> float:
        if not cache: return 0.0
        key = next(iter(cache.keys()))
        hp = cache[key].copy()
        rng = np.random.default_rng(42)
        x = hp + 0.3 * rng.standard_normal(hp.size).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-12)
        return float(ncc_fft_max(x, hp))

    val_inj = self_test_ncc(tmpl_cache)
    print("\n[DEBUG] Self-test NCC (injeção sintética) ...")
    print(f"  NCC(injection): {val_inj:.6f}  (esperado > ~0.05–0.2)")

    # 3) Scoring VAL
    print("\n[3/6] Scoring VAL… (NCC em todos os lags)")
    t1 = time.time()
    s_val = np.zeros(len(df_val), np.float32)
    for i, (fid, sgps) in enumerate(zip(df_val["file_id"].values, df_val["start_gps"].values)):
        s_val[i] = score_window(fid, float(sgps), tmpl_cache)
    t2 = time.time()
    print(f"      tempo={t2-t1:.2f}s | {len(df_val):,} janelas")
    print(f"      val score stats: min={s_val.min():.3g} med={np.median(s_val):.3g} max={s_val.max():.3g}")

    # 4) Thresholds pelo VAL
    print("\n[4/6] Thresholds no VAL…")
    y_val = df_val["label"].to_numpy(int)
    thr, info = pick_threshold(y_val, s_val)

    df_val_out = df_val.copy()
    df_val_out["score"] = s_val
    df_val_out.to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)
    auc_val, ap_val = summarize_and_plot(y_val, s_val, out_dir, prefix="val")
    print(f"      thr={thr:.6g}  mode={info.get('mode','?')}")

    # 5) Scoring TEST
    print("\n[5/6] Scoring TEST…")
    s_test = np.zeros(len(df_test), np.float32)
    for i, (fid, sgps) in enumerate(zip(df_test["file_id"].values, df_test["start_gps"].values)):
        s_test[i] = score_window(fid, float(sgps), tmpl_cache)

    df_test_out = df_test.copy()
    df_test_out["score"] = s_test
    df_test_out.to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)
    auc_test, ap_test = summarize_and_plot(df_test_out["label"].to_numpy(int),
                                           df_test_out["score"].to_numpy(float),
                                           out_dir, prefix="test")

    # 6) Relatório final
    print("\n[6/6] Salvando relatório…")
    y_test = df_test_out["label"].to_numpy(int)
    tn, fp, fn, tp = confusion_at_threshold(y_test, s_test, thr)

    summary = {
        "templates": len(tmpl_cache),
        "fs_target": CFG.FS_TARGET,
        "window_sec": CFG.WINDOW_SEC,
        "auc_val": float(auc_val),
        "ap_val": float(ap_val),
        "auc_test": float(auc_test),
        "ap_test": float(ap_test),
        "threshold": float(thr),
        "threshold_mode": CFG.MODE,
        "confusion_test": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "self_test_ncc": float(val_inj),
        "n_val": int(len(df_val_out)),
        "n_test": int(len(df_test_out)),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Matched Filtering (NCC FFT all-shifts) concluído.")
    print(f" Saída: {out_dir}")
    print(f" Templates: {len(tmpl_cache)} (IMRPhenomD)  | fs_target={CFG.FS_TARGET} Hz")
    print(f" Val AUC: {auc_val:.3g}  | Val AP: {ap_val}")
    print(f" Test AUC: {auc_test:.3g} | Test AP: {ap_test}")
    print(f" Confusion (test): tn={tn} fp={fp} fn={fn} tp={tp}")
    print(f" Threshold usado: {thr:.6g} ({CFG.MODE})")
    print(f" Tempo total: {time.time()-t0:.3f} s")

if __name__ == "__main__":
    main()