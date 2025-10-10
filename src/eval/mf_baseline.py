# src/eval/mf_baseline.py
# --------------------------------------------------------------------------------------
# Baseline de "matched filtering style" via correlação cruzada normalizada (NCC) em
# todos os lags (FFT) contra um banco IMRPhenomD (pycbc), reamostrado para fs alvo.
#
# Lê apenas metadados de janelas em data/processed/*_windows.hdf5 e extrai os trechos
# diretamente dos arquivos *_whitened.hdf5 (via meta_in.source_path corrigido).
#
# Saída: reports/mf_baseline/YYYYmmdd-HHMMSS/ com CSVs, PNGs e summary.json.
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

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from scipy.signal import resample_poly, fftconvolve, get_window
from pycbc.waveform import get_td_waveform

# -----------------------------------
# CONFIG
# -----------------------------------
@dataclass
class CFG:
    # dataset
    PARQUET_PATH: str = "data/processed/dataset.parquet"
    WINDOWS_DIR: str   = "data/processed"
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

    # threshold
    MODE: str = "target_far"   # "target_far" ou "best_f1"
    TARGET_FAR: float = 1e-6

    # relatórios
    OUT_DIR_ROOT: str = "reports/mf_baseline"
    CACHE_FILE: str   = "templates_cache.npz"

    # debug / performance
    MAX_VAL_ROWS: Optional[int] = None
    MAX_TEST_ROWS: Optional[int] = None

    # feedback de progresso
    HEARTBEAT_EVERY_N: int = 2000      # imprime batimento a cada N janelas
    HEARTBEAT_EVERY_SEC: float = 10.0  # ou a cada S segundos, o que vier primeiro
    TQDM_MIN_INTERVAL: float = 0.5     # suaviza barra

CFG = CFG()

# -----------------------------------
# Utils de I/O HDF5 e resolução de caminhos
# -----------------------------------
def read_attrs(obj) -> Dict[str, float | str]:
    out = {}
    try:
        for k, v in obj.attrs.items():
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
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

# -----------------------------------
# Templates IMRPhenomD
# -----------------------------------
def tukey_window(N: int, alpha: float = 0.1) -> np.ndarray:
    try:
        return get_window(("tukey", alpha), N, fftbins=False).astype(np.float32)
    except Exception:
        return get_window("hann", N, fftbins=False).astype(np.float32)

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
    side = max(1, side)
    m1s = np.linspace(CFG.M1_RANGE[0], CFG.M1_RANGE[1], side)
    m2s = np.linspace(CFG.M2_RANGE[0], CFG.M2_RANGE[1], side)
    bank = []
    for m1 in m1s:
        for m2 in m2s:
            bank.append((float(m1), float(m2), CFG.SPIN1, CFG.SPIN2))
            if len(bank) >= n:
                return bank
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
# Leitura de uma janela (via whitened)
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

        with h5py.File(win_file, "r") as f2:
            starts = f2["/windows/start_idx"][()]
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
# NCC via FFT (máximo nos lags)
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
        x = load_window_slice(p, start_gps)
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
# Self-test (injeção)
# -----------------------------------
def self_test_ncc(tmpl_cache: Dict[str, np.ndarray]) -> float:
    if not tmpl_cache:
        return 0.0
    key = next(iter(tmpl_cache.keys()))
    hp = tmpl_cache[key].copy()
    rng = np.random.default_rng(42)
    x = hp + 0.3 * rng.standard_normal(hp.size).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return float(ncc_fft_max(x, hp))

# -----------------------------------
# Plots
# -----------------------------------
def summarize_and_plot(y_true: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")

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
# Scoring com progress feedback
# -----------------------------------
def score_subset(name: str, df_subset: pd.DataFrame, tmpl_cache: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(df_subset)
    scores = np.zeros(n, np.float32)

    hb_n = CFG.HEARTBEAT_EVERY_N
    hb_s = CFG.HEARTBEAT_EVERY_SEC

    print(f"[3/?] Scoring {name}… (NCC em todos os lags)  |  total={n:,}")
    t0 = time.time()
    last_hb_t = t0
    last_i = 0

    with tqdm(total=n, unit="win", mininterval=CFG.TQDM_MIN_INTERVAL, leave=True) as bar:
        for i, (fid, sgps) in enumerate(zip(df_subset["file_id"].values, df_subset["start_gps"].values), 1):
            scores[i-1] = score_window(fid, float(sgps), tmpl_cache)
            bar.update(1)

            # heartbeat por contagem
            do_hb_n = (i % hb_n == 0)
            # heartbeat por tempo
            now = time.time()
            do_hb_t = (now - last_hb_t) >= hb_s

            if do_hb_n or do_hb_t:
                done = i
                elapsed = now - t0 + 1e-9
                rate = done / elapsed
                remain = n - done
                eta = remain / max(rate, 1e-9)
                print(f"  [hb] {name}: {done:,}/{n:,}  ({done*100.0/n:.1f}%)  "
                      f"{rate:.1f} win/s  ETA {eta/60:.1f} min")
                last_hb_t = now
                last_i = i

    elapsed = time.time() - t0
    print(f"      {name} tempo={elapsed:.2f}s | {n:,} janelas | {n/max(elapsed,1e-9):.1f} win/s")
    return scores

# -----------------------------------
# MAIN
# -----------------------------------
def main():
    run_t0 = time.time()
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(CFG.OUT_DIR_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Dataset
    df = pd.read_parquet(CFG.PARQUET_PATH)
    if "subset" not in df.columns:
        # compat: tentar flags antigas
        if {"is_val","is_test"}.issubset(df.columns):
            df = df.copy()
            conds = np.full(len(df), "train", dtype=object)
            conds[df["is_val"].astype(bool).values] = "val"
            conds[df["is_test"].astype(bool).values] = "test"
            df["subset"] = pd.Categorical(conds, categories=["train","val","test"])
        else:
            raise KeyError("dataset.parquet sem coluna 'subset'. Recrie com dataset_builder.py atualizado.")

    keep = df["subset"].isin(CFG.SUBSETS)
    df = df.loc[keep, ["subset", "file_id", "start_gps", "label"]].copy()

    df_val  = df[df["subset"]=="val"].copy()
    df_test = df[df["subset"]=="test"].copy()
    if CFG.MAX_VAL_ROWS:
        df_val = df_val.head(CFG.MAX_VAL_ROWS).copy()
    if CFG.MAX_TEST_ROWS:
        df_test = df_test.head(CFG.MAX_TEST_ROWS).copy()

    print(f"[1/6] Dataset OK: val={len(df_val):,} | test={len(df_test):,}")

    # 2) Banco de templates + cache (por run)
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
        print(f"      Total templates: {len(bank)} (IMRPhenomD) em cache")
        tmpl_cache = build_template_cache(bank, CFG.FS_TARGET, CFG.WINDOW_SEC)
        np.savez(cache_path,
                 keys=np.array(list(tmpl_cache.keys()), dtype=object),
                 waves=np.array(list(tmpl_cache.values()), dtype=object))
    else:
        print("[2/6] Construindo banco e cache de templates…")
        print(f"      Total templates: {len(tmpl_cache)} (IMRPhenomD) em cache")

    # Self-test
    val_inj = self_test_ncc(tmpl_cache)
    print("\n[DEBUG] Self-test NCC (injeção sintética) ...")
    print(f"  NCC(injection): {val_inj:.6f}  (esperado > ~0.05–0.2)\n")

    # 3) Scoring VAL (com feedback real)
    s_val = score_subset("VAL", df_val, tmpl_cache)

    # 4) Thresholds
    print("\n[4/6] Thresholds no VAL…")
    y_val = df_val["label"].to_numpy(int)
    thr, info = pick_threshold(y_val, s_val)
    df_val_out = df_val.copy(); df_val_out["score"] = s_val
    df_val_out.to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)
    auc_val, ap_val = summarize_and_plot(y_val, s_val, out_dir, prefix="val")
    print(f"      thr={thr:.6g}  mode={info.get('mode','?')}  info={{k: v for k,v in info.items() if k!='mode'}}")

    # 5) Scoring TEST
    print("\n[5/6] Scoring TEST…")
    s_test = score_subset("TEST", df_test, tmpl_cache)
    df_test_out = df_test.copy(); df_test_out["score"] = s_test
    df_test_out.to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)
    y_test = df_test_out["label"].to_numpy(int)
    auc_test, ap_test = summarize_and_plot(y_test, s_test, out_dir, prefix="test")

    # 6) Relatório final
    print("\n[6/6] Salvando relatório…")
    tn, fp, fn, tp = confusion_at_threshold(y_test, s_test, thr)
    summary = {
        "templates": int(len(tmpl_cache)),
        "fs_target": int(CFG.FS_TARGET),
        "window_sec": float(CFG.WINDOW_SEC),
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
    print(f" Tempo total: {time.time()-run_t0:.3f} s")

if __name__ == "__main__":
    main()