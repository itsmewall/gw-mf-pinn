# src/eval/mf_baseline.py
# --------------------------------------------------------------------------------------
# Matched-filter baseline via NCC
# Recursos:
#   - GPU via CuPy com fallback para CPU
#   - CPU: dois modos
#       1) Nativo C++ OpenMP/pybind (eval.accel.ncc_fft) para lags curtos
#       2) FFT NumPy como legado
#   - Autotuning de batch e chunk no GPU com backoff em OOM
#   - Index de windows por file_id e caches
#   - Restricao de lag (±MAX_SHIFT_SEC) e passo LAG_STEP
#   - Banco de templates IMRPhenomD com cache .npz
#   - Self-test de injeção rápida
#   - Pooling top-k entre templates
#   - Três modos de threshold: constrained_roc, target_far, best_f1
#   - Relatórios ROC, PR, histogramas, CSV de scores e thresholds.json
#   - N1: Varredura sistemática com VAL pequeno em FAR mais grosso e RECHECK com muitos negativos
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os, sys, json, time, math, warnings, pathlib
from glob import glob
from dataclasses import dataclass, asdict
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from typing import Dict, Tuple, Optional, List
from collections import Counter
from contextlib import contextmanager
import itertools

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed

from scipy.signal import resample_poly, get_window, fftconvolve as np_fftconvolve
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from pycbc.waveform import get_td_waveform

# =========================
# GPU autodetect
# =========================
_HAS_CUPY = False
_CUPY_ERR = None
try:
    import cupy as cp
    try:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        _ = cp.fft.rfft(cp.zeros(8, dtype=cp.float32))
        _HAS_CUPY = (n_gpus > 0)
    except Exception as e:
        _CUPY_ERR = f"cuFFT indisponivel: {e}"
        _HAS_CUPY = False
except Exception as e:
    _CUPY_ERR = f"CuPy indisponivel: {e}"
    _HAS_CUPY = False

# =========================
# Nativo C++ via accel.ncc_fft
# =========================
_HAS_NATIVE = False
_NATIVE_ERR = None
try:
    sys.path.insert(0, str(pathlib.Path("src").resolve()))
    # tente primeiro no pacote com namespace completo
    try:
        from eval.accel import ncc_fft as ncc_native
    except Exception:
        from accel import ncc_fft as ncc_native
    if hasattr(ncc_native, "correlate_all_lags"):
        _HAS_NATIVE = True
except Exception as e:
    _NATIVE_ERR = f"{type(e).__name__}: {e}"
    _HAS_NATIVE = False

# =========================
# Depuracao global
# =========================
DEBUG_STRICT = True
FAIL_COUNTER: Counter = Counter()
FAIL_ROWS: List[dict] = []

# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    # dataset
    PARQUET_PATH: str = "data/processed/dataset.parquet"
    WINDOWS_DIR: str   = "data/processed"
    SUBSETS: Tuple[str, ...] = ("val", "test")

    # janela e amostragem
    WINDOW_SEC: float = 2.0
    FS_TARGET: int    = 4096

    # GPU
    USE_GPU: bool = True
    GPU_INIT_BATCH_WIN: int = 8192
    GPU_MIN_BATCH_WIN: int  = 256
    GPU_INIT_TEMPL_CHUNK: int = 128
    GPU_MIN_TEMPL_CHUNK: int = 8
    USE_FP16_TIME: bool = False

    # CPU nativo
    USE_NATIVE_NCC: bool = True
    NATIVE_MAXSHIFT_HEURISTIC: int = 128  # usa nativo quando max_shift_amostras <= isso

    # banco de templates
    TEMPLATES_N: int = 1024
    TEMPLATE_SUBSAMPLE: int = 2
    M1_RANGE: Tuple[float, float] = (3.0, 80.0)
    M2_RANGE: Tuple[float, float] = (3.0, 80.0)
    SPIN1: float = 0.0
    SPIN2: float = 0.0
    F_LO: float = 20.0
    HP_WINDOW_TAPER: float = 0.1
    CACHE_FILE: str = "templates_cache.npz"

    # lags
    MAX_SHIFT_SEC: float = 0.02
    LAG_STEP: int = 1  # importante: 1

    # veto de glitch
    ENABLE_GLITCH_VETO: bool = False
    GLITCH_VETO_PEAK: float = 20.0
    GLITCH_VETO_KURT: float = 80.0

    # limites de linhas
    MAX_VAL_ROWS: Optional[int] = None
    MAX_TEST_ROWS: Optional[int] = None

    # calibracao de threshold
    MODE: str = "best_f1"          # "constrained_roc" | "target_far" | "best_f1"
    TARGET_FAR: float = 1e-4
    MIN_RECALL_AT_FPR: float = 0.01

    # pooling entre templates
    TOPK_POOL: int = 1

    # injecoes sinteticas
    DO_INJECT_SWEEP: bool = True
    INJECT_N: int = 64
    INJECT_AMP: float = 0.6
    INJECT_EXPECT_MIN: float = 0.15

    # saida
    OUT_DIR_ROOT: str = "reports/mf_baseline"

    # CPU
    N_JOBS: int = -1
    BATCH_SZ: int = 256

    # feedback
    HEARTBEAT_EVERY_N: int = 10000
    HEARTBEAT_EVERY_SEC: float = 100.0
    TQDM_MIN_INTERVAL: float = 0.5

    # =========================
    # N1: Varredura sistemática
    # =========================
    SWEEP_ENABLED: bool = True                   # habilita a varredura antes do run final
    SWEEP_MAX_VAL_ROWS: int = 30000              # tamanho do VAL pequeno
    SWEEP_METRIC: str = "recall_at_far"          # "recall_at_far" ou "auc"
    SWEEP_SMALL_FAR: float = 1e-3                # FAR para ordenar candidatos no VAL pequeno
    SWEEP_FAR: float = 1e-4                      # FAR alvo para recheck e escolha final
    SWEEP_GRID_TEMPLATES_N: Tuple[int, ...] = (256, 512, 1024)
    SWEEP_GRID_TEMPLATE_SUBSAMPLE: Tuple[int, ...] = (1, 2, 4)
    SWEEP_GRID_TOPK: Tuple[int, ...] = (1, 3)
    SWEEP_GRID_M1: Tuple[Tuple[float,float], ...] = ((5.0, 40.0), (10.0, 50.0), (20.0, 80.0))
    SWEEP_GRID_M2: Tuple[Tuple[float,float], ...] = ((5.0, 40.0), (10.0, 50.0), (20.0, 80.0))

    # Recheck
    SWEEP_TOPK_RECHECK: int = 5                  # top-k para pooling no recheck
    SWEEP_EXPAND_NEG: int = 180_000              # max de negativos no recheck
    SWEEP_KEEP_TOP_N_FOR_RECHECK: int = 1        # quantos setups do small passar para recheck

CFG = CFG()

# =========================
# Helpers
# =========================
def _log(s: str): print(s)

def ts_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())

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

@contextmanager
def patch_cfg(**over):
    keep = {}
    for k, v in over.items():
        keep[k] = getattr(CFG, k)
        setattr(CFG, k, v)
    try:
        yield
    finally:
        for k, v in keep.items():
            setattr(CFG, k, v)

def cache_name_for(m1rng, m2rng, n, subs):
    m1a, m1b = m1rng; m2a, m2b = m2rng
    return f"tmplcache_m1_{m1a:.0f}-{m1b:.0f}_m2_{m2a:.0f}-{m2b:.0f}_n{n}_sub{subs}.npz"

# =========================
# Index de windows e caches
# =========================
def build_windows_index(windows_dir: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for p in glob(os.path.join(windows_dir, "**", "*_windows.hdf5"), recursive=True):
        fid = os.path.basename(p)
        idx[fid] = os.path.abspath(p)
    return idx

@lru_cache(maxsize=256)
def cached_start_arrays(win_path: str):
    with h5py.File(win_path, "r") as f:
        sgps = f["/windows/start_gps"][()].astype(float)
        sidx = f["/windows/start_idx"][()].astype(int)
        meta_in = read_attrs(f["/meta_in"]) if "/meta_in" in f else {}
    return sgps, sidx, meta_in

def resolve_whitened_path(win_path: str, meta_in: Dict[str, str]) -> Optional[str]:
    src = meta_in.get("source_path") or meta_in.get("source_file") or meta_in.get("source")
    candidates = []
    if src and os.path.isabs(src):
        candidates.append(src)
    if src:
        candidates += [
            os.path.join(os.path.dirname(win_path), src),
            os.path.join("data", "interim", os.path.basename(src)),
            os.path.join("data", "processed", os.path.basename(src)),
        ]
    bn = os.path.basename(win_path).replace("_windows.hdf5", "_whitened.hdf5")
    candidates += [
        os.path.join(os.path.dirname(win_path), bn),
        os.path.join("data", "interim", bn),
        os.path.join("data", "processed", bn),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    if src:
        base = os.path.basename(str(src))
        guesses = [
            os.path.join(os.path.dirname(win_path), base),
            os.path.join("data", "interim", base),
            os.path.join("data", "processed", base),
        ]
        for g in guesses:
            if os.path.exists(g):
                return os.path.abspath(g)
    return None

# =========================
# Templates IMRPhenomD
# =========================
def tukey(N: int, alpha: float = 0.1) -> np.ndarray:
    try:    return get_window(("tukey", alpha), N, fftbins=False).astype(np.float32)
    except: return get_window("hann", N, fftbins=False).astype(np.float32)

def synth_template(m1: float, m2: float, s1: float, s2: float,
                   fs: int, wlen: int, f_low: float, alpha: float) -> np.ndarray:
    hp, _ = get_td_waveform(approximant="IMRPhenomD",
                            mass1=m1, mass2=m2,
                            spin1z=s1, spin2z=s2,
                            delta_t=1.0/fs,
                            f_lower=f_low)
    x = hp.numpy().astype(np.float32)
    if x.size > wlen: x = x[-wlen:]
    elif x.size < wlen:
        buf = np.zeros(wlen, np.float32); buf[-x.size:] = x; x = buf
    w = tukey(wlen, alpha=max(0.0, min(1.0, alpha)))
    x = (x * w).astype(np.float32)
    x -= x.mean()
    x /= (x.std() + 1e-12)
    return x

def build_template_bank(n: int) -> List[Tuple[float, float, float, float]]:
    side = max(1, int(round(math.sqrt(n))))
    m1s = np.linspace(CFG.M1_RANGE[0], CFG.M1_RANGE[1], side)
    m2s = np.linspace(CFG.M2_RANGE[0], CFG.M2_RANGE[1], side)
    bank = []
    for m1 in m1s:
        for m2 in m2s:
            bank.append((float(m1), float(m2), CFG.SPIN1, CFG.SPIN2))
            if len(bank) >= n: return bank
    return bank

def build_template_cache(bank, fs, window_sec) -> Dict[str, np.ndarray]:
    wlen = int(round(window_sec * fs))
    cache: Dict[str, np.ndarray] = {}
    for (m1, m2, s1, s2) in bank:
        key = f"{m1:.3f}-{m2:.3f}-{s1:.3f}-{s2:.3f}"
        cache[key] = synth_template(m1, m2, s1, s2, fs, wlen, CFG.F_LO, CFG.HP_WINDOW_TAPER)
    return cache

def maybe_load_template_cache(out_dir: str, bank, cache_name: Optional[str] = None) -> Dict[str, np.ndarray]:
    cache_path = os.path.join(out_dir, cache_name or CFG.CACHE_FILE)
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as npz:
                keys = list(npz["keys"])
                waves = list(npz["waves"])
            return {k: w for k, w in zip(keys, waves)}
        except Exception:
            pass
    cache = build_template_cache(bank, CFG.FS_TARGET, CFG.WINDOW_SEC)
    np.savez(cache_path,
             keys=np.array(list(cache.keys()), dtype=object),
             waves=np.array(list(cache.values()), dtype=object))
    return cache

# =========================
# Leitura de janelas
# =========================
def load_window_slice(win_path: str, whitened_path: str, start_gps: float) -> Optional[np.ndarray]:
    try:
        sgps, sidx, _ = cached_start_arrays(win_path)
        idxs = np.where(np.isclose(sgps, float(start_gps), atol=5e-4))[0]
        if idxs.size == 0:
            if DEBUG_STRICT:
                FAIL_COUNTER["miss_sgps"] += 1
                FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                  "reason": "miss_sgps", "start_gps": float(start_gps)})
            return None
        i0 = int(sidx[int(idxs[0])])

        with h5py.File(whitened_path, "r") as g:
            cand = [
                "/strain/StrainWhitened", "/strain/whitened",
                "/data/whitened", "/whitened",
                "/strain/Strain"
            ]
            dset = None
            for c in cand:
                if c in g and isinstance(g[c], h5py.Dataset):
                    dset = g[c]; break
            if dset is None:
                if DEBUG_STRICT:
                    FAIL_COUNTER["channel_not_found"] += 1
                    FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                      "reason": "channel_not_found",
                                      "available": list(g.keys())})
                return None

            a_g = read_attrs(g) | read_attrs(dset)
            fs_in = a_g.get("fs") or a_g.get("sample_rate")
            if not fs_in:
                xs = a_g.get("xspacing") or a_g.get("Xspacing")
                if not xs:
                    if DEBUG_STRICT:
                        FAIL_COUNTER["fs_unknown"] += 1
                        FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                          "reason": "fs_unknown"})
                    return None
                fs_in = 1.0 / float(xs)
            fs_in = int(round(float(fs_in)))

            wlen_in = int(round(CFG.WINDOW_SEC * fs_in))
            n_total = int(dset.shape[0])
            if i0 < 0 or (i0 + wlen_in) > n_total:
                if DEBUG_STRICT:
                    FAIL_COUNTER["out_of_bounds"] += 1
                    FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                      "reason": "out_of_bounds", "i0": i0,
                                      "need": wlen_in, "n_total": n_total})
                return None

            x = dset[i0:i0 + wlen_in].astype(np.float32, copy=True)

        if fs_in != CFG.FS_TARGET:
            frac = Fraction(CFG.FS_TARGET, fs_in).limit_denominator(64)
            x = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32, copy=False)

        wlen_tgt = int(round(CFG.WINDOW_SEC * CFG.FS_TARGET))
        if x.size > wlen_tgt: x = x[-wlen_tgt:]
        elif x.size < wlen_tgt:
            buf = np.zeros(wlen_tgt, np.float32); buf[-x.size:] = x; x = buf

        if not np.any(np.isfinite(x)):
            if DEBUG_STRICT:
                FAIL_COUNTER["non_finite"] += 1
                FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                  "reason": "non_finite"})
            return None

        x -= x.mean()
        x /= (x.std() + 1e-12)

        if CFG.ENABLE_GLITCH_VETO:
            peak = float(np.max(np.abs(x)))
            kurt = float(np.mean(((x - x.mean())/(x.std()+1e-12))**4))
            if peak > CFG.GLITCH_VETO_PEAK:
                FAIL_COUNTER["glitch_veto_peak"] += 1
                FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                  "reason": "glitch_veto_peak", "peak": peak})
                return None
            if kurt > CFG.GLITCH_VETO_KURT:
                FAIL_COUNTER["glitch_veto_kurt"] += 1
                FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                                  "reason": "glitch_veto_kurt", "kurt": kurt})
                return None

        return x
    except Exception as e:
        if DEBUG_STRICT:
            FAIL_COUNTER["exception_load"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "exception_load", "msg": repr(e)})
        return None

# =========================
# NCC helpers CPU
# =========================
def _ncc_fft_series_cpu(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    c = np_fftconvolve(x, h[::-1], mode="full")
    denom = (np.linalg.norm(x) * np.linalg.norm(h)) + 1e-12
    return (c / denom).astype(np.float32, copy=False)

def _ncc_fft_max_cpu(x: np.ndarray, h: np.ndarray, max_shift_samp: int, lag_step: int) -> float:
    c = _ncc_fft_series_cpu(x, h)
    L = c.size
    center = L // 2
    lo = max(0, center - max_shift_samp)
    hi = min(L, center + max_shift_samp + 1)
    return float(np.nanmax(c[lo:hi:lag_step]))

def _ncc_native_max_cpu(x: np.ndarray, h: np.ndarray, max_shift_samp: int, lag_step: int) -> float:
    a = x.astype(np.float64, copy=False)
    b = h.astype(np.float64, copy=False)
    max_shift_samp = int(max(0, min(max_shift_samp, len(a) - 1)))
    if max_shift_samp == 0:
        r = ncc_native.correlate_all_lags(a, b, 0)
        return float(r[0])
    r = ncc_native.correlate_all_lags(a, b, int(max_shift_samp))  # shape 2*max_shift+1
    return float(np.max(np.asarray(r)[::max(1, int(lag_step))]))

# =========================
# GPU streaming
# =========================
def _gpu_autotune(B_init: int, Kc_init: int, nfft: int, n_full: int, F: int) -> Tuple[int, int]:
    B, Kc = B_init, Kc_init
    if not _HAS_CUPY: return B, Kc

    def enough(Bt: int) -> bool:
        try:
            x = cp.empty((Bt, F*2), dtype=cp.float32)
            xf= cp.empty((Bt, F), dtype=cp.complex64)
            ct= cp.empty((Bt, n_full), dtype=cp.float32)
            ht= cp.empty((F,), dtype=cp.complex64)
            del x, xf, ct, ht
            cp.get_default_memory_pool().free_all_blocks()
            return True
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            return False

    while B >= CFG.GPU_MIN_BATCH_WIN and not enough(B):
        B //= 2
    if B < CFG.GPU_MIN_BATCH_WIN:
        B = CFG.GPU_MIN_BATCH_WIN
    if Kc < CFG.GPU_MIN_TEMPL_CHUNK:
        Kc = CFG.GPU_MIN_TEMPL_CHUNK
    return B, Kc

def score_subset_gpu_streaming(name: str, df_sub: pd.DataFrame, idx_map: Dict[str,str],
                               tmpl_cache: Dict[str,np.ndarray]) -> np.ndarray:
    n = len(df_sub)
    scores = np.zeros(n, np.float32)
    if n == 0:
        _log(f"[{name}] janelas=0 | GPU streaming pulado")
        return scores

    keys = list(tmpl_cache.keys())
    if CFG.TEMPLATE_SUBSAMPLE > 1:
        keys = keys[::CFG.TEMPLATE_SUBSAMPLE]
    templates_host = [tmpl_cache[k] for k in keys]
    K = len(templates_host)

    L = int(round(CFG.WINDOW_SEC * CFG.FS_TARGET))
    n_full = 2 * L - 1
    nfft = _next_pow2(n_full)
    F = nfft // 2 + 1

    fs = CFG.FS_TARGET
    max_shift = int(round(CFG.MAX_SHIFT_SEC * fs))
    lag_step  = max(1, int(CFG.LAG_STEP))
    center = n_full // 2
    lo = max(0, center - max_shift)
    hi = min(n_full, center + max_shift + 1)

    B_cur, Kc_cur = _gpu_autotune(CFG.GPU_INIT_BATCH_WIN, CFG.GPU_INIT_TEMPL_CHUNK, nfft, n_full, F)

    idx = df_sub.index.to_list()
    _log(f"[{name}] janelas={n:,} | templates={K} | nfft={nfft} | max_shift={CFG.MAX_SHIFT_SEC:.3f}s | lag_step={lag_step} | GPU streaming")
    _log(f"[{name}] autotune: batch_win={B_cur} | templ_chunk={Kc_cur}")

    t0 = time.time()
    with tqdm(total=n, unit="win", mininterval=CFG.TQDM_MIN_INTERVAL, desc=f"[{name}] NCC", leave=True) as bar:
        b0 = 0
        while b0 < n:
            b_idx = idx[b0:b0+B_cur]

            # carrega lote no host
            X_host = []
            for i in b_idx:
                fid  = df_sub.at[i, "file_id"]
                sgps = float(df_sub.at[i, "start_gps"])
                win_path = idx_map.get(fid)
                if not win_path:
                    X_host.append(None); continue
                _, _, meta_in = cached_start_arrays(win_path)
                wht = resolve_whitened_path(win_path, meta_in)
                if not wht or not os.path.exists(wht):
                    X_host.append(None); continue
                x = load_window_slice(win_path, wht, sgps)
                if x is None or not np.any(np.isfinite(x)):
                    X_host.append(None); continue
                X_host.append(x.astype(np.float32, copy=False))

            mask_valid = np.array([x is not None for x in X_host], dtype=bool)
            if not mask_valid.any():
                bar.update(len(b_idx)); b0 += B_cur; continue

            try:
                Xv = np.stack([x for x in X_host if x is not None], axis=0)  # (Bv, L)
                dtype_time = cp.float16 if CFG.USE_FP16_TIME else cp.float32
                X_dev = cp.asarray(Xv, dtype=dtype_time)
                X_fft = cp.fft.rfft(X_dev.astype(cp.float32, copy=False), nfft, axis=1)  # (Bv,F)
                X_norm = cp.linalg.norm(X_dev.astype(cp.float32, copy=False), axis=1).astype(cp.float32, copy=False)

                # k melhores por janela
                topk = CFG.TOPK_POOL if CFG.TOPK_POOL and CFG.TOPK_POOL > 1 else 1
                if topk == 1:
                    agg = cp.full((X_dev.shape[0],), -cp.inf, dtype=cp.float32)
                else:
                    agg = cp.full((X_dev.shape[0], topk), -cp.inf, dtype=cp.float32)

                for t0_idx in range(0, K, Kc_cur):
                    t1_idx = min(t0_idx + Kc_cur, K)
                    for k in range(t0_idx, t1_idx):
                        h = templates_host[k]
                        h_dev = cp.asarray(h[::-1], dtype=cp.float32)
                        H_fft = cp.fft.rfft(h_dev, nfft)
                        C_fft = X_fft * cp.conj(H_fft)
                        c = cp.fft.irfft(C_fft, nfft, axis=1)[:, :n_full]
                        denom = (X_norm) * cp.linalg.norm(h_dev) + 1e-12
                        c = c / denom[:, None]
                        m = cp.max(c[:, lo:hi:lag_step], axis=1)
                        if topk == 1:
                            agg = cp.maximum(agg, m)
                        else:
                            min_idx = cp.argmin(agg, axis=1)
                            rows = cp.arange(agg.shape[0])
                            replace_mask = m > agg[rows, min_idx]
                            agg[rows, min_idx] = cp.where(replace_mask, m, agg[rows, min_idx])
                        del h_dev, H_fft, C_fft, c, m
                        cp.get_default_memory_pool().free_all_blocks()

                if topk == 1:
                    best_host = agg.get()
                else:
                    best_host = cp.mean(agg, axis=1).get()

                pos_valid = np.where(mask_valid)[0]
                for j, val in zip(pos_valid, best_host):
                    out_pos = df_sub.index.get_loc(b_idx[j])
                    scores[out_pos] = float(val)

                del X_dev, X_fft, X_norm, agg
                cp.get_default_memory_pool().free_all_blocks()

                bar.update(len(b_idx))
                b0 += B_cur

            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()
                if B_cur > CFG.GPU_MIN_BATCH_WIN:
                    B_cur = max(CFG.GPU_MIN_BATCH_WIN, B_cur // 2)
                    _log(f"[{name}] OOM: reduzindo batch_win para {B_cur}")
                    continue
                elif Kc_cur > CFG.GPU_MIN_TEMPL_CHUNK:
                    Kc_cur = max(CFG.GPU_MIN_TEMPL_CHUNK, Kc_cur // 2)
                    _log(f"[{name}] OOM: reduzindo templ_chunk para {Kc_cur}")
                    continue
                else:
                    raise

    _log(f"      [{name}] tempo={time.time()-t0:.1f}s")
    return scores

# =========================
# CPU e roteador
# =========================
def _score_one_cpu_fft(fid: str, sgps: float, idx_map: Dict[str,str], max_shift: int,
                       lag_step: int, templates_host: List[np.ndarray], topk: int) -> float:
    win_path = idx_map.get(fid)
    if not win_path:
        if DEBUG_STRICT: FAIL_COUNTER["no_win_path"] += 1
        return 0.0
    _, _, meta_in = cached_start_arrays(win_path)
    wht = resolve_whitened_path(win_path, meta_in)
    if not wht or not os.path.exists(wht):
        if DEBUG_STRICT:
            FAIL_COUNTER["wht_missing"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "wht_missing",
                              "meta_in": dict(meta_in)})
        return 0.0

    try:
        x = load_window_slice(win_path, wht, float(sgps))
        if x is None or not np.any(np.isfinite(x)):
            if DEBUG_STRICT: FAIL_COUNTER["slice_none"] += 1
            return 0.0
        vals = []
        for h in templates_host:
            s = _ncc_fft_max_cpu(x, h, max_shift_samp=max_shift, lag_step=lag_step)
            vals.append(s)
        if not vals:
            return 0.0
        vals = np.array(vals, dtype=np.float32)
        if topk <= 1:
            return float(np.max(vals))
        k = min(topk, vals.size)
        return float(np.mean(np.partition(vals, -k)[-k:]))
    except Exception as e:
        if DEBUG_STRICT:
            FAIL_COUNTER["exception_score"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "exception_score", "msg": repr(e)})
        return 0.0

def _score_one_cpu_native(fid: str, sgps: float, idx_map: Dict[str,str], max_shift: int,
                          lag_step: int, templates_host: List[np.ndarray], topk: int) -> float:
    win_path = idx_map.get(fid)
    if not win_path:
        if DEBUG_STRICT: FAIL_COUNTER["no_win_path"] += 1
        return 0.0
    _, _, meta_in = cached_start_arrays(win_path)
    wht = resolve_whitened_path(win_path, meta_in)
    if not wht or not os.path.exists(wht):
        if DEBUG_STRICT:
            FAIL_COUNTER["wht_missing"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "wht_missing",
                              "meta_in": dict(meta_in)})
        return 0.0

    try:
        x = load_window_slice(win_path, wht, float(sgps))
        if x is None or not np.any(np.isfinite(x)):
            if DEBUG_STRICT: FAIL_COUNTER["slice_none"] += 1
            return 0.0

        vals = []
        for h in templates_host:
            s = _ncc_native_max_cpu(x, h, max_shift_samp=max_shift, lag_step=lag_step)
            vals.append(s)
        if not vals:
            return 0.0
        vals = np.array(vals, dtype=np.float32)
        if topk <= 1:
            return float(np.max(vals))
        k = min(topk, vals.size)
        return float(np.mean(np.partition(vals, -k)[-k:]))
    except Exception as e:
        if DEBUG_STRICT:
            FAIL_COUNTER["exception_score_native"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "exception_score_native", "msg": repr(e)})
        return 0.0

def score_subset_cpu(name: str, df_sub: pd.DataFrame, idx_map: Dict[str,str],
                     tmpl_cache: Dict[str,np.ndarray]) -> np.ndarray:
    n = len(df_sub)
    scores = np.zeros(n, np.float32)
    if n == 0:
        _log(f"[{name}] janelas=0 | CPU pulado")
        return scores

    keys = list(tmpl_cache.keys())
    if CFG.TEMPLATE_SUBSAMPLE > 1:
        keys = keys[::CFG.TEMPLATE_SUBSAMPLE]
    templates_host = [tmpl_cache[k] for k in keys]

    fs = CFG.FS_TARGET
    max_shift = int(round(CFG.MAX_SHIFT_SEC * fs))
    lag_step  = max(1, int(CFG.LAG_STEP))
    topk = max(1, int(CFG.TOPK_POOL))

    use_native = bool(_HAS_NATIVE and CFG.USE_NATIVE_NCC and max_shift <= CFG.NATIVE_MAXSHIFT_HEURISTIC)
    mode_cpu = "native" if use_native else "fft"

    _log(f"[{name}] janelas={n:,} | templates={len(templates_host)} | topk={topk} | max_shift={CFG.MAX_SHIFT_SEC:.3f}s | lag_step={lag_step} | jobs={CFG.N_JOBS} | CPU mode={mode_cpu}")

    idx = df_sub.index.to_list()
    batches = [idx[i:i+CFG.BATCH_SZ] for i in range(0, n, CFG.BATCH_SZ)]

    t0 = time.time(); last_hb = t0; done = 0
    fn_worker = _score_one_cpu_native if use_native else _score_one_cpu_fft

    with tqdm(total=n, unit="win", mininterval=CFG.TQDM_MIN_INTERVAL, desc=f"[{name}] NCC", leave=True) as bar:
        for b in batches:
            out_final = Parallel(n_jobs=CFG.N_JOBS, backend="loky")(
                delayed(fn_worker)(
                    df_sub.at[i, "file_id"],
                    float(df_sub.at[i, "start_gps"]),
                    idx_map, max_shift, lag_step,
                    templates_host, topk
                ) for i in b
            )
            pos = [df_sub.index.get_loc(i) for i in b]
            scores[pos] = out_final
            done += len(b); bar.update(len(b))

            now = time.time()
            if (done % CFG.HEARTBEAT_EVERY_N == 0) or (now - last_hb >= CFG.HEARTBEAT_EVERY_SEC):
                rate = done / max(now - t0, 1e-9)
                eta  = (n - done) / max(rate, 1e-9)
                print(f"  [hb] {name}: {done:,}/{n:,} ({100*done/n:.1f}%)  {rate:.1f} win/s  ETA {eta/60:.1f} min")
                last_hb = now

    _log(f"      [{name}] tempo={time.time()-t0:.1f}s | ~{(n/max(time.time()-t0,1e-9)):.1f} win/s")
    return scores

def score_subset(name: str, df_sub: pd.DataFrame, idx_map: Dict[str,str],
                 tmpl_cache: Dict[str,np.ndarray]) -> np.ndarray:
    if CFG.USE_GPU and _HAS_CUPY:
        return score_subset_gpu_streaming(name, df_sub, idx_map, tmpl_cache)
    return score_subset_cpu(name, df_sub, idx_map, tmpl_cache)

# =========================
# Thresholds e metricas
# =========================
def _safe_thr_after_pick(thr: float, scores: np.ndarray) -> float:
    if not np.isfinite(thr):
        smax = float(np.max(scores)) if scores.size else 0.0
        return float(np.nextafter(smax, np.float32(-np.inf)))
    smax = float(np.max(scores)) if scores.size else 0.0
    if thr > smax:
        return float(np.nextafter(smax, np.float32(-np.inf)))
    return float(thr)

def _ensure_nonzero_tp(y_true: np.ndarray, scores: np.ndarray, thr: float) -> float:
    yhat = (scores >= thr).astype(int)
    tp = int(np.sum((yhat==1) & (y_true==1)))
    if tp > 0:
        return float(thr)
    q = min(0.999, max(0.95, 1.0 - 10.0/ max((y_true==0).sum(), 1)))
    thr2 = float(np.quantile(scores, q)) if scores.size else thr
    return float(thr2)

def pick_threshold_constrained(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        thr = float(np.percentile(scores, 99.9)) if scores.size > 0 else 0.0
        return thr, {"mode": "degenerate", "note": "labels not separable"}

    fpr, tpr, thr_arr = roc_curve(y_true, scores)
    mask = (fpr <= CFG.TARGET_FAR) & np.isfinite(thr_arr)
    if np.any(mask):
        idx = np.argmax(tpr[mask])
        thr = float(thr_arr[mask][idx])
        info = {"mode": "constrained_roc", "target_fpr": CFG.TARGET_FAR,
                "chosen_fpr": float(fpr[mask][idx]),
                "chosen_tpr": float(tpr[mask][idx])}
    else:
        maskf = np.isfinite(thr_arr)
        if not np.any(maskf):
            thr = float(np.max(scores)) if scores.size else 0.0
            info = {"mode": "constrained_roc", "note": "no finite threshold from ROC"}
        else:
            min_fpr = float(np.min(fpr[maskf]))
            cand = np.where(maskf & (fpr == min_fpr))[0]
            idx2 = cand[np.argmax(tpr[cand])]
            thr = float(thr_arr[idx2])
            info = {"mode": "relaxed_roc", "min_fpr": float(min_fpr), "chosen_tpr": float(tpr[idx2])}

    thr = _safe_thr_after_pick(thr, scores)
    return thr, info

def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    if CFG.MODE == "constrained_roc":
        return pick_threshold_constrained(y_true, scores)

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if CFG.MODE == "target_far":
        neg = scores[y_true == 0]
        if neg.size == 0:
            return 0.0, {"mode": "target_far", "far": CFG.TARGET_FAR, "note": "no_neg"}
        q = 1.0 - CFG.TARGET_FAR
        thr = float(np.quantile(neg, q)) if neg.size > 10 else float(np.max(neg))
        return _safe_thr_after_pick(thr, scores), {"mode": "target_far", "far": CFG.TARGET_FAR}

    # best_f1
    thr_cand = np.unique(scores)
    best_f1, best_thr = -1.0, 0.0
    for t in thr_cand:
        yhat = (scores >= t).astype(int)
        tp = int(np.sum((yhat == 1) & (y_true == 1)))
        fp = int(np.sum((yhat == 1) & (y_true == 0)))
        fn = int(np.sum((yhat == 0) & (y_true == 1)))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    best_thr = _safe_thr_after_pick(best_thr, scores)
    return best_thr, {"mode": "best_f1", "f1": best_f1}

def summarize_and_plot(y_true: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")

    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k:',alpha=.4)
        if np.isfinite(CFG.TARGET_FAR) and CFG.TARGET_FAR > 0:
            plt.axvline(CFG.TARGET_FAR, color="#ef4444", ls="--", lw=0.9, alpha=0.7, label=f"alvo FPR={CFG.TARGET_FAR:g}")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{prefix}.png"), dpi=140); plt.close()
    except Exception:
        pass

    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_{prefix}.png"), dpi=140); plt.close()
    except Exception:
        pass

    try:
        plt.figure(figsize=(6.2,3.6))
        plt.hist(scores[y_true==0], bins=60, alpha=.7, label="neg", density=True, color="#2563eb")
        plt.hist(scores[y_true==1], bins=60, alpha=.7, label="pos", density=True, color="#ef4444")
        plt.legend(); plt.grid(alpha=.2); plt.title(f"Score dist - {prefix}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"hist_{prefix}.png"), dpi=140); plt.close()
    except Exception:
        pass

    return auc, ap

def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float):
    yhat = (scores >= thr).astype(int)
    tn = int(np.sum((yhat==0) & (y_true==0)))
    fp = int(np.sum((yhat==1) & (y_true==0)))
    fn = int(np.sum((yhat==0) & (y_true==1)))
    tp = int(np.sum((yhat==1) & (y_true==1)))
    return tn, fp, fn, tp

def thresholds_report_grid(y_true: np.ndarray, scores: np.ndarray, fars=(1e-6,5e-6,1e-5,5e-4,1e-3,5e-3,1e-2)):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    out = {}
    neg = scores[y_true==0]
    for far in fars:
        if neg.size == 0:
            out[str(far)] = {"threshold": None, "fpr": 0.0, "tpr": 0.0}
            continue
        thr = float(np.quantile(neg, 1.0 - far)) if neg.size > 10 else float(np.max(neg))
        thr = _safe_thr_after_pick(thr, scores)
        yhat = (scores >= thr).astype(int)
        tn = int(np.sum((yhat==0) & (y_true==0)))
        fp = int(np.sum((yhat==1) & (y_true==0)))
        fn = int(np.sum((yhat==0) & (y_true==1)))
        tp = int(np.sum((yhat==1) & (y_true==1)))
        fpr = fp / max(tn+fp, 1)
        tpr = tp / max(tp+fn, 1)
        out[str(far)] = {"threshold": thr, "fpr": float(fpr), "tpr": float(tpr), "tp": tp, "fp": fp}
    return out

# =========================
# INJECT sweep
# =========================
def inject_sweep_quick(idx_map: Dict[str,str], tmpl_cache: Dict[str,np.ndarray]) -> None:
    try:
        import random
        any_fid = next(iter(idx_map.keys()))
        any_win = idx_map[any_fid]
        sgps, _, meta_in = cached_start_arrays(any_win)
        if len(sgps) < 100:
            print("[INJECT] sgps insuficiente, pulando.")
            return
        wht = resolve_whitened_path(any_win, meta_in)
        if not wht or not os.path.exists(wht):
            print("[INJECT] whitened ausente, pulando.")
            return

        ks = list(tmpl_cache.keys())
        random.shuffle(ks)
        ks = ks[:min(CFG.INJECT_N, len(ks))]
        vals = []
        hits = 0
        thr_probe = CFG.INJECT_EXPECT_MIN
        for k in ks:
            h = tmpl_cache[k]
            g = float(np.random.choice(sgps[20:-20]))
            xr = load_window_slice(any_win, wht, g)
            if xr is None:
                continue
            inj = ((xr + CFG.INJECT_AMP*h[:len(xr)]) - np.mean(xr)) / (np.std(xr)+1e-12)
            s = _ncc_fft_max_cpu(inj, h, int(CFG.MAX_SHIFT_SEC*CFG.FS_TARGET), max(1, CFG.LAG_STEP))
            vals.append(s)
            hits += int(s > thr_probe)
        if vals:
            print(f"[INJECT] n={len(vals)} med={np.median(vals):.3f} max={np.max(vals):.3f} recall@{thr_probe:.2f}={hits/len(vals):.2f}")
        else:
            print("[INJECT] sem medidas validas.")
    except Exception as e:
        print(f"[INJECT] falhou: {e}")

# =========================
# N1: Varredura sistemática
# =========================
def run_val_once(df_val: pd.DataFrame, idx_map: Dict[str,str], out_dir: str,
                 cache_name: Optional[str] = None, far_for_eval: Optional[float] = None) -> Dict:
    # monta banco conforme CFG atual
    bank = build_template_bank(CFG.TEMPLATES_N)
    tmpl_cache = maybe_load_template_cache(out_dir, bank, cache_name)
    # score
    s_val = score_subset("VAL", df_val, idx_map, tmpl_cache)
    y_val = df_val["label"].to_numpy(int)
    # métricas
    auc_val, ap_val = summarize_and_plot(y_val, s_val, out_dir, "val_probe")
    # recall@FAR
    far = CFG.SWEEP_FAR if far_for_eval is None else float(far_for_eval)
    rep = thresholds_report_grid(y_val, s_val, fars=(far,))
    key = f"{far}"
    rec_at_far = rep[key]["tpr"] if key in rep else float("nan")
    fpr_at_far = rep[key]["fpr"] if key in rep else float("nan")
    thr_at_far = rep[key]["threshold"] if key in rep else float("nan")
    # contagem de FP no sample para transparência
    yhat = (s_val >= thr_at_far).astype(int)
    fp_cnt = int(np.sum((yhat == 1) & (y_val == 0)))
    return {
        "auc": float(auc_val),
        "ap": float(ap_val),
        "recall_at_far": float(rec_at_far),
        "fpr_at_far": float(fpr_at_far),
        "threshold_at_far": float(thr_at_far),
        "fp_at_thr": int(fp_cnt)
    }

def _build_val_small(df_val_full: pd.DataFrame) -> pd.DataFrame:
    # Inclui todos os positivos e completa com negativos até SWEEP_MAX_VAL_ROWS
    df_pos = df_val_full[df_val_full["label"] == 1]
    df_neg = df_val_full[df_val_full["label"] == 0]
    n_neg_needed = max(0, int(CFG.SWEEP_MAX_VAL_ROWS) - len(df_pos))
    df_neg_small = df_neg.head(n_neg_needed)
    df_small = pd.concat([df_pos, df_neg_small], axis=0, ignore_index=False).sort_index()
    dist = dict(df_small["label"].value_counts().sort_index().to_dict())
    _log(f"[SWEEP] VAL amostrado = {len(df_small):,} | dist={dist}  (pos={dist.get(1,0)}, neg={dist.get(0,0)})")
    return df_small

def _build_val_recheck(df_val_full: pd.DataFrame) -> pd.DataFrame:
    # Recheck: todos positivos + muitos negativos
    df_pos = df_val_full[df_val_full["label"] == 1]
    df_neg = df_val_full[df_val_full["label"] == 0].head(int(CFG.SWEEP_EXPAND_NEG))
    df_big = pd.concat([df_pos, df_neg], axis=0, ignore_index=False).sort_index()
    dist = dict(df_big["label"].value_counts().sort_index().to_dict())
    _log(f"[RECHECK] VAL expandido = {len(df_big):,} | dist={dist}  (pos={dist.get(1,0)}, neg={dist.get(0,0)})")
    return df_big

def sweep_parameters(df_val_full: pd.DataFrame, idx_map: Dict[str,str], out_dir: str) -> Dict:
    # constrói VAL pequeno e VAL grande
    df_val_small = _build_val_small(df_val_full)
    df_val_big   = _build_val_recheck(df_val_full)

    # grade
    grid = list(itertools.product(
        CFG.SWEEP_GRID_M1,
        CFG.SWEEP_GRID_M2,
        CFG.SWEEP_GRID_TEMPLATES_N,
        CFG.SWEEP_GRID_TEMPLATE_SUBSAMPLE,
        CFG.SWEEP_GRID_TOPK
    ))

    rows_small = []
    best_small = None
    t_start = time.time()

    # Avaliação no VAL pequeno com FAR grosso
    for m1rng, m2rng, n_tmpl, subs, topk in grid:
        t0 = time.time()
        cache_name = cache_name_for(m1rng, m2rng, n_tmpl, subs)
        with patch_cfg(
            M1_RANGE=m1rng, M2_RANGE=m2rng,
            TEMPLATES_N=n_tmpl, TEMPLATE_SUBSAMPLE=subs,
            TOPK_POOL=topk,
            CACHE_FILE=cache_name
        ):
            try:
                res = run_val_once(df_val_small, idx_map, out_dir, cache_name=cache_name, far_for_eval=CFG.SWEEP_SMALL_FAR)
                auc = res["auc"]; rec_far = res["recall_at_far"]; fpr_far = res["fpr_at_far"]
                thr_far = res["threshold_at_far"]; fp_cnt = res["fp_at_thr"]
                dt = time.time() - t0
                row = {
                    "m1_lo": m1rng[0], "m1_hi": m1rng[1],
                    "m2_lo": m2rng[0], "m2_hi": m2rng[1],
                    "templates_n": n_tmpl, "subsample": subs, "topk": topk,
                    "metric_small": CFG.SWEEP_METRIC, "far_small": CFG.SWEEP_SMALL_FAR,
                    "auc_small": float(auc), "recall_at_far_small": float(rec_far),
                    "fpr_at_far_small": float(fpr_far), "thr_at_far_small": float(thr_far),
                    "fp_at_thr_small": int(fp_cnt),
                    "time_s_small": dt, "cache": cache_name
                }
                rows_small.append(row)
                score_key = rec_far if CFG.SWEEP_METRIC == "recall_at_far" else auc
                if best_small is None or score_key > best_small["score"]:
                    best_small = {"score": score_key, "row": row}
                _log(f"[SWEEP] n={n_tmpl} sub={subs} topk={topk} m1={m1rng} m2={m2rng} | "
                     f"AUC={auc:.4f} Rec@FAR{CFG.SWEEP_SMALL_FAR:g}={rec_far:.4f} "
                     f"FPR={fpr_far:.4g} thr={thr_far:.5f} FP={fp_cnt} | {dt:.1f}s")
            except Exception as e:
                _log(f"[SWEEP] falha em {m1rng},{m2rng},n={n_tmpl},sub={subs},topk={topk}: {e}")
                continue

    # ordena candidatos do small
    if not rows_small:
        raise RuntimeError("SWEEP não produziu resultados no VAL pequeno")
    df_small = pd.DataFrame(rows_small)
    # define top-N para recheck
    if CFG.SWEEP_METRIC == "recall_at_far":
        df_small["_score_small"] = df_small["recall_at_far_small"]
    else:
        df_small["_score_small"] = df_small["auc_small"]
    df_small = df_small.sort_values("_score_small", ascending=False)
    topN = int(max(1, CFG.SWEEP_KEEP_TOP_N_FOR_RECHECK))
    cand_rows = df_small.head(topN).to_dict(orient="records")

    # Recheck no VAL grande com FAR alvo e TOPK_POOL reforçado
    rows_recheck = []
    for row in cand_rows:
        t0 = time.time()
        with patch_cfg(
            M1_RANGE=(row["m1_lo"], row["m1_hi"]),
            M2_RANGE=(row["m2_lo"], row["m2_hi"]),
            TEMPLATES_N=int(row["templates_n"]),
            TEMPLATE_SUBSAMPLE=int(row["subsample"]),
            TOPK_POOL=int(max(row["topk"], CFG.SWEEP_TOPK_RECHECK)),
            CACHE_FILE=row["cache"]
        ):
            res = run_val_once(df_val_big, idx_map, out_dir, cache_name=row["cache"], far_for_eval=None)  # usa SWEEP_FAR
            auc = res["auc"]; rec_far = res["recall_at_far"]; fpr_far = res["fpr_at_far"]
            thr_far = res["threshold_at_far"]; fp_cnt = res["fp_at_thr"]
            dt = time.time() - t0
            row2 = dict(row)
            row2.update({
                "metric_recheck": CFG.SWEEP_METRIC, "far_recheck": CFG.SWEEP_FAR,
                "auc_recheck": float(auc), "recall_at_far_recheck": float(rec_far),
                "fpr_at_far_recheck": float(fpr_far), "thr_at_far_recheck": float(thr_far),
                "fp_at_thr_recheck": int(fp_cnt), "time_s_recheck": dt,
                "topk_recheck_used": int(max(row["topk"], CFG.SWEEP_TOPK_RECHECK))
            })
            rows_recheck.append(row2)
            _log(f"[RECHECK] m1=({row2['m1_lo']},{row2['m1_hi']}) m2=({row2['m2_lo']},{row2['m2_hi']}) "
                 f"n={row2['templates_n']} sub={row2['subsample']} topk={row2['topk_recheck_used']} | "
                 f"AUC={auc:.4f} Rec@FAR{CFG.SWEEP_FAR:g}={rec_far:.4f} FPR={fpr_far:.4g} thr={thr_far:.5f} FP={fp_cnt} | {dt:.1f}s")

    # escolhe melhor pelo recheck
    if not rows_recheck:
        # fallback: melhor do small
        best_row = best_small["row"]
        _log("[SWEEP] recheck vazio. Usando melhor do small.")
    else:
        df_r = pd.DataFrame(rows_recheck)
        if CFG.SWEEP_METRIC == "recall_at_far":
            df_r["_score_recheck"] = df_r["recall_at_far_recheck"]
        else:
            df_r["_score_recheck"] = df_r["auc_recheck"]
        df_r = df_r.sort_values("_score_recheck", ascending=False)
        best_row = df_r.iloc[0].to_dict()

    # salva experimentos
    df_small.to_csv(os.path.join(out_dir, "experiments_small.csv"), index=False)
    if rows_recheck:
        pd.DataFrame(rows_recheck).to_csv(os.path.join(out_dir, "experiments_recheck.csv"), index=False)

    _log(f"[SWEEP] tempo_total={time.time()-t_start:.1f}s")
    _log(f"[SWEEP] melhor setup: {best_row}")
    return best_row

# =========================
# MAIN
# =========================
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    t_all = time.time()
    out_dir = os.path.join(CFG.OUT_DIR_ROOT, ts_tag()); ensure_dir(out_dir)

    # 0) log de ambiente
    _log("============= PIPELINE GLOBAL =============")
    _log(f"GPU solicitado={CFG.USE_GPU} disponivel={_HAS_CUPY}" + (f" | nota={_CUPY_ERR}" if (_CUPY_ERR and not _HAS_CUPY) else ""))
    _log(f"NAT solicitado={CFG.USE_NATIVE_NCC} disponivel={_HAS_NATIVE}" + (f" | nota={_NATIVE_ERR}" if (_NATIVE_ERR and not _HAS_NATIVE) else ""))
    _log(f"WINDOW_SEC={CFG.WINDOW_SEC} FS_TARGET={CFG.FS_TARGET} LAG_STEP={CFG.LAG_STEP}")
    _log("===========================================")

    # 1) Carrega dataset
    df = pd.read_parquet(CFG.PARQUET_PATH)
    if "subset" in df.columns:
        split_col = "subset"
    elif "split" in df.columns:
        df = df.rename(columns={"split": "subset"})
        split_col = "subset"
    elif {"is_val","is_test"}.issubset(df.columns):
        conds = np.full(len(df), "train", dtype=object)
        conds[df["is_val"].astype(bool).values] = "val"
        conds[df["is_test"].astype(bool).values] = "test"
        df["subset"] = pd.Categorical(conds, categories=["train","val","test"])
        split_col = "subset"
    else:
        raise KeyError("dataset.parquet sem subset nem split")

    cols_need = {"file_id","start_gps","label", split_col}
    missing = cols_need - set(df.columns)
    if missing:
        raise KeyError(f"colunas ausentes no dataset: {missing}")

    df = df[[split_col,"file_id","start_gps","label"]].copy()
    df_val  = df[df[split_col]=="val"].copy()
    df_test = df[df[split_col]=="test"].copy()
    if CFG.MAX_VAL_ROWS:  df_val  = df_val.head(CFG.MAX_VAL_ROWS).copy()
    if CFG.MAX_TEST_ROWS: df_test = df_test.head(CFG.MAX_TEST_ROWS).copy()
    _log(f"[1/6] Dataset OK: val={len(df_val):,} | test={len(df_test):,}")

    # 2) Index de windows
    _log("[2/6] Indexando *_windows.hdf5 ...")
    idx_map = build_windows_index(CFG.WINDOWS_DIR)
    if not idx_map: raise RuntimeError("nenhum *_windows.hdf5 encontrado em WINDOWS_DIR")

    # 2b) Varredura sistemática
    if CFG.SWEEP_ENABLED:
        _log("[SWEEP] Iniciando varredura sistemática em VAL...")
        best_row = sweep_parameters(df_val, idx_map, out_dir)
        # fixa o melhor setup para o run final
        with patch_cfg(
            M1_RANGE=(best_row["m1_lo"], best_row["m1_hi"]),
            M2_RANGE=(best_row["m2_lo"], best_row["m2_hi"]),
            TEMPLATES_N=int(best_row["templates_n"]),
            TEMPLATE_SUBSAMPLE=int(best_row["subsample"]),
            TOPK_POOL=int(best_row.get("topk_recheck_used", best_row["topk"])),
            CACHE_FILE=str(best_row["cache"])
        ):
            _run_final(df_val, df_test, idx_map, out_dir, sweep_info=best_row)
    else:
        _run_final(df_val, df_test, idx_map, out_dir, sweep_info=None)

    _log(f"Tempo total: {time.time()-t_all:.1f}s")

def _run_final(df_val: pd.DataFrame, df_test: pd.DataFrame, idx_map: Dict[str,str],
               out_dir: str, sweep_info: Optional[Dict]):
    # 3) Templates e self-test
    _log("[2/6] Templates ...")
    bank = build_template_bank(CFG.TEMPLATES_N)
    tmpl_cache = maybe_load_template_cache(out_dir, bank, cache_name=CFG.CACHE_FILE)

    # Self-test robusto
    try:
        k0 = next(iter(tmpl_cache.keys()))
        h  = tmpl_cache[k0]
        any_fid = next(iter(idx_map.keys()))
        any_win = idx_map[any_fid]
        sgps, _, meta_in = cached_start_arrays(any_win)
        if len(sgps) > 10:
            test_gps = float(sgps[len(sgps)//3])
            wht = resolve_whitened_path(any_win, meta_in)
            xr  = load_window_slice(any_win, wht, test_gps)
            if xr is not None:
                xr = (xr - xr.mean()) / (xr.std() + 1e-12)
                inj = xr + CFG.INJECT_AMP * h[:len(xr)]
                inj = (inj - inj.mean()) / (inj.std() + 1e-12)
                peak = _ncc_fft_max_cpu(inj, h, max_shift_samp=int(CFG.MAX_SHIFT_SEC*CFG.FS_TARGET),
                                        lag_step=max(1, CFG.LAG_STEP))
                print(f"[DEBUG] Self-test INJECT (CPU): {peak:.3f} (esperado >= {CFG.INJECT_EXPECT_MIN})")
            else:
                print("[DEBUG] Self-test INJECT pulado: janela real nao carregada.")
        else:
            print("[DEBUG] Self-test INJECT pulado: sgps insuficiente.")
    except Exception as e:
        print(f"[DEBUG] Self-test INJECT falhou: {e}")

    if CFG.DO_INJECT_SWEEP:
        inject_sweep_quick(idx_map, tmpl_cache)

    # 4) VAL
    _log("[3/6] Scoring VAL ...")
    s_val = score_subset("VAL", df_val, idx_map, tmpl_cache)
    print(f"[VAL] scores: min={s_val.min():.4g} med={np.median(s_val):.4g} max={s_val.max():.4g} | frac(>1e-6)={(s_val>1e-6).mean()*100:.3f}%")

    # 5) Thresholds
    _log("[4/6] Thresholds no VAL ...")
    y_val = df_val["label"].to_numpy(int)
    thr, info = pick_threshold(y_val, s_val)
    thr = _ensure_nonzero_tp(y_val, s_val, thr)
    auc_val, ap_val = summarize_and_plot(y_val, s_val, out_dir, "val")
    df_val_out = df_val.copy(); df_val_out["score"] = s_val
    df_val_out.to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)

    far_report = thresholds_report_grid(y_val, s_val)

    # 6) TEST
    _log("[5/6] Scoring TEST ...")
    s_test = score_subset("TEST", df_test, idx_map, tmpl_cache)
    print(f"[TEST] scores: min={s_test.min():.4g} med={np.median(s_test):.4g} max={s_test.max():.4g} | frac(>1e-6)={(s_test>1e-6).mean()*100:.3f}%")
    y_test = df_test["label"].to_numpy(int)
    auc_test, ap_test = summarize_and_plot(y_test, s_test, out_dir, "test")
    df_test_out = df_test.copy(); df_test_out["score"] = s_test
    df_test_out.to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)

    # 7) Relatorio final
    _log("[6/6] Salvando relatorio ...")
    tn, fp, fn, tp = confusion_at_threshold(y_test, s_test, thr)
    summary = {
        "cfg": asdict(CFG),
        "templates_cached": int(len(tmpl_cache)),
        "val": {"auc": float(auc_val), "ap": float(ap_val)},
        "test": {"auc": float(auc_test), "ap": float(ap_test),
                 "confusion_at_thr": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}},
        "threshold": float(thr),
        "threshold_info": info,
        "val_far_report": far_report,
        "out_dir": out_dir,
        "gpu": {"requested": bool(CFG.USE_GPU), "available": bool(_HAS_CUPY), "note": _CUPY_ERR},
        "native": {"requested": bool(CFG.USE_NATIVE_NCC), "available": bool(_HAS_NATIVE), "note": _NATIVE_ERR},
        "sweep_best": sweep_info
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump({
            "mode_used": CFG.MODE,
            "chosen_threshold": float(thr),
            "target_far": float(CFG.TARGET_FAR),
            "val_far_report": far_report,
            "notes": "thr ajustado com fallback para evitar tp=0 e thr=inf"
        }, f, indent=2)

    if DEBUG_STRICT and (sum(FAIL_COUNTER.values()) > 0):
        dbg_dir = os.path.join(out_dir, "_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        with open(os.path.join(dbg_dir, "fail_counters.json"), "w") as f:
            json.dump(dict(FAIL_COUNTER), f, indent=2)
        try:
            pd.DataFrame(FAIL_ROWS).to_csv(os.path.join(dbg_dir, "fail_rows.csv"), index=False)
        except Exception:
            pass
        print("[DEBUG] MF fail counters:", dict(FAIL_COUNTER))
        print(f"[DEBUG] Detalhes: {os.path.join(dbg_dir, 'fail_rows.csv')}")

    _log("\n[OK] MF concluido.")
    _log(f" Saida: {out_dir}")
    _log(f" Templates: {len(tmpl_cache)} | fs={CFG.FS_TARGET} Hz | lag_step={CFG.LAG_STEP} | ±shift={CFG.MAX_SHIFT_SEC}s | GPU={CFG.USE_GPU and _HAS_CUPY}")
    _log(f" Val AUC={auc_val:.4g} AP={ap_val:.4g} | Test AUC={auc_test:.4g} AP={ap_test:.4g}")
    _log(f" Confusion@test_thr: tn={tn} fp={fp} fn={fn} tp={tp}")

if __name__ == "__main__":
    main()