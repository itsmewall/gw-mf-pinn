# src/eval/mf_baseline.py
# --------------------------------------------------------------------------------------
# Matched-filter baseline via NCC com FFT:
# - GPU via CuPy com fallback automatico para CPU
# - Sem alocacoes gigantes: multiplica X_fft (B,F) por 1 template de cada vez
# - Autotuning de batch (janelas) e chunk de templates com backoff em OOM
# - Index de windows por file_id (1x por run) e caches
# - Restricoes de lag (±MAX_SHIFT_SEC) + passo LAG_STEP
# - Banco de templates IMRPhenomD com cache .npz
# - Depuracao forte: contadores e CSV de falhas em _debug/
# - Self-test INJECT
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, math
from glob import glob
from dataclasses import dataclass, asdict
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from typing import Dict, Tuple, Optional, List
from collections import Counter

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
    GPU_INIT_BATCH_WIN: int = 4096         # palpite inicial de janelas por lote
    GPU_MIN_BATCH_WIN: int  = 256
    GPU_INIT_TEMPL_CHUNK: int = 64         # templates por chunk (ajusta em OOM)
    GPU_MIN_TEMPL_CHUNK: int  = 8
    USE_FP16_TIME: bool = False            # manter tempo em fp32 para estabilidade

    # banco de templates
    TEMPLATES_N: int = 256
    TEMPLATE_SUBSAMPLE: int = 1
    M1_RANGE: Tuple[float, float] = (5.0, 40.0)
    M2_RANGE: Tuple[float, float] = (5.0, 40.0)
    SPIN1: float = 0.0
    SPIN2: float = 0.0
    F_LO: float = 20.0
    HP_WINDOW_TAPER: float = 0.1
    CACHE_FILE: str = "templates_cache.npz"

    # lags
    MAX_SHIFT_SEC: float = 0.25
    LAG_STEP: int = 8

    # limites
    MAX_VAL_ROWS: Optional[int] = None
    MAX_TEST_ROWS: Optional[int] = None

    # threshold
    MODE: str = "target_far"
    TARGET_FAR: float = 1e-6

    # saida
    OUT_DIR_ROOT: str = "reports/mf_baseline"

    # CPU
    N_JOBS: int = -1
    BATCH_SZ: int = 256

    # feedback
    HEARTBEAT_EVERY_N: int = 10000
    HEARTBEAT_EVERY_SEC: float = 100.0
    TQDM_MIN_INTERVAL: float = 0.5

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

def maybe_load_template_cache(out_dir: str, bank) -> Dict[str, np.ndarray]:
    cache_path = os.path.join(out_dir, CFG.CACHE_FILE)
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
        return x
    except Exception as e:
        if DEBUG_STRICT:
            FAIL_COUNTER["exception_load"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "exception_load", "msg": repr(e)})
        return None

# =========================
# NCC via FFT CPU
# =========================
def _ncc_fft_max_cpu(x: np.ndarray, h: np.ndarray, max_shift_samp: int, lag_step: int) -> float:
    c = np_fftconvolve(x, h[::-1], mode="full")
    denom = (np.linalg.norm(x) * np.linalg.norm(h)) + 1e-12
    c = c / denom
    L = c.size
    center = L // 2
    lo = max(0, center - max_shift_samp)
    hi = min(L, center + max_shift_samp + 1)
    return float(np.nanmax(c[lo:hi:lag_step]))

# =========================
# GPU batelado sem tensor gigantes
# =========================
def _gpu_autotune(B_init: int, Kc_init: int, nfft: int, n_full: int, F: int) -> Tuple[int, int]:
    """
    Ajusta B (batch janelas) e Kc (chunk de templates) para caber na GPU.
    Estima consumo e faz backoff em caso de OOM.
    """
    B, Kc = B_init, Kc_init
    if not _HAS_CUPY: return B, Kc

    def enough(Bt: int, Kt: int) -> bool:
        try:
            # buffers principais: X_dev(B,L) ~ float32, X_fft(B,F) complex64,
            # c_tmp(B,n_full) float32, e espectro 1 template complex64 (F)
            # alocacao tentativa pequena para validar
            L = n_full // 2 + 1  # aproximacao
            x = cp.empty((Bt, F*2), dtype=cp.float32)  # aprox
            xf= cp.empty((Bt, F), dtype=cp.complex64)
            ct= cp.empty((Bt, n_full), dtype=cp.float32)
            ht= cp.empty((F,), dtype=cp.complex64)
            del x, xf, ct, ht
            cp.get_default_memory_pool().free_all_blocks()
            return True
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            return False

    while B >= CFG.GPU_MIN_BATCH_WIN and not enough(B, Kc):
        B //= 2
    if B < CFG.GPU_MIN_BATCH_WIN:
        B = CFG.GPU_MIN_BATCH_WIN
    # Kc entra só como loop externo; deixamos pelo menos 8
    if Kc < CFG.GPU_MIN_TEMPL_CHUNK:
        Kc = CFG.GPU_MIN_TEMPL_CHUNK
    return B, Kc

def score_subset_gpu_streaming(name: str, df_sub: pd.DataFrame, idx_map: Dict[str,str],
                               tmpl_cache: Dict[str,np.ndarray]) -> np.ndarray:
    n = len(df_sub)
    scores = np.zeros(n, np.float32)

    # templates host
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

    # autotune batch
    B0, Kc0 = _gpu_autotune(CFG.GPU_INIT_BATCH_WIN, CFG.GPU_INIT_TEMPL_CHUNK, nfft, n_full, F)
    B_cur, Kc_cur = B0, Kc0

    idx = df_sub.index.to_list()
    _log(f"[{name}] janelas={n:,} | templates={K} | nfft={nfft} | max_shift={CFG.MAX_SHIFT_SEC:.3f}s | lag_step={lag_step} | GPU streaming")
    _log(f"[{name}] autotune: batch_win={B_cur} | templ_chunk={Kc_cur}")

    t0 = time.time(); done = 0
    with tqdm(total=n, unit="win", mininterval=CFG.TQDM_MIN_INTERVAL, desc=f"[{name}] NCC", leave=True) as bar:
        b0 = 0
        while b0 < n:
            b_idx = idx[b0:b0+B_cur]

            # carrega lote de sinais no host
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
                x = x.astype(np.float32, copy=False)
                x -= x.mean(); x /= (x.std() + 1e-12)
                X_host.append(x)

            mask_valid = np.array([x is not None for x in X_host], dtype=bool)
            if not mask_valid.any():
                done += len(b_idx); bar.update(len(b_idx))
                b0 += B_cur
                continue

            try:
                Xv = np.stack([x for x in X_host if x is not None], axis=0)  # (Bv, L)
                dtype_time = cp.float16 if CFG.USE_FP16_TIME else cp.float32
                X_dev = cp.asarray(Xv, dtype=dtype_time)
                X_fft = cp.fft.rfft(X_dev.astype(cp.float32, copy=False), nfft, axis=1)  # (Bv,F) complex64
                X_norm = cp.linalg.norm(X_dev.astype(cp.float32, copy=False), axis=1).astype(cp.float32, copy=False)
                best = cp.full((X_dev.shape[0],), -cp.inf, dtype=cp.float32)

                # percorre templates em chunks pequenos para manter cache quente e baixar overhead de CPU->GPU
                for t0_idx in range(0, K, Kc_cur):
                    t1_idx = min(t0_idx + Kc_cur, K)
                    # precompute espectros dos templates desse chunk no host e sobe 1 a 1
                    for k in range(t0_idx, t1_idx):
                        h = templates_host[k]
                        h_dev = cp.asarray(h[::-1], dtype=cp.float32)
                        H_fft = cp.fft.rfft(h_dev, nfft)  # (F,) complex64
                        # produto por broadcasting: (Bv,F) * (F,) -> (Bv,F)
                        C_fft = X_fft * cp.conj(H_fft)
                        c = cp.fft.irfft(C_fft, nfft, axis=1)[:, :n_full]  # (Bv,n_full) float32
                        denom = (X_norm) * cp.linalg.norm(h_dev) + 1e-12   # (Bv,)
                        c = c / denom[:, None]
                        # pega max nos lags de interesse
                        m = cp.max(c[:, lo:hi:lag_step], axis=1)            # (Bv,)
                        best = cp.maximum(best, m)

                        # libera
                        del h_dev, H_fft, C_fft, c, m
                        cp.get_default_memory_pool().free_all_blocks()

                best_host = best.get()

                # escreve de volta nas posicoes do dataframe
                pos_valid = np.where(mask_valid)[0]
                for j, val in zip(pos_valid, best_host):
                    out_pos = df_sub.index.get_loc(b_idx[j])
                    scores[out_pos] = float(val)

                done += len(b_idx); bar.update(len(b_idx))
                del X_dev, X_fft, X_norm, best
                cp.get_default_memory_pool().free_all_blocks()
                b0 += B_cur

            except cp.cuda.memory.OutOfMemoryError:
                # backoff agressivo
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
                    raise  # sem como reduzir mais

    _log(f"      [{name}] tempo={time.time()-t0:.1f}s")
    return scores

# =========================
# CPU legado e roteador
# =========================
def _score_one_cpu(fid: str, sgps: float, idx_map: Dict[str,str], max_shift: int,
                   lag_step: int, templates_host: List[np.ndarray]) -> float:
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

        best = 0.0
        for h in templates_host:
            s = _ncc_fft_max_cpu(x, h, max_shift_samp=max_shift, lag_step=lag_step)
            if s > best: best = s
        return best
    except Exception as e:
        if DEBUG_STRICT:
            FAIL_COUNTER["exception_score"] += 1
            FAIL_ROWS.append({"file_id": os.path.basename(win_path),
                              "reason": "exception_score", "msg": repr(e)})
        return 0.0

def score_subset_cpu(name: str, df_sub: pd.DataFrame, idx_map: Dict[str,str],
                     tmpl_cache: Dict[str,np.ndarray]) -> np.ndarray:
    keys = list(tmpl_cache.keys())
    if CFG.TEMPLATE_SUBSAMPLE > 1:
        keys = keys[::CFG.TEMPLATE_SUBSAMPLE]
    templates_host = [tmpl_cache[k] for k in keys]

    fs = CFG.FS_TARGET
    max_shift = int(round(CFG.MAX_SHIFT_SEC * fs))
    lag_step  = max(1, int(CFG.LAG_STEP))

    n = len(df_sub)
    scores = np.zeros(n, np.float32)

    _log(f"[{name}] janelas={n:,} | templates={len(templates_host)} | max_shift={CFG.MAX_SHIFT_SEC:.3f}s | lag_step={lag_step} | jobs={CFG.N_JOBS} | GPU=False")

    idx = df_sub.index.to_list()
    batches = [idx[i:i+CFG.BATCH_SZ] for i in range(0, n, CFG.BATCH_SZ)]

    t0 = time.time(); last_hb = t0; done = 0
    with tqdm(total=n, unit="win", mininterval=CFG.TQDM_MIN_INTERVAL, desc=f"[{name}] NCC", leave=True) as bar:
        for b in batches:
            out_final = Parallel(n_jobs=CFG.N_JOBS, backend="loky")(
                delayed(_score_one_cpu)(
                    df_sub.at[i, "file_id"],
                    float(df_sub.at[i, "start_gps"]),
                    idx_map, max_shift, lag_step,
                    templates_host
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
def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if CFG.MODE == "target_far":
        neg = scores[y_true == 0]
        if neg.size == 0:
            return 0.0, {"mode": "target_far", "far": CFG.TARGET_FAR, "note": "no_neg"}
        q = 1.0 - CFG.TARGET_FAR
        thr = float(np.quantile(neg, q)) if neg.size > 10 else float(np.max(neg))
        return thr, {"mode": "target_far", "far": CFG.TARGET_FAR}

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
        plt.plot([0,1],[0,1],'k--',alpha=.4)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
        plt.legend(); plt.grid(alpha=.2); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{prefix}.png"), dpi=140); plt.close()
    except Exception: pass

    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
        plt.legend(); plt.grid(alpha=.2); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_{prefix}.png"), dpi=140); plt.close()
    except Exception: pass

    try:
        plt.figure(figsize=(6.2,3.6))
        plt.hist(scores[y_true==0], bins=60, alpha=.7, label="neg", density=True)
        plt.hist(scores[y_true==1], bins=60, alpha=.7, label="pos", density=True)
        plt.legend(); plt.grid(alpha=.2); plt.title(f"Score dist - {prefix}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"hist_{prefix}.png"), dpi=140); plt.close()
    except Exception: pass

    return auc, ap

def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float):
    yhat = (scores >= thr).astype(int)
    tn = int(np.sum((yhat==0) & (y_true==0)))
    fp = int(np.sum((yhat==1) & (y_true==0)))
    fn = int(np.sum((yhat==0) & (y_true==1)))
    tp = int(np.sum((yhat==1) & (y_true==1)))
    return tn, fp, fn, tp

# =========================
# MAIN
# =========================
def main():
    t_all = time.time()
    out_dir = os.path.join(CFG.OUT_DIR_ROOT, ts_tag()); ensure_dir(out_dir)

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
    _log(f"[GPU] solicitado={CFG.USE_GPU} | disponivel={_HAS_CUPY}" + (f" | nota={_CUPY_ERR}" if (_CUPY_ERR and not _HAS_CUPY) else ""))

    # 2) Index de windows e templates
    _log("[2/6] Indexando *_windows.hdf5 ...")
    idx_map = build_windows_index(CFG.WINDOWS_DIR)
    if not idx_map: raise RuntimeError("nenhum *_windows.hdf5 encontrado em WINDOWS_DIR")

    _log("[2/6] Templates ...")
    bank = build_template_bank(CFG.TEMPLATES_N)
    tmpl_cache = maybe_load_template_cache(out_dir, bank)
    total_templates = len(tmpl_cache)

    # Self-test
    try:
        k0 = next(iter(tmpl_cache.keys()))
        h  = tmpl_cache[k0]
        any_fid = next(iter(idx_map.keys()))
        any_win = idx_map[any_fid]
        sgps, _, meta_in = cached_start_arrays(any_win)
        if len(sgps) > 10:
            test_gps = float(sgps[10])
            wht = resolve_whitened_path(any_win, meta_in)
            xr  = load_window_slice(any_win, wht, test_gps)
            if xr is not None:
                xr = (xr - xr.mean()) / (xr.std() + 1e-12)
                inj = xr + 0.5 * h[:len(xr)]
                inj = (inj - inj.mean()) / (inj.std() + 1e-12)
                peak = _ncc_fft_max_cpu(inj, h, max_shift_samp=int(CFG.MAX_SHIFT_SEC*CFG.FS_TARGET),
                                        lag_step=max(1, CFG.LAG_STEP))
                print(f"[DEBUG] Self-test INJECT (CPU): {peak:.3f} (esperado >= 0.1)")
            else:
                print("[DEBUG] Self-test INJECT pulado: janela real nao carregada.")
        else:
            print("[DEBUG] Self-test INJECT pulado: sgps insuficiente.")
    except Exception as e:
        print(f"[DEBUG] Self-test INJECT falhou: {e}")

    # 3) VAL
    _log("[3/6] Scoring VAL ...")
    s_val = score_subset("VAL", df_val, idx_map, tmpl_cache)
    print(f"[VAL] scores: min={s_val.min():.4g} med={np.median(s_val):.4g} max={s_val.max():.4g} | frac(>1e-6)={(s_val>1e-6).mean()*100:.3f}%")

    # 4) Thresholds
    _log("[4/6] Thresholds no VAL ...")
    y_val = df_val["label"].to_numpy(int)
    thr, info = pick_threshold(y_val, s_val)
    auc_val, ap_val = summarize_and_plot(y_val, s_val, out_dir, "val")
    df_val_out = df_val.copy(); df_val_out["score"] = s_val
    df_val_out.to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)

    # 5) TEST
    _log("[5/6] Scoring TEST ...")
    s_test = score_subset("TEST", df_test, idx_map, tmpl_cache)
    print(f"[TEST] scores: min={s_test.min():.4g} med={np.median(s_test):.4g} max={s_test.max():.4g} | frac(>1e-6)={(s_test>1e-6).mean()*100:.3f}%")
    y_test = df_test["label"].to_numpy(int)
    auc_test, ap_test = summarize_and_plot(y_test, s_test, out_dir, "test")
    df_test_out = df_test.copy(); df_test_out["score"] = s_test
    df_test_out.to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)

    # 6) Relatorio
    _log("[6/6] Salvando relatorio ...")
    tn, fp, fn, tp = confusion_at_threshold(y_test, s_test, thr)
    summary = {
        "cfg": asdict(CFG),
        "templates_cached": int(total_templates),
        "val": {"auc": float(auc_val), "ap": float(ap_val)},
        "test": {"auc": float(auc_test), "ap": float(ap_test),
                 "confusion_at_thr": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}},
        "threshold": float(thr),
        "threshold_info": info,
        "out_dir": out_dir,
        "gpu": {"requested": bool(CFG.USE_GPU), "available": bool(_HAS_CUPY), "note": _CUPY_ERR}
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

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
    _log(f" Templates: {total_templates} | fs={CFG.FS_TARGET} Hz | lag_step={CFG.LAG_STEP} | ±shift={CFG.MAX_SHIFT_SEC}s | GPU={CFG.USE_GPU and _HAS_CUPY}")
    _log(f" Val AUC={auc_val:.4g} AP={ap_val:.4g} | Test AUC={auc_test:.4g} AP={ap_test:.4g}")
    _log(f" Confusion@test_thr: tn={tn} fp={fp} fn={fn} tp={tp}")
    _log(f" Tempo total: {time.time()-t_all:.1f}s")

if __name__ == "__main__":
    main()
