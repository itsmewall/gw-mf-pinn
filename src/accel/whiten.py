# src/accel/whiten.py
from __future__ import annotations
import numpy as np

_HAS_CUPY = False
try:
    import cupy as cp
    from cupyx.scipy.signal import welch as cp_welch
    _HAS_CUPY = (cp.cuda.runtime.getDeviceCount() > 0)
except Exception:
    _HAS_CUPY = False

from scipy.signal import welch as np_welch

def _interp_gpu(x, xp, fp):
    # interp1d simples em GPU (linear) usando searchsorted
    idx = cp.searchsorted(xp, x, side='left')
    idx = cp.clip(idx, 1, xp.size - 1)
    x0 = xp[idx - 1]; x1 = xp[idx]
    y0 = fp[idx - 1]; y1 = fp[idx]
    w = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)

def whiten_block(x: np.ndarray, fs: int, psd_sec: float, use_gpu: bool = True) -> np.ndarray:
    """
    Whitening por divisão espectral: FFT(x) / sqrt(PSD(f)).
    - PSD via Welch com nperseg = psd_sec*fs
    - GPU (CuPy) se disponível, senão CPU (NumPy/SciPy).
    Retorna float32 no host.
    """
    x = np.asarray(x, dtype=np.float32)
    nper = max(8, int(psd_sec * fs))
    if use_gpu and _HAS_CUPY:
        x_d = cp.asarray(x, dtype=cp.float32)
        f, Pxx = cp_welch(x_d, fs=fs, nperseg=nper)
        Pxx = cp.maximum(Pxx, cp.median(Pxx) * 1e-6)
        Xf = cp.fft.rfft(x_d)
        fx = cp.fft.rfftfreq(x_d.size, d=1.0 / fs)
        S  = cp.sqrt(_interp_gpu(fx, f, Pxx))
        Yf = Xf / (S + 1e-12)
        y  = cp.fft.irfft(Yf, n=x_d.size)
        out = cp.asnumpy(y).astype(np.float32, copy=False)
        # libera VRAM
        del x_d, f, Pxx, Xf, fx, S, Yf, y
        cp.get_default_memory_pool().free_all_blocks()
        return out
    # CPU
    f, Pxx = np_welch(x, fs=fs, nperseg=nper)
    Pxx = np.maximum(Pxx, np.median(Pxx) * 1e-6)
    Xf = np.fft.rfft(x)
    fx = np.fft.rfftfreq(x.size, d=1.0 / fs)
    S  = np.sqrt(np.interp(fx, f, Pxx))
    y  = np.fft.irfft(Xf / (S + 1e-12), n=x.size)
    return y.astype(np.float32, copy=False)
