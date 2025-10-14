# src/eval/mf_psd.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import welch, inverse_spectrum_truncation
from pycbc.filter import matched_filter, chisq
from pycbc.events.ranking import newsnr
from pycbc.waveform import get_fd_waveform

@dataclass(frozen=True)
class MFConfig:
    f_low: float = 35.0
    psd_seg: float = 4.0         # segundos
    psd_overlap: float = 0.5     # fração
    chisq_bins: int = 16
    max_shift_sec: float = 0.02  # 20 ms de cada lado
    taper: str = "tukey"         # coerente com o que aplicar no sinal

def _estimate_psd(x: np.ndarray, fs: int, cfg: MFConfig) -> FrequencySeries:
    ts = TimeSeries(x, delta_t=1.0/fs)
    seg = int(cfg.psd_seg * fs)
    stride = int(cfg.psd_seg * (1.0 - cfg.psd_overlap) * fs)
    psd = welch(ts, seg_len=seg, seg_stride=stride, window='hann')
    psd = inverse_spectrum_truncation(psd, seg, low_frequency_cutoff=cfg.f_low)
    return psd

def _template_fd(m1, m2, s1z, s2z, n, fs, f_low) -> FrequencySeries:
    df = fs / n
    hp_fd, _ = get_fd_waveform(approximant="IMRPhenomD",
                               mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z,
                               delta_f=df, f_lower=f_low)
    # recorta ao Nyquist
    return hp_fd.crop_f0fmax(f_low, fs/2.0)

def reweighted_snr_peak(x: np.ndarray,
                        fs: int,
                        template: Tuple[float,float,float,float],
                        cfg: MFConfig = MFConfig()) -> Tuple[float, float]:
    """Retorna (new_snr_peak, snr_peak) no entorno de ±max_shift_sec."""
    assert np.isfinite(x).all(), "Entrada contém NaN/Inf"
    n = len(x)
    ts = TimeSeries(x, delta_t=1.0/fs)
    psd = _estimate_psd(x, fs, cfg)
    hp_fd = _template_fd(*template, n=n, fs=fs, f_low=cfg.f_low)

    snr_ts = matched_filter(hp_fd, ts, psd=psd,
                            low_frequency_cutoff=cfg.f_low)
    # χ² por bandas
    redx2 = chisq(ts, hp_fd, snr_ts, psd, cfg.chisq_bins,
                  low_frequency_cutoff=cfg.f_low).reduced_chisq
    # newSNR
    new_snr = newsnr(snr_ts.numpy(), redx2.numpy())
    new_snr = TimeSeries(new_snr, delta_t=snr_ts.delta_t)

    # pico restrito em ±max_shift_sec
    half = int(cfg.max_shift_sec * fs)
    mid = int(len(new_snr) // 2)
    lo, hi = max(0, mid - half), min(len(new_snr), mid + half + 1)
    peak_rw = float(np.max(np.abs(new_snr[lo:hi].numpy())))
    peak_snr = float(np.max(np.abs(snr_ts[lo:hi].numpy())))
    return peak_rw, peak_snr
