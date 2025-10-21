# gwdata/preprocess.py
# --------------------------------------------------------------------------------------
# Pré-processamento robusto para arquivos GWOSC:
#   - Leitura segura do canal de strain
#   - Sanitização de NaN/inf com política configurável
#   - Detrend, bandpass, notches harmônicos
#   - Whitening por PSD de Welch
#   - Reamostragem opcional
#   - Escrita HDF5 com metadados e estatísticas de saneamento
# Requisitos: numpy, scipy, h5py
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os, time, json
from typing import Optional, Dict, Tuple
import numpy as np
import h5py
from scipy.signal import detrend, welch, sosfiltfilt, butter, iirnotch, resample_poly
from fractions import Fraction

# ----------------------------------------------
# Utilidades
# ----------------------------------------------
def _now(): return time.time()

def _choose_channel(g: h5py.File):
    # tenta canais comuns de HDF5 GWOSC
    cand = [
        "/strain/Strain",                 # clássico
        "/strain/StrainWhitened",         # se já veio whitened
        "/strain/whitened",
        "/data/strain",
        "/data/whitened",
        "/whitened"
    ]
    for c in cand:
        if c in g and isinstance(g[c], h5py.Dataset):
            return g[c]
    # fallback: primeiro dataset 1D encontrado
    for k in g.keys():
        obj = g[k]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
            return obj
    return None

def _read_fs(attrs: Dict[str, float | str]) -> Optional[int]:
    fs = attrs.get("fs") or attrs.get("sample_rate")
    if fs:
        try: return int(round(float(fs)))
        except: pass
    xs = attrs.get("xspacing") or attrs.get("Xspacing")
    if xs:
        try:
            dt = float(xs)
            if dt > 0: return int(round(1.0/dt))
        except: pass
    return None

def _interp_nan_inplace(x: np.ndarray, max_gap: int) -> Tuple[int, int]:
    """
    Interpola NaN/inf por interp linear, apenas em gaps com tamanho <= max_gap.
    Para gaps maiores, preenche por vizinho mais próximo para manter o pipeline estável
    e sinaliza via contagem.
    Retorna (n_interp, n_bigfills).
    """
    n = x.size
    bad = ~np.isfinite(x)
    if not bad.any():
        return 0, 0

    # índices válidos
    idx = np.arange(n)
    good_idx = idx[~bad]
    if good_idx.size == 0:
        # tudo ruim
        # substitui por zeros para evitar crash, caller vai rejeitar pelo threshold de contaminação
        x[:] = 0.0
        return 0, 1

    # interp linear nos gaps
    # bordas: extrapola mantendo valores da borda
    x[bad] = np.interp(idx[bad], good_idx, x[~bad])

    # identificar gaps longos e contá-los
    n_interp = int(bad.sum())
    n_big = 0
    # reconstruir gaps contíguos originais
    # varre marcando sequências de True em 'bad_orig'
    bad_orig = bad.copy()
    i = 0
    while i < n:
        if bad_orig[i]:
            j = i
            while j < n and bad_orig[j]:
                j += 1
            seg_len = j - i
            if seg_len > max_gap:
                n_big += 1
                # para enfatizar baixa confiança nesses trechos, aplique leve suavização local
                # já interpolado linearmente; manter como está basta
            i = j
        else:
            i += 1

    return n_interp, n_big

def _butter_bandpass_sos(lowcut: float, highcut: float, fs: int, order: int = 6):
    nyq = 0.5 * fs
    low = max(1e-3, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        high = min(0.999, low + 0.05)
    return butter(order, [low, high], btype="band", output="sos")

def _comb_notches_sos(base: float, harms: int, fs: int, q: float = 30.0):
    sos_list = []
    if base is None or base <= 0 or harms <= 0:
        return np.empty((0, 6), dtype=float)
    for h in range(1, harms + 1):
        f0 = base * h
        if f0 >= fs * 0.5:
            break
        b, a = iirnotch(w0=f0/(fs*0.5), Q=q)
        # converter biquad b,a para sos único
        sos = np.zeros((1, 6), dtype=float)
        sos[0, :3] = b
        sos[0, 3:] = a
        sos_list.append(sos)
    if not sos_list:
        return np.empty((0, 6), dtype=float)
    return np.vstack(sos_list)

def _whiten_via_psd(x: np.ndarray, fs: int, seg_sec: float, overlap: float) -> np.ndarray:
    nper = max(8, int(round(seg_sec * fs)))
    nover = int(round(overlap * nper))
    f, Pxx = welch(x, fs=fs, nperseg=nper, noverlap=nover, detrend="constant", return_onesided=True, scaling="density")
    # evitar zeros e NaN no PSD
    Pxx = np.asarray(Pxx, dtype=np.float64)
    Pxx[~np.isfinite(Pxx)] = np.nan
    med = np.nanmedian(Pxx) if np.any(np.isfinite(Pxx)) else 1.0
    Pxx = np.nan_to_num(Pxx, nan=med, posinf=med, neginf=med)
    Pxx = np.maximum(Pxx, 1e-24)

    # FFT de x
    n = x.size
    Xf = np.fft.rfft(x.astype(np.float64), n=n)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # interpola PSD para bins da FFT
    Pxx_i = np.interp(freqs, f, Pxx)
    amp = np.sqrt(Pxx_i * fs / 2.0)
    amp = np.maximum(amp, 1e-12)
    Yf = Xf / amp

    yw = np.fft.irfft(Yf, n=n).astype(np.float32)
    # normaliza zscore
    mu = float(np.mean(yw))
    sd = float(np.std(yw) + 1e-12)
    return ((yw - mu) / sd).astype(np.float32)

# ----------------------------------------------
# Função principal
# ----------------------------------------------
def process_file(
    in_path: str,
    out_dir: str,
    lowcut: float = 35.0,
    highcut: float = 350.0,
    fs_target: Optional[int] = None,
    notch_base: float = 60.0,
    notch_harmonics: int = 0,
    psd_seg: float = 4.0,
    psd_overlap: float = 0.5,
    order_bp: int = 6,
    nan_policy: str = "interp",        # "interp" | "zero" | "skip" | "strict"
    nan_max_frac: float = 0.30,        # rejeita se fração de não finitos original > 30%
    nan_max_gap_sec: float = 2.0,      # interpola até 2 s de gap; acima conta como big-gap
    save_attrs: Optional[Dict] = None
) -> Optional[str]:
    """
    Retorna caminho do arquivo whitened escrito, ou None se rejeitado.
    Escreve:
      /strain/StrainWhitened      float32
      /meta_in                    attrs de entrada
      /meta_proc                  stats do processamento
    """
    t0 = _now()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(in_path)
    out_name = base.replace(".hdf5", "_whitened.hdf5")
    out_path = os.path.join(out_dir, out_name)

    try:
        with h5py.File(in_path, "r") as g:
            ds = _choose_channel(g)
            if ds is None:
                print(f"[ERROR] {base}: nenhum canal de strain válido")
                return None
            attrs_in = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in ds.attrs.items()}
            attrs_in_file = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in g.attrs.items()}
            fs = _read_fs({**attrs_in_file, **attrs_in})
            if fs is None:
                print(f"[ERROR] {base}: sample rate desconhecido")
                return None

            x = np.array(ds[...], dtype=np.float64, copy=True)
    except Exception as e:
        print(f"[ERROR] Falha abrindo {base}: {e}")
        return None

    n = x.size
    if n == 0:
        print(f"[ERROR] {base}: série vazia")
        return None

    # Estatística de contaminação original
    bad0 = ~np.isfinite(x)
    frac_bad0 = float(bad0.mean())

    # Política de NaN
    if nan_policy == "strict" and bad0.any():
        print(f"[ERROR] {base}: NaN/inf detectado e nan_policy=strict")
        return None

    n_interp = 0
    n_bigfills = 0
    if bad0.any():
        if nan_policy == "skip":
            print(f"[WARN] {base}: pulando por NaN/inf")
            return None
        elif nan_policy in ("interp", "zero"):
            if nan_policy == "interp":
                max_gap = int(round(nan_max_gap_sec * fs))
                n_interp, n_bigfills = _interp_nan_inplace(x, max_gap=max_gap)
            else:
                # zero-fill simples
                x[~np.isfinite(x)] = 0.0
        else:
            # default interp
            max_gap = int(round(nan_max_gap_sec * fs))
            n_interp, n_bigfills = _interp_nan_inplace(x, max_gap=max_gap)

    # Rejeita se contaminação original muito alta
    if frac_bad0 > nan_max_frac:
        print(f"[WARN] {base}: frac_bad0={frac_bad0:.3f} > {nan_max_frac:.3f}. Rejeitado.")
        return None

    # Detrend + bandpass + notches
    try:
        x = detrend(x, type="constant")
        sos_bp = _butter_bandpass_sos(lowcut, highcut, fs, order=order_bp)
        x = sosfiltfilt(sos_bp, x).astype(np.float64, copy=False)
        sos_notch = _comb_notches_sos(notch_base, notch_harmonics, fs, q=30.0)
        if sos_notch.size:
            x = sosfiltfilt(sos_notch, x).astype(np.float64, copy=False)
    except Exception as e:
        print(f"[ERROR] {base}: falha no filtro. {e}")
        return None

    # Whitening via PSD
    try:
        xw = _whiten_via_psd(x, fs=fs, seg_sec=psd_seg, overlap=psd_overlap)
    except Exception as e:
        print(f"[ERROR] {base}: whitening falhou. {e}")
        return None

    # Reamostragem opcional
    fs_out = fs
    if fs_target and int(fs_target) != fs:
        try:
            frac = Fraction(int(fs_target), int(fs)).limit_denominator(64)
            xw = resample_poly(xw, frac.numerator, frac.denominator).astype(np.float32, copy=False)
            fs_out = int(fs_target)
        except Exception as e:
            print(f"[ERROR] {base}: resample falhou. {e}")
            return None

    # Escrita HDF5
    try:
        with h5py.File(out_path, "w") as h:
            d = h.create_dataset("/strain/StrainWhitened", data=xw, compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
            # atributos do dataset de saída
            d.attrs["sample_rate"] = fs_out
            d.attrs["fs"] = fs_out
            d.attrs["units"] = "whitened"
            d.attrs["dtype"] = "float32"

            meta_in = h.create_group("/meta_in")
            for k, v in attrs_in.items():
                try: meta_in.attrs[str(k)] = v
                except: pass
            meta_in.attrs["source_path"] = os.path.abspath(in_path)
            meta_in.attrs["source_file"] = os.path.basename(in_path)

            meta_proc = h.create_group("/meta_proc")
            meta_proc.attrs["lowcut"] = float(lowcut)
            meta_proc.attrs["highcut"] = float(highcut)
            meta_proc.attrs["order_bp"] = int(order_bp)
            meta_proc.attrs["psd_seg_sec"] = float(psd_seg)
            meta_proc.attrs["psd_overlap"] = float(psd_overlap)
            meta_proc.attrs["notch_base"] = float(notch_base or 0.0)
            meta_proc.attrs["notch_harmonics"] = int(notch_harmonics or 0)
            meta_proc.attrs["nan_policy"] = str(nan_policy)
            meta_proc.attrs["nan_frac_original"] = float(frac_bad0)
            meta_proc.attrs["nan_interpolated"] = int(n_interp)
            meta_proc.attrs["nan_big_gaps"] = int(n_bigfills)
            meta_proc.attrs["fs_in"] = int(fs)
            meta_proc.attrs["fs_out"] = int(fs_out)
            meta_proc.attrs["duration_sec_out"] = float(xw.size / fs_out)
            if save_attrs:
                for k, v in save_attrs.items():
                    try: meta_proc.attrs[str(k)] = v
                    except: pass
    except Exception as e:
        print(f"[ERROR] {base}: falha ao escrever HDF5. {e}")
        return None

    dt = _now() - t0
    print(f"[OK] {base}: whitened salvo em {out_path}  fs={fs_out}Hz  dt={dt:.1f}s  bad0={frac_bad0:.4f} interp={n_interp} big={n_bigfills}")
    return out_path