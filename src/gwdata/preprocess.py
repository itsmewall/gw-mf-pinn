"""
preprocess.py
-------------
Pré-processamento dos sinais GW:
 - Bandpass (ex.: 35–350 Hz)
 - Notch lines (ex.: 60 Hz e harmônicos)
 - PSD (Welch) e Whitening
 - Salvamento em HDF5 padronizado em data/interim/

Uso:
  python -m gwdata.preprocess \
    --in_dir data/raw \
    --out_dir data/interim \
    --lowcut 35 --highcut 350 \
    --fs 4096 \
    --notch 60 --notch-harmonics 5 \
    --psd-seg 4 --psd-overlap 0.5

ou usando YAML:
  python -m gwdata.preprocess --config configs/data.yaml

Requisitos: numpy, scipy, h5py, pyyaml (opcional), tqdm
"""

from __future__ import annotations
import os
import glob
import math
import argparse
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import h5py
from scipy.signal import butter, filtfilt, iirnotch, welch
from tqdm import tqdm

from src.accel.whiten import whiten_block

try:
    import yaml  # opcional, só se usar --config
    HAS_YAML = True
except Exception:
    HAS_YAML = False


# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("preprocess")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)


# ---------------------------
# Filtros
# ---------------------------
def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if not (0 < low < 1) or not (0 < high < 1) or not (low < high):
        raise ValueError(f"Bandpass inválido: low={lowcut}, high={highcut}, fs={fs}")
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(x: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 6) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt = zero-phase
    return filtfilt(b, a, x, method="gust")


def apply_notch_lines(x: np.ndarray, fs: float, base_hz: float = 60.0, harmonics: int = 0, Q: float = 30.0) -> np.ndarray:
    """
    Remove linha de 60 Hz e harmônicos (ex.: 120, 180, ...).
    """
    y = x.copy()
    for k in range(1, harmonics + 1):
        f0 = k * base_hz
        if f0 >= fs / 2:
            break
        b, a = iirnotch(w0=f0 / (fs / 2), Q=Q)
        y = filtfilt(b, a, y)
    return y


# ---------------------------
# PSD & Whitening
# ---------------------------
def estimate_psd(x: np.ndarray, fs: float, seg_sec: float = 4.0, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    PSD via Welch.
    seg_sec: tamanho da janela (s)
    overlap: fração de sobreposição (0-1)
    """
    nperseg = int(seg_sec * fs)
    noverlap = int(nperseg * overlap)
    if nperseg <= 0:
        raise ValueError("nperseg <= 0")
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return f, Pxx


def whiten(x: np.ndarray, fs: float, f_psd: np.ndarray, Pxx: np.ndarray) -> np.ndarray:
    """
    Whitening no domínio da frequência: X(f)/sqrt(PSD(f)).
    Implementação simples usando FFT com interpolação da PSD.
    """
    n = len(x)
    # FFT de x
    Xf = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # PSD -> mesma grade de freqs da FFT
    # Evita zeros na PSD
    Pxx_safe = np.maximum(Pxx, np.finfo(float).eps)
    Pxx_interp = np.interp(freqs, f_psd, Pxx_safe)

    # Whitening
    Xw = Xf / np.sqrt(Pxx_interp / 2.0)  # /2 por convenção unilateral
    xw = np.fft.irfft(Xw, n=n)
    return xw


# ---------------------------
# IO helpers
# ---------------------------
def read_strain_hdf5(path: str) -> Tuple[np.ndarray, float, Dict]:
    """
    Lê /strain/Strain de um arquivo HDF5 GWOSC.
    Retorna: (strain, fs, meta_dict)
    """
    with h5py.File(path, "r") as f:
        dset = f["strain"]["Strain"]
        strain = dset[()]  # ndarray
        dt = dset.attrs.get("Xspacing", None)
        if dt is None or dt <= 0:
            raise ValueError("Atributo Xspacing ausente ou inválido.")
        fs = 1.0 / dt

        meta = {
            "detector": dset.attrs.get("Detector", "unknown"),
            "xspacing": float(dt),
            "xstart": float(dset.attrs.get("Xstart", 0.0)),
            "units": str(dset.attrs.get("Units", "strain")),
            "length": int(len(strain)),
        }
    return strain, fs, meta


def write_interim_hdf5(path: str,
                       strain_w: np.ndarray,
                       fs: float,
                       meta_in: Dict,
                       band: Tuple[float, float],
                       notch: Dict,
                       psd_info: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("strain")
        d = g.create_dataset("StrainWhitened", data=strain_w, dtype="f8")
        d.attrs["fs"] = fs
        d.attrs["bandpass_low"] = band[0]
        d.attrs["bandpass_high"] = band[1]
        d.attrs["notch_base"] = notch.get("base_hz", 60.0)
        d.attrs["notch_harmonics"] = notch.get("harmonics", 0)

        # metadados de entrada
        m = f.create_group("meta")
        for k, v in meta_in.items():
            m.attrs[k] = v

        # PSD info resumida
        p = f.create_group("psd")
        p.attrs["seg_sec"] = psd_info.get("seg_sec", 4.0)
        p.attrs["overlap"] = psd_info.get("overlap", 0.5)


# ---------------------------
# Pipeline por arquivo
# ---------------------------
def process_file(in_path: str,
                 out_dir: str,
                 lowcut: float,
                 highcut: float,
                 fs_target: Optional[float],
                 notch_base: float,
                 notch_harmonics: int,
                 psd_seg: float,
                 psd_overlap: float,
                 order_bp: int = 6) -> Optional[str]:
    """
    Executa pipeline completo num arquivo HDF5 do GWOSC:
      - leitura
      - (opcional) checagem de fs
      - bandpass
      - notch
      - PSD (Welch)
      - whitening
      - escrita em data/interim/
    """
    try:
        x, fs, meta = read_strain_hdf5(in_path)
        if fs_target and abs(fs - fs_target) > 1e-6:
            logger.warning(f"fs do arquivo ({fs}) difere de fs_target ({fs_target}). "
                           f"Reamostragem não implementada neste passo — usando fs original.")

        # Bandpass
        xb = apply_bandpass(x, fs=fs, lowcut=lowcut, highcut=highcut, order=order_bp)
        # Notch lines
        xn = apply_notch_lines(xb, fs=fs, base_hz=notch_base, harmonics=notch_harmonics, Q=30.0)
        # PSD & Whitening
        xw = whiten_block(xn, fs=int(fs), psd_sec=psd_seg, use_gpu=True)

        # Nome de saída
        base = os.path.basename(in_path)
        out_name = base.replace(".hdf5", "_whitened.hdf5")
        out_path = os.path.join(out_dir, out_name)

        write_interim_hdf5(
            out_path,
            strain_w=xw,
            fs=fs,
            meta_in=meta,
            band=(lowcut, highcut),
            notch={"base_hz": notch_base, "harmonics": notch_harmonics},
            psd_info={"seg_sec": psd_seg, "overlap": psd_overlap},
        )
        return out_path

    except Exception as e:
        logger.error(f"Falha ao processar {in_path}: {e}")
        return None


# ---------------------------
# Config & CLI
# ---------------------------
def load_config(path: Optional[str]) -> Dict:
    cfg: Dict = {}
    if path:
        if not HAS_YAML:
            raise RuntimeError("PyYAML não instalado. Instale com `pip install pyyaml` ou use flags da CLI.")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Defaults
    defaults = {
        "paths": {"in_dir": "data/raw", "out_dir": "data/interim"},
        "filters": {"lowcut": 35.0, "highcut": 350.0, "order": 6},
        "sampling": {"fs_target": None},
        "notch": {"base_hz": 60.0, "harmonics": 0},
        "psd": {"seg_sec": 4.0, "overlap": 0.5},
    }

    # Merge shallow (suficiente aqui)
    def _merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    return _merge(defaults, cfg)


def main():
    ap = argparse.ArgumentParser(description="Pré-processamento de sinais GW (bandpass, notch, whitening).")
    ap.add_argument("--config", type=str, default=None, help="YAML com parâmetros (opcional).")
    ap.add_argument("--in_dir", type=str, default=None, help="Diretório com HDF5 brutos (sobrescreve config).")
    ap.add_argument("--out_dir", type=str, default=None, help="Diretório de saída (sobrescreve config).")
    ap.add_argument("--lowcut", type=float, default=None, help="Freq. baixa do bandpass (Hz).")
    ap.add_argument("--highcut", type=float, default=None, help="Freq. alta do bandpass (Hz).")
    ap.add_argument("--fs", type=float, default=None, help="Fs alvo (Hz) — apenas checagem nesta etapa.")
    ap.add_argument("--notch", type=float, default=None, help="Frequência base das notch lines (Hz), ex.: 60.")
    ap.add_argument("--notch-harmonics", type=int, default=None, help="Qtde de harmônicos para notch (0 desliga).")
    ap.add_argument("--psd-seg", type=float, default=None, help="Janela (s) do Welch.")
    ap.add_argument("--psd-overlap", type=float, default=None, help="Sobreposição (0–1) do Welch.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Overrides pela CLI
    if args.in_dir: cfg["paths"]["in_dir"] = args.in_dir
    if args.out_dir: cfg["paths"]["out_dir"] = args.out_dir
    if args.lowcut is not None: cfg["filters"]["lowcut"] = args.lowcut
    if args.highcut is not None: cfg["filters"]["highcut"] = args.highcut
    if args.fs is not None: cfg["sampling"]["fs_target"] = args.fs
    if args.notch is not None: cfg["notch"]["base_hz"] = args.notch
    if args.notch_harmonics is not None: cfg["notch"]["harmonics"] = args.notch_harmonics
    if args.psd_seg is not None: cfg["psd"]["seg_sec"] = args.psd_seg
    if args.psd_overlap is not None: cfg["psd"]["overlap"] = args.psd_overlap

    in_dir = cfg["paths"]["in_dir"]
    out_dir = cfg["paths"]["out_dir"]
    lowcut = float(cfg["filters"]["lowcut"])
    highcut = float(cfg["filters"]["highcut"])
    order_bp = int(cfg["filters"]["order"])
    fs_target = cfg["sampling"]["fs_target"]
    notch_base = float(cfg["notch"]["base_hz"])
    notch_harm = int(cfg["notch"]["harmonics"])
    psd_seg = float(cfg["psd"]["seg_sec"])
    psd_ovl = float(cfg["psd"]["overlap"])

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.hdf5")))
    if not files:
        logger.warning(f"Nenhum HDF5 encontrado em {in_dir}.")
        return

    logger.info(f"Arquivos encontrados: {len(files)}")
    logger.info(f"Bandpass: {lowcut}-{highcut} Hz | Notch: {notch_base} Hz x {notch_harm} harmônicos | PSD: {psd_seg}s/{psd_ovl}")
    logger.info(f"Saída: {out_dir}")

    saved: List[str] = []
    for fpath in tqdm(files, desc="Pré-processando"):
        out = process_file(
            in_path=fpath,
            out_dir=out_dir,
            lowcut=lowcut,
            highcut=highcut,
            fs_target=fs_target,
            notch_base=notch_base,
            notch_harmonics=notch_harm,
            psd_seg=psd_seg,
            psd_overlap=psd_ovl,
            order_bp=order_bp,
        )
        if out:
            saved.append(out)

    logger.info("Concluído.")
    if saved:
        logger.info("Arquivos gerados:")
        for p in saved:
            logger.info(f" - {p}")


if __name__ == "__main__":
    main()
