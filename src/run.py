"""
run.py
------
Motor único SEM ARGUMENTOS.

Fluxo automático (configurável por toggles):
  1) (opcional) Descobre eventos públicos e baixa arquivos que faltam
  2) (opcional) Pré-processa (bandpass + notch + PSD + whitening) só os novos
  3) (opcional) Gera janelas deslizantes (SNR RMS/Peak) só dos whitened novos
  4) Resumo final

Requisitos (venv):
  pip install requests h5py numpy scipy tqdm pyyaml gwosc
"""

from __future__ import annotations
import os
import sys
import re
import time
import glob
import shutil
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# --- módulos do projeto ---
from gwdata import preprocess as pp
from gwdata.windows import process_whitened_file  # etapa 3: janelas/SNR


# =============================================================================
# TOGGLES — ligue/desligue estágios
# =============================================================================
ENABLE_DOWNLOAD = False     # coloque False para NÃO baixar nada
ENABLE_WHITEN   = False     # coloque False para NÃO fazer whitening
ENABLE_WINDOWS  = True     # coloque False para NÃO gerar janelas/SNR

# =============================================================================
# PERFIS / LIMITES — "initial" ≈ ≤10 GB/rodada
# =============================================================================
PROFILE = "initial"     # "initial" | "extended" | "full"

# Diretórios
DATA_RAW       = "data/raw"
DATA_INTERIM   = "data/interim"
DATA_PROCESSED = "data/processed"
LOG_DIR        = "logs"

# Preferências de arquivo (qualidade x tamanho)
PREFER_HDF5           = True
PREFER_4KHZ           = True
PREFER_DURATION_4096S = True
ALLOW_DETECTORS       = {"H1", "L1"}  # adicione "V1" para incluir Virgo

# Limites por perfil
if PROFILE == "initial":
    MAX_DOWNLOAD_BYTES_PER_RUN = 10 * 1024**3  # 10 GB
    MAX_EVENTS_PER_RUN         = 20
    MAX_FILES_PER_EVENT        = 2             # tipicamente H1 + L1
    MAX_SINGLE_FILE_BYTES      = 1 * 1024**3
elif PROFILE == "extended":
    MAX_DOWNLOAD_BYTES_PER_RUN = 50 * 1024**3
    MAX_EVENTS_PER_RUN         = 100
    MAX_FILES_PER_EVENT        = 4
    MAX_SINGLE_FILE_BYTES      = 2 * 1024**3
else:  # "full"
    MAX_DOWNLOAD_BYTES_PER_RUN = 500 * 1024**3
    MAX_EVENTS_PER_RUN         = 10000
    MAX_FILES_PER_EVENT        = 100
    MAX_SINGLE_FILE_BYTES      = 8 * 1024**3

# Sentinela de parada segura
STOP_SENTINEL = "STOP"

# =============================================================================
# PARÂMETROS DE PRÉ-PROCESSAMENTO (whitening)
# =============================================================================
LOWCUT       = 35.0
HIGHCUT      = 350.0
ORDER_BP     = 6
FS_TARGET    = None       # só checa; sem reamostragem aqui
NOTCH_BASE   = 60.0
NOTCH_HARMS  = 0          # use 5 p/ 60/120/180/240/300 Hz (BR)
PSD_SEG      = 4.0
PSD_OVERLAP  = 0.5

# =============================================================================
# PARÂMETROS DE JANELAMENTO / SNR
# =============================================================================
WINDOW_SEC     = 2.0
STRIDE_SEC     = 0.5
SNR_THRESHOLD  = None     # ex. 6.0 para manter só janelas "fortes"
SAVE_WINDOWS   = False    # True salva dados das janelas (pode pesar)

# =============================================================================
# GWOSC API (v2)
# =============================================================================
GWOSC        = "https://gwosc.org"
EVENTS_API   = f"{GWOSC}/api/v2/event-versions"     # lista eventos
STRAIN_API   = f"{GWOSC}/api/v2/events"             # /<event>/strain-files

# =============================================================================
# LOGGING
# =============================================================================
def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.handlers[:] = [ch, fh]
    return logger

# =============================================================================
# HELPERS
# =============================================================================
def _list_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))

def _basename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path) or "file.hdf5"

def _enough_space(path: str, need_bytes: int) -> bool:
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    return free - need_bytes > 2 * 1024**3  # reserva 2 GB

def _match_detector_from_name(fname: str) -> Optional[str]:
    m = re.match(r"^[A-Z]-([HLV]1)_", os.path.basename(fname))
    return m.group(1) if m else None

def _contains_4khz(fname: str) -> bool:
    return "4KHZ" in os.path.basename(fname).upper()

def _contains_4096s(fname: str) -> bool:
    return "-4096." in os.path.basename(fname)

# =============================================================================
# API v2: TODOS os eventos (paginado)
# =============================================================================
def fetch_all_events() -> List[str]:
    logger = logging.getLogger("run")
    events: List[str] = []
    url = EVENTS_API
    while url:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("results", []):
            name = item.get("name") or item.get("event_name")
            if not name:
                continue
            if "-v" in name:
                name = name.split("-v")[0]
            if name not in events:
                events.append(name)
        url = payload.get("next")
        time.sleep(0.15)
    logger.info(f"Eventos públicos descobertos: {len(events)}")
    return events

# =============================================================================
# API v2: strain-files por evento
# =============================================================================
def fetch_strain_files_for_event(event: str) -> List[Dict]:
    url = f"{STRAIN_API}/{event}/strain-files"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return payload.get("results", [])

def _choose_url_and_name(item: Dict) -> Optional[Tuple[str, str]]:
    url_hdf5 = item.get("hdf5_url")
    url_gwf  = item.get("gwf_url")
    detail   = item.get("detail_url")
    url = url_hdf5 if (PREFER_HDF5 and url_hdf5) else (url_hdf5 or url_gwf)
    if not url and detail:
        try:
            rd = requests.get(detail, timeout=20)
            rd.raise_for_status()
            dj = rd.json()
            url = dj.get("hdf5_url") or dj.get("gwf_url")
        except Exception:
            url = None
    if not url:
        return None
    fname = _basename_from_url(url)
    if PREFER_4KHZ and not _contains_4khz(fname):
        return None
    if PREFER_DURATION_4096S and not _contains_4096s(fname):
        return None
    det = _match_detector_from_name(fname)
    if det and ALLOW_DETECTORS and (det not in ALLOW_DETECTORS):
        return None
    return url, fname

# =============================================================================
# DOWNLOAD CONTROLADO
# =============================================================================
def ensure_event_downloaded(event: str, out_dir: str, budget_state: Dict[str, int]) -> int:
    logger = logging.getLogger("run")
    os.makedirs(out_dir, exist_ok=True)
    items = fetch_strain_files_for_event(event)
    if not items:
        logger.info(f"{event}: nenhum strain-file disponível.")
        return 0

    downloaded = 0
    for item in items:
        if os.path.exists(STOP_SENTINEL):
            logger.warning("STOP detectado — interrompendo downloads com segurança.")
            break
        if downloaded >= MAX_FILES_PER_EVENT:
            logger.info(f"{event}: limite {MAX_FILES_PER_EVENT} atingido.")
            break
        if budget_state["bytes"] >= MAX_DOWNLOAD_BYTES_PER_RUN:
            logger.info("Cota de download desta execução atingida.")
            break

        choice = _choose_url_and_name(item)
        if not choice:
            continue
        url, fname = choice
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            logger.info(f"{event}: já existe → {fname} (pulando)")
            continue

        # HEAD para estimar tamanho
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            head.raise_for_status()
            size = int(head.headers.get("Content-Length", "0")) if head.headers.get("Content-Length") else 0
        except Exception:
            size = 0

        if size and size > MAX_SINGLE_FILE_BYTES:
            logger.info(f"{event}: {fname} ({size/1e9:.2f} GB) > limite de arquivo. Pulando.")
            continue

        need = size if size else 256 * 1024**2
        if (budget_state["bytes"] + need) > MAX_DOWNLOAD_BYTES_PER_RUN:
            logger.info("Ultrapassaria cota desta execução. Parando.")
            break
        if not _enough_space(out_dir, need):
            logger.error("Espaço em disco insuficiente.")
            break

        # Download com progresso
        logger.info(f"{event}: baixando {fname} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None
            with open(fpath, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"↓ {fname}", leave=False
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 14):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if total:
                        pbar.update(len(chunk))

        downloaded += 1
        budget_state["bytes"] += (total or need)

    return downloaded

# =============================================================================
# WHITENING — apenas arquivos novos
# =============================================================================
def whiten_missing(in_dir: str, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    raw_files = _list_files(os.path.join(in_dir, "*.hdf5"))
    saved = []
    for fpath in tqdm(raw_files, desc="Pré-processando (novos)", unit="arq"):
        base = os.path.basename(fpath)
        out_name = base.replace(".hdf5", "_whitened.hdf5")
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            continue
        out = pp.process_file(
            in_path=fpath,
            out_dir=out_dir,
            lowcut=LOWCUT,
            highcut=HIGHCUT,
            fs_target=FS_TARGET,
            notch_base=NOTCH_BASE,
            notch_harmonics=NOTCH_HARMS,
            psd_seg=PSD_SEG,
            psd_overlap=PSD_OVERLAP,
            order_bp=ORDER_BP,
        )
        if out:
            saved.append(out)
    return saved

# =============================================================================
# WINDOWS/SNR — apenas whitened novos
# =============================================================================
def window_missing(interim_dir: str, processed_dir: str) -> List[str]:
    os.makedirs(processed_dir, exist_ok=True)
    done = []
    for wp in _list_files(os.path.join(interim_dir, "*_whitened.hdf5")):
        base = os.path.basename(wp).replace("_whitened.hdf5", "")
        out_name = f"{base}_windows.hdf5"
        out_path = os.path.join(processed_dir, out_name)
        if os.path.exists(out_path):
            continue
        out = process_whitened_file(
            in_path=wp,
            out_dir=processed_dir,
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC,
            snr_threshold=SNR_THRESHOLD,
            save_windows=SAVE_WINDOWS,
        )
        if out:
            done.append(out)
    return done

# =============================================================================
# EXECUÇÃO
# =============================================================================
def main():
    logger = setup_logging()
    # Dirs
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED):
        os.makedirs(d, exist_ok=True)

    logger.info("==== PIPELINE (sem argumentos) ====")
    logger.info(f"Toggles: download={ENABLE_DOWNLOAD} | whiten={ENABLE_WHITEN} | windows={ENABLE_WINDOWS}")
    logger.info(f"PROFILE = {PROFILE}")
    logger.info(f"RAW={DATA_RAW} | INTERIM={DATA_INTERIM} | PROCESSED={DATA_PROCESSED}")
    logger.info(f"Preferências: HDF5={PREFER_HDF5} | 4kHz={PREFER_4KHZ} | 4096s={PREFER_DURATION_4096S} | Detectores={sorted(ALLOW_DETECTORS)}")
    logger.info(f"Limites: MAX_BYTES_RUN={MAX_DOWNLOAD_BYTES_PER_RUN/1e9:.2f} GB | MAX_EVENTS_RUN={MAX_EVENTS_PER_RUN} | MAX_FILES_EVENT={MAX_FILES_PER_EVENT}")
    logger.info(f"Pré-proc: bandpass=({LOWCUT},{HIGHCUT}) ord={ORDER_BP} | notch={NOTCH_BASE}Hz x {NOTCH_HARMS} | PSD={PSD_SEG}s/{PSD_OVERLAP}")
    logger.info(f"Windows: win={WINDOW_SEC}s stride={STRIDE_SEC}s | thr={SNR_THRESHOLD} | save={SAVE_WINDOWS}")
    logger.info("====================================")

    total_downloaded = 0
    whitened_now: List[str] = []
    windows_now: List[str] = []

    # 1) Download controlado
    if ENABLE_DOWNLOAD:
        all_events = fetch_all_events()
        events = all_events[:MAX_EVENTS_PER_RUN] if MAX_EVENTS_PER_RUN else all_events
        logger.info(f"Processando nesta execução até {len(events)} evento(s) para DOWNLOAD.")
        budget = {"bytes": 0}
        for ev in tqdm(events, desc="Eventos (download se faltar)", unit="evt"):
            try:
                total_downloaded += ensure_event_downloaded(ev, DATA_RAW, budget)
                if budget["bytes"] >= MAX_DOWNLOAD_BYTES_PER_RUN or os.path.exists(STOP_SENTINEL):
                    break
            except Exception as e:
                logger.error(f"Falha no evento {ev}: {e}")
    else:
        logger.info("Download desativado (ENABLE_DOWNLOAD=False).")

    # 2) Whitening (novos)
    if ENABLE_WHITEN:
        whitened_now = whiten_missing(DATA_RAW, DATA_INTERIM)
    else:
        logger.info("Whitening desativado (ENABLE_WHITEN=False).")

    # 3) Janelas/SNR (novos)
    if ENABLE_WINDOWS:
        windows_now = window_missing(DATA_INTERIM, DATA_PROCESSED)
    else:
        logger.info("Janelas/SNR desativado (ENABLE_WINDOWS=False).")

    # 4) Resumo
    raw_count       = len(_list_files(os.path.join(DATA_RAW, "*.hdf5")))
    interim_count   = len(_list_files(os.path.join(DATA_INTERIM, "*_whitened.hdf5")))
    processed_count = len(_list_files(os.path.join(DATA_PROCESSED, "*_windows.hdf5")))
    logger.info("======== RESUMO ========")
    logger.info(f"Baixados nesta execução : {total_downloaded} arquivo(s)")
    logger.info(f"Whitened (novos)        : {len(whitened_now)} arquivo(s)")
    logger.info(f"Windows/SNR (novos)     : {len(windows_now)} arquivo(s)")
    logger.info(f"Total em RAW            : {raw_count}")
    logger.info(f"Total em INTERIM        : {interim_count}")
    logger.info(f"Total em PROCESSED      : {processed_count}")
    logger.info("=========================")

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    main()
