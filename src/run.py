"""
run.py
------
Motor único SEM ARGUMENTOS.

Fluxo:
  1) Descobre todos os eventos públicos (API v2 GWOSC, com paginação)
  2) Para cada evento, lista strain-files e baixa SÓ o que for necessário
     para a análise inicial (configurável nas variáveis abaixo)
  3) Pré-processa APENAS os arquivos novos (bandpass + notch + whitening)
  4) Mostra progresso (tqdm) e salva log em logs/pipeline.log
  5) Resumo final de quantos arquivos baixou/processou

Requisitos (venv ativo):
  pip install requests h5py numpy scipy tqdm pyyaml
"""

from __future__ import annotations
import os
import sys
import re
import time
import glob
import math
import shutil
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Usa o módulo de pré-processamento do projeto
from gwdata import preprocess as pp


# =============================================================================
# PERFIS / LIMITES (EDITE AQUI)
# =============================================================================
# Perfil padrão de "análise inicial" — ~≤ 10 GB por rodada (ajuste se quiser)
PROFILE = "initial"     # "initial" | "extended" | "full"

# Diretórios
DATA_RAW     = "data/raw"
DATA_INTERIM = "data/interim"
LOG_DIR      = "logs"

# Preferências de arquivo (filtros de qualidade/tamanho)
PREFER_HDF5                = True      # sempre preferir HDF5 (ao invés de GWF)
PREFER_4KHZ                = True      # preferir taxa 4 kHz (menor e suficiente p/ começo)
PREFER_DURATION_4096S      = True      # preferir janelas de 4096 s (ótimas p/ PSD)
ALLOW_DETECTORS            = {"H1", "L1"}  # para começar, só LIGO Hanford/Livingston

# Limites de download por rodada (orçamento)
if PROFILE == "initial":
    MAX_DOWNLOAD_BYTES_PER_RUN = 10 * 1024**3  # 10 GB
    MAX_EVENTS_PER_RUN         = 20            # baixa no máx. N eventos por execução
    MAX_FILES_PER_EVENT        = 2             # ~1 por detector (H1 + L1)
    MAX_SINGLE_FILE_BYTES      = 1 * 1024**3   # 1 GB por arquivo
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

# Sentinela para parar com segurança
STOP_SENTINEL = "STOP"     # se existir um arquivo "STOP" na raiz, downloads param

# Pré-processamento (ajuste aqui conforme sua necessidade)
LOWCUT       = 35.0
HIGHCUT      = 350.0
ORDER_BP     = 6
FS_TARGET    = None        # apenas checagem; sem reamostragem aqui
NOTCH_BASE   = 60.0
NOTCH_HARMS  = 0           # use 5 para remover 60/120/180/240/300 Hz
PSD_SEG      = 4.0
PSD_OVERLAP  = 0.5


# =============================================================================
# GWOSC API (v2)
# =============================================================================
GWOSC        = "https://gwosc.org"
EVENTS_API   = f"{GWOSC}/api/v2/event-versions"     # lista eventos (paginado)
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
    path = urlparse(url).path
    base = os.path.basename(path)
    return base or "file.hdf5"

def _enough_space(path: str, need_bytes: int) -> bool:
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    # reserva 2 GB livres
    return free - need_bytes > 2 * 1024**3

def _match_detector_from_name(fname: str) -> Optional[str]:
    # padrões comuns: H-H1_..., L-L1_..., V-V1_...
    m = re.match(r"^[A-Z]-([HLV]1)_", os.path.basename(fname))
    return m.group(1) if m else None

def _contains_4khz(fname: str) -> bool:
    return "4KHZ" in os.path.basename(fname).upper()

def _contains_4096s(fname: str) -> bool:
    # sufixos "-4096.hdf5", "-4096.gwf", etc.
    return "-4096." in os.path.basename(fname)


# =============================================================================
# API v2: listar TODOS os eventos (paginado)
# =============================================================================
def fetch_all_events() -> List[str]:
    """
    Varre /api/v2/event-versions com paginação e retorna nomes (GW150914, ...)
    Removendo sufixos de versão (-v1, -v2) se houver.
    """
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
        time.sleep(0.15)  # cortesia com a API

    logger.info(f"Eventos públicos descobertos: {len(events)}")
    return events


# =============================================================================
# API v2: strain-files de um evento
# =============================================================================
def fetch_strain_files_for_event(event: str) -> List[Dict]:
    """
    Retorna itens contendo hdf5_url/gwf_url/detail_url.
    """
    url = f"{STRAIN_API}/{event}/strain-files"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return payload.get("results", [])


def _choose_url_and_name(item: Dict) -> Optional[Tuple[str, str]]:
    """
    Decide a melhor URL e nome do arquivo para baixar, respeitando preferências:
      - HDF5 > GWF (se PREFER_HDF5)
      - 4 kHz e 4096 s (se preferido)
      - Retorna (url, filename)
    """
    # candidatos
    url_hdf5 = item.get("hdf5_url")
    url_gwf  = item.get("gwf_url")
    detail   = item.get("detail_url")

    # 1) preferir HDF5 se existir
    url = url_hdf5 if (PREFER_HDF5 and url_hdf5) else (url_hdf5 or url_gwf)

    # 2) se não houver link direto, tentar via detail_url
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

    # filtros 4kHz / 4096 s (apenas se preferido)
    if PREFER_4KHZ and not _contains_4khz(fname):
        return None
    if PREFER_DURATION_4096S and not _contains_4096s(fname):
        return None

    # filtro por detector permitido
    det = _match_detector_from_name(fname)
    if det and ALLOW_DETECTORS and (det not in ALLOW_DETECTORS):
        return None

    return url, fname


# =============================================================================
# DOWNLOAD CONTROLADO
# =============================================================================
def ensure_event_downloaded(event: str,
                            out_dir: str,
                            budget_state: Dict[str, int]) -> int:
    """
    Baixa arquivos de um evento respeitando:
      - MAX_FILES_PER_EVENT
      - MAX_DOWNLOAD_BYTES_PER_RUN (via budget_state["bytes"])
      - MAX_SINGLE_FILE_BYTES
      - STOP_SENTINEL
    Retorna quantos arquivos foram baixados.
    """
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
            logger.info(f"{event}: atingiu limite de {MAX_FILES_PER_EVENT} arquivo(s).")
            break
        if budget_state["bytes"] >= MAX_DOWNLOAD_BYTES_PER_RUN:
            logger.info("Cota de download desta execução atingida. Parando.")
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

        # verifica cota + espaço
        need = size if size else 256 * 1024**2  # assume ~256MB se desconhecido
        if (budget_state["bytes"] + need) > MAX_DOWNLOAD_BYTES_PER_RUN:
            logger.info("Cota de download ficaria acima do limite. Parando.")
            break
        if not _enough_space(out_dir, need):
            logger.error("Espaço em disco insuficiente para continuar.")
            break

        # Download com barra de progresso
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
# PRÉ-PROCESSAMENTO: somente novos
# =============================================================================
def whiten_missing(in_dir: str, out_dir: str) -> int:
    """
    Percorre *.hdf5 em in_dir e processa só os que ainda não têm *_whitened.hdf5 em out_dir.
    Retorna quantos processou nesta execução.
    """
    logger = logging.getLogger("run")
    os.makedirs(out_dir, exist_ok=True)

    raw_files = _list_files(os.path.join(in_dir, "*.hdf5"))
    if not raw_files:
        logger.warning(f"Nenhum .hdf5 encontrado em {in_dir}.")
        return 0

    processed_now = 0
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
            processed_now += 1

    return processed_now


# =============================================================================
# EXECUÇÃO
# =============================================================================
def main():
    logger = setup_logging()

    # Dirs
    os.makedirs(DATA_RAW, exist_ok=True)
    os.makedirs(DATA_INTERIM, exist_ok=True)

    logger.info("==== PIPELINE (sem argumentos) ====")
    logger.info(f"PROFILE = {PROFILE}")
    logger.info(f"RAW={DATA_RAW} | INTERIM={DATA_INTERIM}")
    logger.info(f"Preferências: HDF5={PREFER_HDF5} | 4kHz={PREFER_4KHZ} | 4096s={PREFER_DURATION_4096S} | Detectores={sorted(ALLOW_DETECTORS)}")
    logger.info(f"Limites: MAX_BYTES_RUN={MAX_DOWNLOAD_BYTES_PER_RUN/1e9:.2f} GB | MAX_EVENTS_RUN={MAX_EVENTS_PER_RUN} | MAX_FILES_EVENT={MAX_FILES_PER_EVENT}")
    logger.info(f"Pré-proc: bandpass=({LOWCUT},{HIGHCUT}) ord={ORDER_BP} | notch={NOTCH_BASE} Hz x {NOTCH_HARMS} | PSD={PSD_SEG}s/{PSD_OVERLAP}")
    logger.info("====================================")

    # 1) Descobrir eventos
    all_events = fetch_all_events()
    if MAX_EVENTS_PER_RUN and len(all_events) > MAX_EVENTS_PER_RUN:
        events = all_events[:MAX_EVENTS_PER_RUN]
    else:
        events = all_events
    logger.info(f"Processando nesta execução até {len(events)} evento(s).")

    # 2) Downloads controlados
    budget = {"bytes": 0}
    total_downloaded = 0
    for ev in tqdm(events, desc="Eventos (download se faltar)", unit="evt"):
        try:
            total_downloaded += ensure_event_downloaded(ev, DATA_RAW, budget)
            if budget["bytes"] >= MAX_DOWNLOAD_BYTES_PER_RUN:
                break
            if os.path.exists(STOP_SENTINEL):
                logger.warning("STOP detectado — encerrando ciclo de downloads.")
                break
        except Exception as e:
            logger.error(f"Falha no evento {ev}: {e}")

    # 3) Pré-processar somente o que falta
    processed_now = whiten_missing(DATA_RAW, DATA_INTERIM)

    # 4) Resumo
    raw_count     = len(_list_files(os.path.join(DATA_RAW, "*.hdf5")))
    interim_count = len(_list_files(os.path.join(DATA_INTERIM, "*_whitened.hdf5")))
    logger.info("======== RESUMO ========")
    logger.info(f"Baixados nesta execução : {total_downloaded} arquivo(s)  (~{budget['bytes']/1e9:.2f} GB)")
    logger.info(f"Pré-processados (novos) : {processed_now} arquivo(s)")
    logger.info(f"Total em RAW            : {raw_count}")
    logger.info(f"Total em INTERIM        : {interim_count}")
    logger.info("=========================")


if __name__ == "__main__":
    # permite: python src/run.py
    sys.path.append(os.path.dirname(__file__))
    main()
