# run.py
# --------------------------------------------------------------------------------------
# Motor global SEM ARGUMENTOS: GWOSC → preprocess → windows → dataset → baselines (ML/MF)
# Configure pelos TOGGLES abaixo. Logs em logs/pipeline.log
# Requisitos: requests, h5py, numpy, scipy, tqdm, pyyaml, gwosc, pandas, scikit-learn, joblib, pycbc
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os, sys, time, glob, shutil, re, logging, json
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# ============================================================
# IMPORTA MÓDULOS DO PROJETO (ajusta sys.path)
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gwdata import preprocess as pp
from gwdata.windows import process_whitened_file

# Os módulos de eval expõem main(); chamaremos direto:
from eval import dataset_builder as dsb
from eval import baseline_ml as bml
from eval import mf_baseline as mbf

# ============================================================
# TOGGLES — ligue/desligue estágios
# ============================================================
ENABLE_DOWNLOAD      = False   # baixa GWOSC (quota controlada)
ENABLE_WHITEN        = False   # pré-processa (bandpass/PSD/whiten) só novos
ENABLE_WINDOWS       = False    # gera janelas/SNR só novos
ENABLE_DATASET       = False    # recria dataset.parquet
ENABLE_BASELINE_ML   = False    # roda baseline_ml.py
ENABLE_MF_BASELINE   = True    # roda mf_baseline.py

# PERFIL de download (se ENABLE_DOWNLOAD=True)
PROFILE = "extended"            # "initial" | "extended" | "full"

# Diretórios
DATA_RAW       = os.path.join(ROOT, "data", "raw")
DATA_INTERIM   = os.path.join(ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS_DIR    = os.path.join(ROOT, "reports")
LOG_DIR        = os.path.join(ROOT, "logs")

# Preferências de arquivo (qualidade x tamanho)
PREFER_HDF5           = True
PREFER_4KHZ           = True
PREFER_DURATION_4096S = True
ALLOW_DETECTORS       = {"H1", "L1"}   # inclua "V1" se quiser

# Quotas por perfil
if PROFILE == "initial":
    MAX_DOWNLOAD_BYTES_PER_RUN = 10 * 1024**3  # 10 GB
    MAX_EVENTS_PER_RUN         = 20
    MAX_FILES_PER_EVENT        = 2
    MAX_SINGLE_FILE_BYTES      = 1 * 1024**3
elif PROFILE == "extended":
    MAX_DOWNLOAD_BYTES_PER_RUN = 50 * 1024**3
    MAX_EVENTS_PER_RUN         = 100
    MAX_FILES_PER_EVENT        = 4
    MAX_SINGLE_FILE_BYTES      = 2 * 1024**3
else:
    MAX_DOWNLOAD_BYTES_PER_RUN = 500 * 1024**3
    MAX_EVENTS_PER_RUN         = 10000
    MAX_FILES_PER_EVENT        = 100
    MAX_SINGLE_FILE_BYTES      = 8 * 1024**3

STOP_SENTINEL = "STOP"         # criar um arquivo STOP na raiz interrompe downloads

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================
LOWCUT       = 35.0
HIGHCUT      = 350.0
ORDER_BP     = 6
FS_TARGET    = None            # sem resample aqui; só checa
NOTCH_BASE   = 60.0
NOTCH_HARMS  = 0               # BR: 0 ou 5 (60/120/180/240/300)
PSD_SEG      = 4.0
PSD_OVERLAP  = 0.5

# ============================================================
# JANELAMENTO / SNR
# ============================================================
WINDOW_SEC     = 2.0
STRIDE_SEC     = 0.5
SNR_THRESHOLD  = None          # ex.: 6.0 (opcional)
SAVE_WINDOWS   = False         # True salva matrizes (grande)

# ============================================================
# GWOSC API v2
# ============================================================
GWOSC        = "https://gwosc.org"
EVENTS_API   = f"{GWOSC}/api/v2/event-versions"
STRAIN_API   = f"{GWOSC}/api/v2/events"

# ============================================================
# LOGGING
# ============================================================
def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.handlers[:] = [ch, fh]
    return logger

# ============================================================
# HELPERS
# ============================================================
def _list_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))

def _basename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path) or "file.hdf5"

def _enough_space(path: str, need_bytes: int) -> bool:
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    return (free - need_bytes) > 2 * 1024**3  # reserva 2 GB

def _match_detector_from_name(fname: str) -> Optional[str]:
    m = re.match(r"^[A-Z]-([HLV]1)_", os.path.basename(fname))
    return m.group(1) if m else None

def _contains_4khz(fname: str) -> bool:
    return "4KHZ" in os.path.basename(fname).upper()

def _contains_4096s(fname: str) -> bool:
    return "-4096." in os.path.basename(fname)

def _timer():
    t0 = time.time()
    return lambda: time.time() - t0

# === ADD: helpers para resumir o MF ===
def _latest_subdir(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return max(subs, key=os.path.getmtime) if subs else None

def _summarize_latest_mf(logger):
    mf_root = os.path.join(REPORTS_DIR, "mf_baseline")
    latest = _latest_subdir(mf_root)
    if not latest:
        logger.info("[MF] Nenhum relatório encontrado em reports/mf_baseline/")
        return
    summ = os.path.join(latest, "summary.json")
    if not os.path.exists(summ):
        logger.info(f"[MF] Sem summary.json em {latest}")
        return
    try:
        with open(summ, "r", encoding="utf-8") as f:
            s = json.load(f)
        cfg = s.get("cfg", {})
        val = s.get("val", {})
        test = s.get("test", {})
        thr  = s.get("threshold")
        tinfo = s.get("threshold_info", {})
        conf = test.get("confusion_at_thr", {}) or {}
        tn, fp, fn, tp = (conf.get("tn",0), conf.get("fp",0), conf.get("fn",0), conf.get("tp",0))

        # métricas derivadas no limiar utilizado
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        fpr  = fp / max(tn+fp, 1)

        logger.info("--------------- MF SUMMARY ---------------")
        logger.info(f"Saída: {latest}")
        logger.info(f"GPU: requested={s.get('gpu',{}).get('requested')} available={s.get('gpu',{}).get('available')}")
        logger.info(f"Templates={cfg.get('TEMPLATES_N')} | fs={cfg.get('FS_TARGET')} Hz | lag_step={cfg.get('LAG_STEP')} | ±shift={cfg.get('MAX_SHIFT_SEC')}s")
        logger.info(f"MODE={cfg.get('MODE')} | target_far={cfg.get('TARGET_FAR')}")
        logger.info(f"VAL: AUC={val.get('auc')} | AP={val.get('ap')}")
        logger.info(f"TEST: AUC={test.get('auc')} | AP={test.get('ap')}")
        logger.info(f"Limiar usado: {thr} | info={tinfo}")
        logger.info(f"Conf@thr: tn={tn} fp={fp} fn={fn} tp={tp}")
        logger.info(f"Prec@thr={prec:.6f} | Recall@thr={rec:.6f} | FPR@thr={fpr:.6f}")
        logger.info("-----------------------------------------")
    except Exception as e:
        logger.error(f"[MF] Falha ao ler summary.json: {e}")

# ============================================================
# API v2: eventos (paginado)
# ============================================================
def fetch_all_events(logger) -> List[str]:
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
    logger.info(f"[GWOSC] Eventos públicos descobertos: {len(events)}")
    return events

def fetch_strain_files_for_event(event: str) -> List[Dict]:
    url = f"{STRAIN_API}/{event}/strain-files"
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json().get("results", [])

def _choose_url_and_name(item: Dict) -> Optional[Tuple[str, str]]:
    url_hdf5 = item.get("hdf5_url")
    url_gwf  = item.get("gwf_url")
    detail   = item.get("detail_url")
    url = url_hdf5 if (PREFER_HDF5 and url_hdf5) else (url_hdf5 or url_gwf)
    if not url and detail:
        try:
            dj = requests.get(detail, timeout=20).json()
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

def ensure_event_downloaded(event: str, out_dir: str, budget_state: Dict[str, int], logger) -> int:
    os.makedirs(out_dir, exist_ok=True)
    items = fetch_strain_files_for_event(event)
    if not items:
        logger.info(f"[GWOSC] {event}: nenhum strain-file disponível.")
        return 0

    downloaded = 0
    for item in items:
        if os.path.exists(os.path.join(ROOT, STOP_SENTINEL)):
            logger.warning("[GWOSC] STOP detectado — interrompendo downloads com segurança.")
            break
        if downloaded >= MAX_FILES_PER_EVENT:
            logger.info(f"[GWOSC] {event}: limite {MAX_FILES_PER_EVENT} atingido.")
            break
        if budget_state["bytes"] >= MAX_DOWNLOAD_BYTES_PER_RUN:
            logger.info("[GWOSC] Cota desta execução atingida.")
            break

        choice = _choose_url_and_name(item)
        if not choice:
            continue
        url, fname = choice
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            logger.info(f"[GWOSC] {event}: já existe → {fname} (pulando)")
            continue

        # HEAD p/ tamanho
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            head.raise_for_status()
            size = int(head.headers.get("Content-Length", "0")) if head.headers.get("Content-Length") else 0
        except Exception:
            size = 0

        if size and size > MAX_SINGLE_FILE_BYTES:
            logger.info(f"[GWOSC] {event}: {fname} ({size/1e9:.2f} GB) > limite. Pulando.")
            continue

        need = size if size else 256 * 1024**2
        if (budget_state["bytes"] + need) > MAX_DOWNLOAD_BYTES_PER_RUN:
            logger.info("[GWOSC] Ultrapassaria cota — parando.")
            break
        if not _enough_space(out_dir, need):
            logger.error("[GWOSC] Espaço em disco insuficiente.")
            break

        logger.info(f"[GWOSC] {event}: baixando {fname} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None
            with open(fpath, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"↓ {fname}", leave=False) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 14):
                    if not chunk: continue
                    f.write(chunk)
                    if total: pbar.update(len(chunk))

        downloaded += 1
        budget_state["bytes"] += (total or need)

    return downloaded

# ============================================================
# ESTÁGIOS
# ============================================================
def stage_download(logger) -> Dict[str, int]:
    if not ENABLE_DOWNLOAD:
        logger.info("[DL] Download desativado.")
        return {"downloaded": 0, "bytes": 0}

    events = fetch_all_events(logger)
    events = events[:MAX_EVENTS_PER_RUN] if MAX_EVENTS_PER_RUN else events
    logger.info(f"[DL] Até {len(events)} evento(s) nesta execução.")
    budget = {"bytes": 0}
    downloaded = 0
    for ev in tqdm(events, desc="[DL] Eventos", unit="evt"):
        try:
            downloaded += ensure_event_downloaded(ev, DATA_RAW, budget, logger)
            if budget["bytes"] >= MAX_DOWNLOAD_BYTES_PER_RUN or os.path.exists(os.path.join(ROOT, STOP_SENTINEL)):
                break
        except Exception as e:
            logger.error(f"[DL] {ev}: {e}")
    return {"downloaded": downloaded, "bytes": budget["bytes"]}

def stage_whiten(logger) -> List[str]:
    if not ENABLE_WHITEN:
        logger.info("[PP] Whitening desativado.")
        return []
    os.makedirs(DATA_INTERIM, exist_ok=True)
    raw_files = _list_files(os.path.join(DATA_RAW, "*.hdf5"))
    saved = []
    for fpath in tqdm(raw_files, desc="[PP] Pré-processando (novos)", unit="arq"):
        base = os.path.basename(fpath)
        out_name = base.replace(".hdf5", "_whitened.hdf5")
        out_path = os.path.join(DATA_INTERIM, out_name)
        if os.path.exists(out_path):
            continue
        out = pp.process_file(
            in_path=fpath,
            out_dir=DATA_INTERIM,
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

def stage_windows(logger) -> List[str]:
    if not ENABLE_WINDOWS:
        logger.info("[WIN] Janelas desativado.")
        return []
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    done = []
    for wp in _list_files(os.path.join(DATA_INTERIM, "*_whitened.hdf5")):
        base = os.path.basename(wp).replace("_whitened.hdf5", "")
        out_path = os.path.join(DATA_PROCESSED, f"{base}_windows.hdf5")
        if os.path.exists(out_path):  # já tem (índice leve)
            try:
                # garante meta mínima (source_path) p/ MF
                from tools.fix_windows_meta import ensure_source_path_attr
                ensure_source_path_attr(out_path)
            except Exception:
                pass
            continue
        out = process_whitened_file(
            in_path=wp,
            out_dir=DATA_PROCESSED,
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC,
            snr_threshold=SNR_THRESHOLD,
            save_windows=SAVE_WINDOWS,
        )
        if out:
            # adiciona source_path meta
            try:
                from tools.fix_windows_meta import ensure_source_path_attr
                ensure_source_path_attr(out)
            except Exception:
                pass
            done.append(out)
    return done

def stage_dataset(logger) -> Dict[str, int]:
    if not ENABLE_DATASET:
        logger.info("[DS] Dataset builder desativado.")
        return {"rows": 0, "pos": 0}
    logger.info("[DS] Construindo dataset.parquet …")
    dsb.main()  # usa as configs internas do dataset_builder
    # tenta ler o meta JSON para dar feedback
    meta_path = os.path.join(DATA_PROCESSED, "dataset_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                m = json.load(f)
                logger.info(f"[DS] Linhas: {m.get('n_rows')} | Positivas: {m.get('n_pos')}")
                return {"rows": int(m.get("n_rows", 0)), "pos": int(m.get("n_pos", 0))}
            except Exception:
                pass
    return {"rows": 0, "pos": 0}

def stage_baseline_ml(logger) -> Optional[str]:
    if not ENABLE_BASELINE_ML:
        logger.info("[ML] Baseline ML desativado.")
        return None
    logger.info("[ML] Rodando baseline_ml …")
    bml.main()  # salva em reports/baseline_ml/<timestamp> (no seu script)
    return "baseline_ml"

def stage_mf_baseline(logger) -> Optional[str]:
    if not ENABLE_MF_BASELINE:
        logger.info("[MF] Matched Filtering baseline desativado.")
        return None
    logger.info("[MF] Rodando mf_baseline …")
    mbf.main()  # salva em reports/mf_baseline/<timestamp>
    return "mf_baseline"

# ============================================================
# MAIN
# ============================================================
def main():
    logger = setup_logging()
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)

    logger.info("============== PIPELINE GLOBAL ==============")
    logger.info(f"Toggles: download={ENABLE_DOWNLOAD} | whiten={ENABLE_WHITEN} | windows={ENABLE_WINDOWS} | dataset={ENABLE_DATASET} | ML={ENABLE_BASELINE_ML} | MF={ENABLE_MF_BASELINE}")
    logger.info(f"Profile={PROFILE} | RAW={DATA_RAW} | INTERIM={DATA_INTERIM} | PROCESSED={DATA_PROCESSED}")
    logger.info("=============================================")

    t_all = _timer()

    # 1) DOWNLOAD
    t = _timer()
    dl_stats = stage_download(logger)
    logger.info(f"[DL] Concluído em {t():.1f}s | novos={dl_stats['downloaded']} | bytes≈{dl_stats['bytes']/1e9:.2f} GB")

    # 2) WHITENING
    t = _timer()
    wh_now = stage_whiten(logger)
    logger.info(f"[PP] Concluído em {t():.1f}s | novos={len(wh_now)}")

    # 3) WINDOWS
    t = _timer()
    win_now = stage_windows(logger)
    logger.info(f"[WIN] Concluído em {t():.1f}s | novos={len(win_now)}")

    # 4) DATASET
    t = _timer()
    ds_stats = stage_dataset(logger)
    logger.info(f"[DS] Concluído em {t():.1f}s | linhas={ds_stats['rows']} | pos={ds_stats['pos']}")

    # 5) BASELINE ML
    t = _timer()
    tag_ml = stage_baseline_ml(logger)
    if tag_ml:
        logger.info(f"[ML] Concluído em {t():.1f}s")

    # 6) MF BASELINE
    t = _timer()
    tag_mf = stage_mf_baseline(logger)
    if tag_mf:
        logger.info(f"[MF] Concluído em {t():.1f}s")
        _summarize_latest_mf(logger)


    # RESUMO
    raw_count       = len(_list_files(os.path.join(DATA_RAW, "*.hdf5")))
    interim_count   = len(_list_files(os.path.join(DATA_INTERIM, "*_whitened.hdf5")))
    processed_count = len(_list_files(os.path.join(DATA_PROCESSED, "*_windows.hdf5")))

    logger.info("=============== RESUMO FINAL ===============")
    logger.info(f"RAW total:       {raw_count}")
    logger.info(f"INTERIM total:   {interim_count}")
    logger.info(f"PROCESSED total: {processed_count}")
    logger.info(f"Dataset:         linhas={ds_stats['rows']} | positivas={ds_stats['pos']}")
    logger.info(f"Baselines:       ML={'ok' if tag_ml else '-'} | MF={'ok' if tag_mf else '-'}")
    logger.info(f"Tempo total:     {t_all():.1f} s")
    logger.info("============================================")

if __name__ == "__main__":
    main()