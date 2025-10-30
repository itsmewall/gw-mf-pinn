# run.py
# --------------------------------------------------------------------------------------
# Motor global SEM ARGUMENTOS: GWOSC -> preprocess -> windows -> dataset -> baselines (ML/MF)
# Configure pelos TOGGLES abaixo. Logs em logs/pipeline.log
# Requisitos: requests, h5py, numpy, scipy, tqdm, pyyaml, gwosc, pandas, scikit-learn,
# joblib, pycbc, torch (se MF2 for usado)
# --------------------------------------------------------------------------------------

from __future__ import annotations
import os, sys, time, glob, shutil, re, logging, json, traceback
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass
from time import perf_counter

import requests
from tqdm import tqdm
import numpy as np
import pandas as pd

# ============================================================
# IMPORTA MODULOS DO PROJETO (ajusta sys.path)
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gwdata import preprocess as pp
from gwdata.windows import process_whitened_file

# Módulos de eval expõem main(); chamaremos direto
from eval import dataset_builder as dsb
from eval import baseline_ml as bml
from eval import mf_baseline as mbf

# Viz
from viz import plot_scores_timeline as viz_timeline
from viz import ligo_overlay as viz_overlay

# ============================================================
# TOGGLES
# ============================================================
ENABLE_DOWNLOAD      = False
ENABLE_WHITEN        = False
ENABLE_WINDOWS       = False
ENABLE_DATASET       = False
ENABLE_BASELINE_ML   = False
ENABLE_MF_BASELINE   = False
ENABLE_MF_STAGE2     = True

# Pós MF2
ENABLE_POST_MF2_FREEZE    = True
ENABLE_POST_MF2_EXPORT    = True
ENABLE_POST_MF2_COINC     = True
ENABLE_POST_MF2_CALIBRATE = True

# Pré-flight
ENABLE_PREFLIGHT        = True
PREFLIGHT_FAIL_FAST     = True
PREFLIGHT_SAMPLE_VAL    = 64
PREFLIGHT_MIN_WINDOWS   = 1
PREFLIGHT_MIN_WHITENED  = 1
PREFLIGHT_MIN_DISK_GB   = 3

# PERFIL de download
PROFILE = "extended"            # initial | extended | full

# Diretórios
DATA_RAW       = os.path.join(ROOT, "data", "raw")
DATA_INTERIM   = os.path.join(ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS_DIR    = os.path.join(ROOT, "reports")
LOG_DIR        = os.path.join(ROOT, "logs")

# Preferências de arquivo
PREFER_HDF5           = True
PREFER_4KHZ           = True
PREFER_DURATION_4096S = True
ALLOW_DETECTORS       = {"H1", "L1"}

# Quotas por perfil
if PROFILE == "initial":
    MAX_DOWNLOAD_BYTES_PER_RUN = 10 * 1024**3
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

STOP_SENTINEL = "STOP"

# ============================================================
# PRE PROCESSAMENTO
# ============================================================
LOWCUT       = 35.0
HIGHCUT      = 350.0
ORDER_BP     = 6
FS_TARGET    = None
NOTCH_BASE   = 60.0
NOTCH_HARMS  = 0
PSD_SEG      = 4.0
PSD_OVERLAP  = 0.5

# ============================================================
# JANELAMENTO e SNR
# ============================================================
WINDOW_SEC     = 2.0
STRIDE_SEC     = 0.5
SNR_THRESHOLD  = None
SAVE_WINDOWS   = False

# ============================================================
# GWOSC API v2
# ============================================================
GWOSC      = "https://gwosc.org"
EVENTS_API = f"{GWOSC}/api/v2/event-versions"
STRAIN_API = f"{GWOSC}/api/v2/events"

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
    return (free - need_bytes) > 2 * 1024**3

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

def _latest_mf2_run_dir() -> Optional[str]:
    root = os.path.join(REPORTS_DIR, "mf_stage2")
    return _latest_subdir(root)

def freeze_split_after_dataset():
    base = os.path.join(DATA_PROCESSED, "dataset.parquet")
    side = os.path.join(DATA_PROCESSED, "dataset_with_split.parquet")
    if not (os.path.exists(base) and os.path.exists(side)):
        return
    df_base = pd.read_parquet(base)
    df_side = pd.read_parquet(side)
    keys = [c for c in ["file_id", "start_gps"] if c in df_base.columns and c in df_side.columns]
    if len(keys) < 2:
        print("[FREEZE] chaves insuficientes para restaurar split")
        return
    keep = keys + [c for c in ["split", "subset"] if c in df_side.columns]
    df_side = df_side[keep].drop_duplicates()
    if "subset" in df_side.columns and "split" not in df_side.columns:
        df_side = df_side.rename(columns={"subset": "split"})
    df_out = df_base.drop(columns=["split", "subset"], errors="ignore").merge(df_side, on=keys, how="left")
    miss = df_out["split"].isna().sum()
    if miss:
        print(f"[FREEZE] {miss} linhas sem split no sidecar. Atribuindo train.")
        df_out["split"] = df_out["split"].fillna("train")
    df_out.to_parquet(side, index=False)
    df_out.to_parquet(base, index=False)
    print(f"[FREEZE] split restaurado em {base} e {side}")

# ============================================================
# PRÉ-FLIGHT
# ============================================================
@dataclass
class TestResult:
    name: str
    ok: bool
    duration_s: float
    detail: str = ""
    critical: bool = True

def _human_gb(bytes_val: int) -> str:
    return f"{bytes_val/(1024**3):.2f} GB"

def _free_space_bytes(path: str) -> int:
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    return free

def preflight(logger) -> List[TestResult]:
    results: List[TestResult] = []

    def run(name: str, critical: bool, fn):
        t0 = perf_counter()
        try:
            msg = fn()
            results.append(TestResult(name=name, ok=True, duration_s=perf_counter()-t0, detail=msg or "", critical=critical))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            results.append(TestResult(name=name, ok=False, duration_s=perf_counter()-t0, detail=f"{type(e).__name__}: {e}\n{tb}", critical=critical))

    # 0) Disco e escrita
    def test_disk():
        free_root = _free_space_bytes(ROOT)
        if free_root < PREFLIGHT_MIN_DISK_GB*(1024**3):
            raise RuntimeError(f"Espaço livre insuficiente: {_human_gb(free_root)}")
        for d in (LOG_DIR, REPORTS_DIR):
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
        return f"Livre={_human_gb(free_root)} escrita=ok"
    run("disk_write_space", True, test_disk)

    # 1) Imports essenciais
    def test_imports():
        import numpy, scipy, pandas, sklearn, pycbc  # noqa
        return "numpy, scipy, pandas, sklearn, pycbc importados"
    run("imports_core_libs", True, test_imports)

    # 2) Dados básicos presentes
    def test_min_files():
        wht = _list_files(os.path.join(DATA_INTERIM, "*_whitened.hdf5"))
        win = _list_files(os.path.join(DATA_PROCESSED, "*_windows.hdf5"))
        if len(wht) < PREFLIGHT_MIN_WHITENED:
            raise RuntimeError(f"Sem arquivos whitened suficientes em {DATA_INTERIM}")
        if len(win) < PREFLIGHT_MIN_WINDOWS:
            raise RuntimeError(f"Sem arquivos windows suficientes em {DATA_PROCESSED}")
        return f"whitened={len(wht)} windows={len(win)}"
    run("data_files_presence", True, test_min_files)

    # 3) Dataset.parquet sanidade
    def test_dataset_parquet():
        p = os.path.join(DATA_PROCESSED, "dataset.parquet")
        if not os.path.exists(p):
            raise FileNotFoundError(f"dataset.parquet ausente em {DATA_PROCESSED}")
        df = pd.read_parquet(p)
        cols_ok = {"file_id","start_gps","label"}
        if "subset" not in df.columns and "split" in df.columns:
            df = df.rename(columns={"split":"subset"})
        if "subset" not in df.columns:
            raise KeyError("dataset.parquet sem coluna subset")
        miss = cols_ok - set(df.columns)
        if miss:
            raise KeyError(f"colunas ausentes: {sorted(miss)}")
        if df["label"].isna().any():
            raise ValueError("labels com NaN")
        labs = set(int(x) for x in df["label"].unique())
        if not labs.issubset({0,1}):
            raise ValueError(f"labels fora de {{0,1}}: {labs}")
        subs = set(df["subset"].astype(str).unique())
        if not {"val","test"}.intersection(subs):
            raise ValueError("subset sem val ou test")
        return f"rows={len(df)} subsets={sorted(subs)}"
    run("dataset_structure", True, test_dataset_parquet)

    # 4) Leitura de janela + map para whitened via meta
    def test_window_slice():
        idx = mbf.build_windows_index(DATA_PROCESSED)
        if not idx:
            raise RuntimeError("index de windows vazio")
        any_fid, any_win = next(iter(idx.items()))
        sgps, _, meta_in = mbf.cached_start_arrays(any_win)
        if len(sgps) == 0:
            raise RuntimeError("start_gps vazio neste windows")
        wht = mbf.resolve_whitened_path(any_win, meta_in)
        if not wht or not os.path.exists(wht):
            raise RuntimeError("whitened correspondente não encontrado")
        x = mbf.load_window_slice(any_win, wht, float(sgps[min(3, len(sgps)-1)]))
        if x is None or not np.isfinite(x).all():
            raise RuntimeError("janela inválida ou com NaN")
        return f"window_len={len(x)} file_id={any_fid}"
    run("window_read", True, test_window_slice)

    # 5) PyCBC gera waveform
    def test_pycbc_waveform():
        from pycbc.waveform import get_td_waveform
        hp, _ = get_td_waveform(approximant="IMRPhenomD", mass1=10, mass2=10, delta_t=1.0/4096, f_lower=20.0)
        arr = hp.numpy()
        if arr.size == 0 or not np.isfinite(arr).any():
            raise RuntimeError("waveform vazio")
        return f"waveform_len={arr.size}"
    run("pycbc_waveform", True, test_pycbc_waveform)

    # 6) Smoke do MF em CPU com amostra pequena
    def test_mf_smoke():
        p = os.path.join(DATA_PROCESSED, "dataset.parquet")
        df = pd.read_parquet(p)
        if "subset" not in df.columns and "split" in df.columns:
            df = df.rename(columns={"split":"subset"})
        df = df[["subset","file_id","start_gps","label"]]

        df_sub = pd.DataFrame()
        for split in ("val","test"):
            take = df[df["subset"]==split].head(PREFLIGHT_SAMPLE_VAL).copy()
            if len(take) >= max(8, min(PREFLIGHT_SAMPLE_VAL, 32)):
                df_sub = take
                break
        if df_sub.empty:
            raise RuntimeError("sem linhas suficientes para smoke MF")

        idx_map = mbf.build_windows_index(DATA_PROCESSED)
        df_sub = df_sub[df_sub["file_id"].astype(str).isin(idx_map.keys())].reset_index(drop=True)
        if df_sub.empty:
            raise RuntimeError("file_ids do dataset não batem com *_windows.hdf5")

        smoke_dir = os.path.join(REPORTS_DIR, "mf_baseline_smoke")
        os.makedirs(smoke_dir, exist_ok=True)

        with mbf.patch_cfg(
            USE_GPU=False,
            USE_NATIVE_NCC=False,
            MAX_SHIFT_SEC=0.010,
            LAG_STEP=8,
            TEMPLATES_N=16,
            TEMPLATE_SUBSAMPLE=4,
            PRESELECT_ENABLE=False,
            TOPK_POOL=1,
            OUT_DIR_ROOT=smoke_dir,
            MAX_VAL_ROWS=None,
            MAX_TEST_ROWS=None,
            NO_ABORT=True
        ):
            bank  = mbf.build_template_bank(mbf.CFG.TEMPLATES_N)
            cache = mbf.maybe_load_template_cache(smoke_dir, bank, cache_name="__smoke_cache.npz")
            scores = mbf.score_subset("SMOKE", df_sub, idx_map, cache)
            if not np.isfinite(scores).all():
                raise RuntimeError("scores do MF com NaN ou inf")
            smax = float(np.max(scores)) if scores.size else 0.0
            return f"rows={len(df_sub)} templates={len(cache)} smax={smax:.4f}"
    run("mf_smoke_cpu", True, test_mf_smoke)

    # 7) ML baseline
    def test_ml_presence():
        if not hasattr(bml, "main") or not callable(bml.main):
            raise RuntimeError("baseline_ml sem função main()")
        p = os.path.join(DATA_PROCESSED, "dataset.parquet")
        df = pd.read_parquet(p)
        if "subset" not in df.columns and "split" in df.columns:
            df = df.rename(columns={"split":"subset"})
        has_train = (df["subset"].astype(str)=="train").any()
        has_val   = (df["subset"].astype(str)=="val").any()
        if not (has_train or has_val):
            raise RuntimeError("dataset sem train ou val para ML")
        return "baseline_ml import ok e dataset com train ou val"
    run("ml_presence_struct", True, test_ml_presence)

    # 8) MF2 presença opcional
    def test_mf2_presence():
        try:
            import importlib.util, pathlib
            tmf_path = pathlib.Path(SRC) / "training" / "train_mf.py"
            if tmf_path.exists():
                return "training.train_mf disponível por caminho de arquivo"
            return "MF2 opcional ausente: train_mf.py não encontrado"
        except Exception as e:
            return f"MF2 opcional ausente: {e}"
    run("mf2_presence", False, test_mf2_presence)

    # 9) CuPy opcional
    def test_gpu_optional():
        try:
            import cupy as cp  # noqa
            _ = cp.cuda.runtime.getDeviceCount()
            return "CuPy disponível"
        except Exception as e:
            return f"CuPy indisponível: {e}"
    run("gpu_optional_cupy", False, test_gpu_optional)

    ok_all = True
    logging.getLogger("pipeline").info("=============== PREFLIGHT ===============")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        crit   = "critico" if r.critical else "opcional"
        logging.getLogger("pipeline").info(f"[{status}] {r.name} ({crit}) em {r.duration_s:.2f}s - {r.detail}")
        if r.critical and not r.ok:
            ok_all = False
    logging.getLogger("pipeline").info("=========================================")

    if PREFLIGHT_FAIL_FAST and not ok_all:
        raise SystemExit("Pré-flight falhou em teste crítico. Abortado.")

    logging.getLogger("pipeline").info(f"[PREFLIGHT] ok={sum(r.ok for r in results)}/{len(results)}")
    return results

# ============================================================
# API v2: eventos
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
            logger.warning("[GWOSC] STOP detectado - interrompendo downloads com segurança.")
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
            logger.info(f"[GWOSC] {event}: já existe -> {fname} (pulando)")
            continue

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
            logger.info("[GWOSC] Ultrapassaria cota - parando.")
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
        if os.path.exists(out_path):
            try:
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
    dsb.main()
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
    bml.main()
    return "baseline_ml"

def stage_mf_baseline(logger) -> Optional[str]:
    if not ENABLE_MF_BASELINE:
        logger.info("[MF] Matched Filtering baseline desativado.")
        return None
    logger.info("[MF] Rodando mf_baseline …")
    mbf.main()
    return "mf_baseline"

def stage_mf_stage2(logger) -> Optional[str]:
    if not ENABLE_MF_STAGE2:
        logger.info("[MF2] Multi Fidelity Stage 2 desativado.")
        return None

    logger.info("[MF2] Treinando MF Stage 2 …")

    import importlib.util, sys

    tmf_path = os.path.join(SRC, "training", "train_mf.py")
    if not os.path.exists(tmf_path):
        logger.error(f"[MF2] train_mf.py não encontrado em {tmf_path}")
        return None

    # use um nome de pacote estável
    spec = importlib.util.spec_from_file_location("training.train_mf", tmf_path)
    mf2_dyn = importlib.util.module_from_spec(spec)

    # registrar antes de executar, para o dataclass enxergar o módulo
    sys.modules[spec.name] = mf2_dyn
    mf2_dyn.__package__ = "training"

    try:
        spec.loader.exec_module(mf2_dyn)  # type: ignore
    except Exception as e:
        logger.error(f"[MF2] Falha ao carregar {tmf_path}: {e}")
        return None

    out_root = os.path.join(REPORTS_DIR, "mf_stage2")
    os.makedirs(out_root, exist_ok=True)

    try:
        if hasattr(mf2_dyn, "run_mf_stage2") and callable(mf2_dyn.run_mf_stage2):
            mf2_dyn.run_mf_stage2(out_dir_root=out_root)
            return "mf_stage2"
        elif hasattr(mf2_dyn, "train") and callable(mf2_dyn.train):
            hp = {
                "PARQUET_PATH": os.path.join(DATA_PROCESSED, "dataset.parquet"),
                "WINDOWS_DIR": DATA_PROCESSED,
                "OUT_DIR": out_root,
                "DEVICE": "cuda",
            }
            mf2_dyn.train(hparams=hp)
            return "mf_stage2"
        else:
            logger.error("[MF2] train_mf.py não expõe run_mf_stage2() nem train().")
            return None
    except Exception as e:
        logger.error(f"[MF2] Execução falhou: {e}")
        return None

# ------------------------------------------------------------
# Pós MF2: FREEZE, EXPORT, COINC, CALIBRATE
# ------------------------------------------------------------
def stage_post_mf2_freeze(logger) -> Optional[str]:
    if not ENABLE_POST_MF2_FREEZE:
        logger.info("[MF2:FREEZE] desativado.")
        return None
    run = _latest_mf2_run_dir()
    if not run:
        logger.info("[MF2:FREEZE] nenhum run em reports/mf_stage2.")
        return None
    need = ["summary.json", "thresholds.json", "scores_val.csv", "scores_test.csv"]
    missing = [n for n in need if not os.path.exists(os.path.join(run, n))]
    if missing:
        logger.info(f"[MF2:FREEZE] aguardando arquivos: {missing}. Pulando por enquanto.")
        return None
    out = os.path.join("artifacts", "mf2", os.path.basename(run.rstrip(os.sep)))
    os.makedirs(out, exist_ok=True)
    for n in need:
        shutil.copy2(os.path.join(run, n), os.path.join(out, n))
    for cfg in ("configs/train_mf.yaml", "configs/train_mf2.yaml"):
        if os.path.exists(cfg):
            shutil.copy2(cfg, os.path.join(out, os.path.basename(cfg)))
    logger.info(f"[MF2:FREEZE] artefatos congelados em {out}")
    return out

def stage_post_mf2_export(logger) -> Optional[str]:
    if not ENABLE_POST_MF2_EXPORT:
        logger.info("[MF2:EXPORT] desativado.")
        return None
    run = _latest_mf2_run_dir()
    if not run:
        logger.info("[MF2:EXPORT] nenhum run em reports/mf_stage2.")
        return None
    thr_path  = os.path.join(run, "thresholds.json")
    test_path = os.path.join(run, "scores_test.csv")
    if not (os.path.exists(thr_path) and os.path.exists(test_path)):
        logger.info("[MF2:EXPORT] thresholds.json ou scores_test.csv ausente.")
        return None
    with open(thr_path, "r") as f:
        data_thr = json.load(f)
    thr = data_thr.get("chosen_threshold", None)
    if thr is None:
        logger.info("[MF2:EXPORT] chosen_threshold ausente.")
        return None
    df = pd.read_csv(test_path)
    if "score" not in df.columns:
        logger.info("[MF2:EXPORT] scores_test.csv sem coluna score.")
        return None
    cand = df[df["score"] >= float(thr)].sort_values("score", ascending=False)
    outdir = os.path.join(run, "candidates")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "test_candidates.csv")
    cand.to_csv(out, index=False)
    logger.info(f"[MF2:EXPORT] {len(cand)} candidatos salvos em {out}")
    return out

def _det_from_file_id(fid: str) -> str:
    if "H1" in fid: return "H1"
    if "L1" in fid: return "L1"
    if "V1" in fid: return "V1"
    m = re.search(r"([HLV]1)", fid)
    return m.group(1) if m else "UNK"

def _count_coinc(times_a, times_b, dt):
    i = j = 0
    n = 0
    while i < len(times_a) and j < len(times_b):
        da = times_a[i] - times_b[j]
        if abs(da) <= dt:
            n += 1; i += 1; j += 1
        elif da > dt:
            j += 1
        else:
            i += 1
    return n

def stage_post_mf2_coinc(logger, dt_sec: float = 0.015, slides: int = 100, slide_step: float = 1.3) -> Optional[str]:
    if not ENABLE_POST_MF2_COINC:
        logger.info("[MF2:COINC] desativado.")
        return None
    run = _latest_mf2_run_dir()
    if not run:
        logger.info("[MF2:COINC] nenhum run em reports/mf_stage2.")
        return None
    cand_path = os.path.join(run, "candidates", "test_candidates.csv")
    if not os.path.exists(cand_path):
        logger.info("[MF2:COINC] candidatos ausentes. Rode export primeiro.")
        return None
    df = pd.read_csv(cand_path)
    need = {"file_id","start_gps","score"}
    if not need.issubset(df.columns):
        logger.info("[MF2:COINC] candidatos sem colunas esperadas.")
        return None
    df["det"] = df["file_id"].astype(str).map(_det_from_file_id)
    tH = np.sort(df.loc[df["det"]=="H1","start_gps"].astype(float).values)
    tL = np.sort(df.loc[df["det"]=="L1","start_gps"].astype(float).values)
    if len(tH)==0 or len(tL)==0:
        logger.info("[MF2:COINC] sem H1 e L1 suficientes.")
        return None
    tmin = max(tH.min(), tL.min()); tmax = min(tH.max(), tL.max())
    Tobs = max(0.0, tmax - tmin)
    if Tobs <= 0:
        logger.info("[MF2:COINC] Tobs nao positivo.")
        return None
    n_real = _count_coinc(tH, tL, dt_sec)
    rng = np.random.default_rng(42)
    bg = []
    for k in range(1, slides+1):
        shift = (slide_step * k) + float(rng.uniform(0.1, 0.9))
        bg.append(_count_coinc(tH, tL + shift, dt_sec))
    bg = np.asarray(bg, float)
    far = float(bg.mean() / Tobs) if Tobs > 0 else float("nan")
    out = {
        "dt_sec": float(dt_sec),
        "slides": int(slides),
        "slide_step_sec": float(slide_step),
        "coinc_real": int(n_real),
        "bg_mean": float(bg.mean()),
        "bg_std": float(bg.std(ddof=1)) if len(bg)>1 else 0.0,
        "Tobs_sec": float(Tobs),
        "FAR_per_sec": far,
        "FAR_per_day": far * 86400.0
    }
    out_path = os.path.join(run, "coincidence.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"[MF2:COINC] escrito {out_path}")
    return out_path

def stage_post_mf2_calibrate(logger) -> Optional[str]:
    if not ENABLE_POST_MF2_CALIBRATE:
        logger.info("[MF2:CALIB] desativado.")
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
        from joblib import dump
    except Exception as e:
        logger.error(f"[MF2:CALIB] dependencias ausentes: {e}")
        return None
    run = _latest_mf2_run_dir()
    if not run:
        logger.info("[MF2:CALIB] nenhum run em reports/mf_stage2.")
        return None
    valp = os.path.join(run, "scores_val.csv")
    tsp  = os.path.join(run, "scores_test.csv")
    if not (os.path.exists(valp) and os.path.exists(tsp)):
        logger.info("[MF2:CALIB] scores_val.csv ou scores_test.csv ausente.")
        return None
    val = pd.read_csv(valp); test = pd.read_csv(tsp)
    if not {"score","label"}.issubset(val.columns):
        logger.info("[MF2:CALIB] VAL sem score ou label.")
        return None
    y = val["label"].astype(int).values
    s = val["score"].astype(float).values
    cal = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    dump(cal, os.path.join(run, "calibrator_isotonic.joblib"))
    val["p_mf2"] = cal.predict(val["score"].astype(float).values)
    test["p_mf2"] = cal.predict(test["score"].astype(float).values)
    val.to_csv(os.path.join(run, "scores_val_calibrated.csv"), index=False)
    test.to_csv(os.path.join(run, "scores_test_calibrated.csv"), index=False)
    logger.info("[MF2:CALIB] calibracao isotonica aplicada.")
    return "ok"

# ============================================================
# MAIN
# ============================================================
def main():
    logger = setup_logging()
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)

    logger.info("============== PIPELINE GLOBAL ==============")
    logger.info(
        "Toggles: "
        f"download={ENABLE_DOWNLOAD} | whiten={ENABLE_WHITEN} | windows={ENABLE_WINDOWS} | "
        f"dataset={ENABLE_DATASET} | ML={ENABLE_BASELINE_ML} | MF={ENABLE_MF_BASELINE} | MF2={ENABLE_MF_STAGE2}"
    )
    logger.info(f"Profile={PROFILE} | RAW={DATA_RAW} | INTERIM={DATA_INTERIM} | PROCESSED={DATA_PROCESSED}")
    logger.info("=============================================")

    # PREFLIGHT
    if ENABLE_PREFLIGHT:
        try:
            preflight(logger)
        except SystemExit:
            logger.error("Pré-flight detectou problemas críticos. Corrija e rode novamente.")
            return
        except Exception as e:
            logger.error(f"Pré-flight falhou de modo inesperado: {e}")
            if PREFLIGHT_FAIL_FAST:
                return

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
    freeze_split_after_dataset()
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
        try:
            logger.info("[VIZ] Gerando timeline de scores (MF e ML) …")
            viz_timeline.main()
        except Exception as e:
            logger.error(f"[VIZ] timeline falhou: {e}")
        try:
            logger.info("[VIZ] Gerando overlay H1 L1 …")
            viz_overlay.main()
        except Exception as e:
            logger.error(f"[VIZ] overlay falhou: {e}")

    # 7) MF STAGE 2
    t = _timer()
    tag_mf2 = stage_mf_stage2(logger)
    if tag_mf2:
        logger.info(f"[MF2] Concluído em {t():.1f}s")
        # Pós MF2
        t = _timer(); stage_post_mf2_freeze(logger);    logger.info(f"[MF2:FREEZE] em {t():.1f}s")
        t = _timer(); stage_post_mf2_export(logger);    logger.info(f"[MF2:EXPORT] em {t():.1f}s")
        t = _timer(); stage_post_mf2_coinc(logger);     logger.info(f"[MF2:COINC] em {t():.1f}s")
        t = _timer(); stage_post_mf2_calibrate(logger); logger.info(f"[MF2:CALIB] em {t():.1f}s")

    # RESUMO
    raw_count       = len(_list_files(os.path.join(DATA_RAW, "*.hdf5")))
    interim_count   = len(_list_files(os.path.join(DATA_INTERIM, "*_whitened.hdf5")))
    processed_count = len(_list_files(os.path.join(DATA_PROCESSED, "*_windows.hdf5")))

    logger.info("=============== RESUMO FINAL ===============")
    logger.info(f"RAW total:       {raw_count}")
    logger.info(f"INTERIM total:   {interim_count}")
    logger.info(f"PROCESSED total: {processed_count}")
    logger.info(f"Dataset:         linhas={ds_stats['rows']} | positivas={ds_stats['pos']}")
    logger.info(f"Baselines:       ML={'ok' if tag_ml else '-'} | MF={'ok' if tag_mf else '-'} | MF2={'ok' if tag_mf2 else '-'}")
    logger.info(f"Tempo total:     {t_all():.1f} s")
    logger.info("============================================")

if __name__ == "__main__":
    main()