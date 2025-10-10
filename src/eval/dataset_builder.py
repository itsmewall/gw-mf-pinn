"""
dataset_builder.py (rev A)
--------------------------
Gera dataset rotulado a partir de *_windows.hdf5:

- Lê data/processed/*_windows.hdf5
- Busca 'tc' (tempo de coalescência) no GWOSC API v2 com ?include-default-parameters
  + cache local com TTL
- Rotula janelas: label=1 se [start_gps, end_gps] intersecta [tc - pre, tc + pos]
  * Se existir event_hint (ex.: "GW150914"), usa diretamente.
  * Senão, faz matching POR ARQUIVO: encontra todos os tc que caem no range GPS
    desse arquivo e marca as janelas que intersectam cada tc.
- Split por GRUPO (evento quando houver hint, senão file_id) — evita leakage.
- Salva Parquet otimizado + CSV preview + meta.json.

Requisitos:
  pip install pandas numpy h5py requests scikit-learn pyarrow tqdm
"""

from __future__ import annotations
import os
import re
import json
import time
import glob
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Cfg:
    PROCESSED_DIR: str = "data/processed"
    OUT_PARQUET: str = os.path.join("data", "processed", "dataset.parquet")
    OUT_PREVIEW: str = os.path.join("data", "processed", "dataset_preview.csv")
    META_JSON: str = os.path.join("data", "processed", "dataset_meta.json")

    # Margem ao redor do tc
    T_MARGIN_PRE: float = 2.0
    T_MARGIN_POS: float = 2.0

    # Splits
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15  # do restante após tirar test

    # Cache de tc
    EV2TC_CACHE: str = os.path.join("data", "processed", "_cache_ev2tc.json")
    EV2TC_TTL_DAYS: int = 7

    # Escrita Parquet
    PARQUET_ROWGROUP_SIZE: int = 250_000
    PARQUET_COMPRESSION: str = "zstd"

    # Sanidade
    MAX_WINDOWS_PER_FILE_WARN: int = 2_000_000

CFG = Cfg()

GWOSC = "https://gwosc.org"
# COM include-default-parameters para trazer 'default_parameters' (inclui tc)
EVENTS_API = f"{GWOSC}/api/v2/event-versions?include-default-parameters"

# =========================
# LOGGING
# =========================
def setup_logger() -> logging.Logger:
    os.makedirs(os.path.dirname(CFG.OUT_PARQUET), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("dataset")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = logging.FileHandler(os.path.join("logs", "dataset_builder.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.handlers[:] = [ch, fh]
    return logger

log = setup_logger()

# =========================
# HTTP com retries
# =========================
def http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "gw-mf-pinn-dataset-builder/1.0"})
    return s

# =========================
# Utils
# =========================
_EVENT_RE = re.compile(r"(GW\d{6,})", re.IGNORECASE)
_DET_RE = re.compile(r"^[A-Z]-([HLV]1)_", re.IGNORECASE)

def _infer_event_name_from_source(source_file: str) -> str:
    m = _EVENT_RE.search(os.path.basename(source_file))
    return m.group(1).upper() if m else ""

def _infer_detector_from_source(source_file: str) -> str:
    m = _DET_RE.match(os.path.basename(source_file))
    return m.group(1).upper() if m else "UNK"

def _now_ts() -> int:
    return int(time.time())

def _is_cache_fresh(path: str, ttl_days: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ts = int(obj.get("_fetched_ts", 0))
        return (_now_ts() - ts) < (ttl_days * 86400)
    except Exception:
        return False

# =========================
# Buscar tc
# =========================

def fetch_event_coalescence_map() -> Dict[str, float]:
    """
    Retorna {EVENTO: gps_coalescencia} usando a API v2:
      GET /api/v2/event-versions?lastver=true&pagesize=200
    Pagina até acabar. Campo 'gps' já vem na lista.
    """
    if _is_cache_fresh(CFG.EV2TC_CACHE, CFG.EV2TC_TTL_DAYS):
        try:
            with open(CFG.EV2TC_CACHE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {k.upper(): float(v) for k, v in payload.get("map", {}).items()}
        except Exception:
            pass

    sess = http_session()
    url = f"{GWOSC}/api/v2/event-versions?lastver=true&pagesize=200"
    ev2tc: Dict[str, float] = {}

    while url:
        r = sess.get(url, timeout=30)
        r.raise_for_status()
        page = r.json()

        for item in page.get("results", []):
            name = item.get("name") or item.get("shortName")
            gps  = item.get("gps")
            if name is not None and gps is not None:
                ev2tc[str(name).upper()] = float(gps)

        # seguir paginação
        nxt = page.get("next")
        url = nxt if nxt else None

    # cache
    try:
        os.makedirs(os.path.dirname(CFG.EV2TC_CACHE), exist_ok=True)
        with open(CFG.EV2TC_CACHE, "w", encoding="utf-8") as f:
            json.dump({"_fetched_ts": _now_ts(), "map": ev2tc}, f)
        log.info(f"Cache salvo: {CFG.EV2TC_CACHE} ({len(ev2tc)} eventos).")
    except Exception as e:
        log.warning(f"Falha ao salvar cache: {e}")

    return ev2tc

# =========================
# Leitura de um *_windows.hdf5
# =========================
def load_windows_h5(path: str) -> pd.DataFrame:
    """
    Retorna DataFrame com:
      file_id, source_file, event_hint, detector,
      fs, window_sec, stride_sec,
      start_idx, start_gps, end_gps, snr_rms, snr_peak, crest_factor
    """
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        fs = float(attrs.get("fs", 4096.0))
        window_sec = float(attrs.get("window_sec", 2.0))
        stride_sec = float(attrs.get("stride_sec", 0.5))
        source_file = str(attrs.get("source_file", os.path.basename(path)))
        g = f["windows"]
        start_idx = g["start_idx"][()]
        start_gps = g["start_gps"][()]
        snr_rms   = g["snr_rms"][()]
        snr_peak  = g["snr_peak"][()]

    n = len(start_idx)
    if n > CFG.MAX_WINDOWS_PER_FILE_WARN:
        log.warning(f"{os.path.basename(path)} tem {n} janelas (> {CFG.MAX_WINDOWS_PER_FILE_WARN}). Revise parâmetros.")

    end_gps = start_gps + window_sec
    crest = np.maximum(snr_peak / np.maximum(snr_rms, 1e-12), 0.0).astype(np.float32)

    det = _infer_detector_from_source(source_file)
    ev_hint = _infer_event_name_from_source(source_file)

    df = pd.DataFrame({
        "file_id": os.path.basename(path),
        "source_file": source_file,
        "event_hint": ev_hint,
        "detector": det,
        "fs": np.float32(fs),
        "window_sec": np.float32(window_sec),
        "stride_sec": np.float32(stride_sec),
        "start_idx": start_idx.astype(np.int64, copy=False),
        "start_gps": start_gps.astype(np.float64, copy=False),
        "end_gps":   end_gps.astype(np.float64, copy=False),
        "snr_rms":   snr_rms.astype(np.float32, copy=False),
        "snr_peak":  snr_peak.astype(np.float32, copy=False),
        "crest_factor": crest
    })
    return df

# =========================
# Rotulagem
# =========================
def label_windows(df: pd.DataFrame, ev2tc: Dict[str, float],
                  t_margin_pre: float, t_margin_pos: float) -> pd.DataFrame:
    """
    label=1 se [start_gps, end_gps] intersecta [tc - pre, tc + pos].
    1) Se houver event_hint e tc estiver em ev2tc, usa diretamente.
    2) Para linhas sem hint/sem tc, faz matching POR ARQUIVO (file_id):
       - calcula [min(start_gps) - pre, max(end_gps) + pos] daquele arquivo
       - pega TODOS os tc que caem nesse intervalo
       - marca janelas que intersectam cada tc candidato
    """
    df = df.copy()
    df["label"] = np.uint8(0)

    # 1) por hint
    mask_have_hint = df["event_hint"].ne("")
    if mask_have_hint.any():
        for ev, sub in df[mask_have_hint].groupby("event_hint"):
            tc = ev2tc.get(ev.upper())
            if tc is None:
                continue
            lo, hi = tc - t_margin_pre, tc + t_margin_pos
            hit = (sub["end_gps"].to_numpy() >= lo) & (sub["start_gps"].to_numpy() <= hi)
            df.loc[sub.index[hit], "label"] = 1

    # 2) por arquivo (fallback robusto)
    mask_unlabeled = df["label"].eq(0)
    if mask_unlabeled.any():
        for fid, sub in df[mask_unlabeled].groupby("file_id"):
            sg_min = float(sub["start_gps"].min())
            eg_max = float(sub["end_gps"].max())
            lo_all = sg_min - t_margin_pre
            hi_all = eg_max + t_margin_pos
            # todos os tc que caem dentro do range do arquivo
            tcs = [tc for tc in ev2tc.values() if lo_all <= tc <= hi_all]
            if not tcs:
                continue
            # marque janelas que intersectam CADA tc
            starts = sub["start_gps"].to_numpy()
            ends   = sub["end_gps"].to_numpy()
            indices = sub.index.to_numpy()
            for tc in tcs:
                lo, hi = tc - t_margin_pre, tc + t_margin_pos
                hit = (ends >= lo) & (starts <= hi)
                if hit.any():
                    df.loc[indices[hit], "label"] = 1

    return df

# =========================
# Split por grupo (evento ou arquivo)
# =========================
def make_group_splits(df: pd.DataFrame,
                      test_size: float, val_size: float,
                      seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    # grupo é o evento quando há hint; senão file_id
    df["group_id"] = df["event_hint"].where(df["event_hint"].ne(""), df["file_id"]).astype("category")

    def _split_by_group(idx_all, groups_all, test_size, seed):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(idx_all, groups=groups_all))
        return tr_idx, te_idx

    idx_all = np.arange(len(df))
    groups_all = df["group_id"].astype(str).to_numpy()

    # test
    trv_idx, te_idx = _split_by_group(idx_all, groups_all, test_size, seed)
    train_val = df.iloc[trv_idx].copy()
    test = df.iloc[te_idx].copy(); test["split"] = "test"

    # val do restante
    rel_val = val_size / (1.0 - test_size)
    idx_trv = np.arange(len(train_val))
    groups_trv = train_val["group_id"].astype(str).to_numpy()
    tr_idx, va_idx = _split_by_group(idx_trv, groups_trv, rel_val, seed + 1)
    train = train_val.iloc[tr_idx].copy(); train["split"] = "train"
    val   = train_val.iloc[va_idx].copy(); val["split"]   = "val"

    out = pd.concat([train, val, test], ignore_index=True)
    # log distribuição
    for tag, part in (("TRAIN", train), ("VAL", val), ("TEST", test)):
        posp = 100.0 * float(part["label"].sum()) / max(len(part), 1)
        log.info(f"{tag}: n={len(part):,} | pos%={posp:.2f}")
    return out

# =========================
# Escrita Parquet incremental
# =========================
def write_parquet(df: pd.DataFrame, out_path: str, compression: str, rowgroup_size: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # dtypes otimizados
    df = df.copy()
    for col in ("detector", "event_hint", "split"):
        if col in df:
            df[col] = df[col].astype("category")
    for col in ("label",):
        if col in df:
            df[col] = df[col].astype(np.uint8)
    for col in ("fs", "window_sec", "stride_sec", "snr_rms", "snr_peak", "crest_factor"):
        if col in df:
            df[col] = df[col].astype(np.float32)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table, out_path,
        compression=compression,
        version="2.6",
        use_dictionary=True,
        data_page_size=None,
        write_statistics=True
    )
    log.info(f"Parquet escrito: {out_path} (linhas: {len(df):,})")

# =========================
# MAIN
# =========================
def main():
    os.makedirs(CFG.PROCESSED_DIR, exist_ok=True)

    # 1) caminhos
    paths = sorted(glob.glob(os.path.join(CFG.PROCESSED_DIR, "*_windows.hdf5")))
    if not paths:
        log.error("Nenhum *_windows.hdf5 em data/processed/. Rode seu run_gwosc.py com ENABLE_WINDOWS=True.")
        return
    log.info(f"Arquivos de janelas encontrados: {len(paths)}")

    # 2) tc (com cache + include-default-parameters)
    ev2tc = fetch_event_coalescence_map()
    log.info(f"Eventos com tc disponíveis: {len(ev2tc)}")

    # 3) ler e rotular por arquivo
    frames: List[pd.DataFrame] = []
    for p in tqdm(paths, desc="Lendo/rotulando", unit="file"):
        try:
            df_file = load_windows_h5(p)
            df_file = label_windows(
                df_file, ev2tc,
                t_margin_pre=CFG.T_MARGIN_PRE,
                t_margin_pos=CFG.T_MARGIN_POS
            )
            frames.append(df_file)
        except Exception as e:
            log.error(f"Falha em {os.path.basename(p)}: {e}")

    if not frames:
        log.error("Nenhum dado válido após leitura/rotulagem.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # 4) splits por grupo
    df_all = make_group_splits(
        df_all,
        test_size=CFG.TEST_SIZE,
        val_size=CFG.VAL_SIZE,
        seed=42
    )

    # 5) colunas finais
    keep_cols = [
        "file_id", "source_file", "event_hint", "detector",
        "fs", "start_gps", "end_gps", "start_idx",
        "window_sec", "stride_sec",
        "snr_rms", "snr_peak", "crest_factor",
        "label", "split"
    ]
    df_all = df_all[keep_cols].copy()

    # preview
    df_all.head(2000).to_csv(CFG.OUT_PREVIEW, index=False)
    log.info(f"Preview salvo: {CFG.OUT_PREVIEW}")

    # 6) parquet
    write_parquet(
        df=df_all,
        out_path=CFG.OUT_PARQUET,
        compression=CFG.PARQUET_COMPRESSION,
        rowgroup_size=CFG.PARQUET_ROWGROUP_SIZE
    )

    # 7) meta
    meta = {
        "t_margin_pre": CFG.T_MARGIN_PRE,
        "t_margin_pos": CFG.T_MARGIN_POS,
        "test_size": CFG.TEST_SIZE,
        "val_size": CFG.VAL_SIZE,
        "parquet_compression": CFG.PARQUET_COMPRESSION,
        "rowgroup_size": CFG.PARQUET_ROWGROUP_SIZE,
        "n_rows": int(len(df_all)),
        "n_pos": int(df_all["label"].sum()),
    }
    with open(CFG.META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Meta salvo: {CFG.META_JSON}")

    # resumo
    tot = len(df_all)
    pos = int(df_all["label"].sum())
    log.info("===== RESUMO DATASET =====")
    log.info(f"Linhas totais : {tot:,}")
    log.info(f"Positivas     : {pos:,}  ({100.0*pos/max(tot,1):.2f}%)")
    log.info(f"Parquet       : {CFG.OUT_PARQUET}")
    log.info("===========================")

if __name__ == "__main__":
    main()
