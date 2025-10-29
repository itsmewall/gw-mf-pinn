# src/eval/dataset_builder.py
"""
Gera dataset rotulado a partir de *_windows.hdf5.

Pipeline:
1) Lê data/processed/*_windows.hdf5
2) Busca mapa {EVENTO -> tc_gps} no GWOSC API v2 com cache local
3) Rotula janela como label=1 se [start_gps, end_gps] intersecta [tc - pre, tc + pos]
   - Se existir event_hint (ex.: "GW150914"), usa diretamente
   - Senão, matching por arquivo: pega t_c que caem no range coberto e marca interseções
4) Split por grupo em train val test, com mínimos garantidos e fallback para split aleatório
5) Salva Parquet, CSV preview e meta.json

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

    # Margem temporal ao redor do tc
    T_MARGIN_PRE: float = float(os.getenv("DB_T_MARGIN_PRE", 2.0))
    T_MARGIN_POS: float = float(os.getenv("DB_T_MARGIN_POS", 2.0))

    # Splits alvo
    TEST_SIZE: float = float(os.getenv("DB_TEST_SIZE", 0.15))
    VAL_SIZE: float = float(os.getenv("DB_VAL_SIZE", 0.15))  # do restante após remover TEST

    # Mínimos por subset para não quebrar pré-flight
    MIN_TRAIN_ROWS: int = int(os.getenv("DB_MIN_TRAIN", 200))
    MIN_VAL_ROWS: int = int(os.getenv("DB_MIN_VAL", 60))
    MIN_TEST_ROWS: int = int(os.getenv("DB_MIN_TEST", 60))

    # Cache do mapa evento->tc
    EV2TC_CACHE: str = os.path.join("data", "processed", "_cache_ev2tc.json")
    EV2TC_TTL_DAYS: int = int(os.getenv("DB_EV2TC_TTL_DAYS", 7))

    # Escrita Parquet
    PARQUET_ROWGROUP_SIZE: int = int(os.getenv("DB_ROWGROUP", 250_000))
    PARQUET_COMPRESSION: str = os.getenv("DB_PARQUET_CODEC", "zstd")

    # Sanidade
    MAX_WINDOWS_PER_FILE_WARN: int = int(os.getenv("DB_MAX_WIN_PER_FILE_WARN", 2_000_000))

CFG = Cfg()
GWOSC = "https://gwosc.org"

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
    s.headers.update({"User-Agent": "gw-mf-pinn-dataset-builder/1.2"})
    return s

# =========================
# Utils
# =========================
_EVENT_RE = re.compile(r"(GW\d{6,})", re.IGNORECASE)
# Espera padrão tipo H-H1_..., L-L1_..., V-V1_...
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
# GWOSC event map com cache e fallback offline
# =========================
def fetch_event_coalescence_map() -> Tuple[Dict[str, float], np.ndarray]:
    """
    Retorna:
      ev2tc: {EVENTO: gps_coalescencia}
      tc_sorted: array ordenado de t_c
    """
    if _is_cache_fresh(CFG.EV2TC_CACHE, CFG.EV2TC_TTL_DAYS):
        try:
            with open(CFG.EV2TC_CACHE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ev2tc = {k.upper(): float(v) for k, v in payload.get("map", {}).items()}
            tc_sorted = np.sort(np.array(list(ev2tc.values()), dtype=np.float64))
            return ev2tc, tc_sorted
        except Exception:
            pass

    ev2tc: Dict[str, float] = {}
    tc_sorted = np.array([], dtype=np.float64)

    try:
        sess = http_session()
        url = f"{GWOSC}/api/v2/event-versions?lastver=true&pagesize=200"
        while url:
            r = sess.get(url, timeout=30)
            r.raise_for_status()
            page = r.json()
            for item in page.get("results", []):
                name = item.get("name") or item.get("shortName")
                gps = item.get("gps")
                if name is not None and gps is not None:
                    ev2tc[str(name).upper()] = float(gps)
            url = page.get("next") or None

        tc_sorted = np.sort(np.array(list(ev2tc.values()), dtype=np.float64))

        # cache
        try:
            os.makedirs(os.path.dirname(CFG.EV2TC_CACHE), exist_ok=True)
            with open(CFG.EV2TC_CACHE, "w", encoding="utf-8") as f:
                json.dump({"_fetched_ts": _now_ts(), "map": ev2tc}, f)
            log.info(f"Cache salvo: {CFG.EV2TC_CACHE} ({len(ev2tc)} eventos)")
        except Exception as e:
            log.warning(f"Falha ao salvar cache: {e}")
    except Exception as e:
        log.warning(f"GWOSC offline ou erro na API: {e}. Seguindo sem mapa de eventos.")

    return ev2tc, tc_sorted

def tcs_in_range(tc_sorted: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if tc_sorted.size == 0:
        return tc_sorted
    i0 = int(np.searchsorted(tc_sorted, lo, side="left"))
    i1 = int(np.searchsorted(tc_sorted, hi, side="right"))
    if i0 >= i1:
        return tc_sorted[:0]
    return tc_sorted[i0:i1]

# =========================
# Leitura de um *_windows.hdf5
# =========================
def load_windows_h5(path: str) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        fs = float(attrs.get("fs", 4096.0))
        window_sec = float(attrs.get("window_sec", 2.0))
        stride_sec = float(attrs.get("stride_sec", 0.5))
        source_file = str(attrs.get("source_file", os.path.basename(path)))

        if "windows" not in f:
            raise KeyError("Grupo 'windows' ausente")

        g = f["windows"]
        start_idx = np.array(g.get("start_idx", []))
        start_gps = np.array(g.get("start_gps", []))
        snr_rms = np.array(g.get("snr_rms", np.zeros_like(start_idx, dtype=np.float32)))
        snr_peak = np.array(g.get("snr_peak", np.zeros_like(start_idx, dtype=np.float32)))

    n = int(len(start_idx))
    if n != len(start_gps):
        raise ValueError("start_idx e start_gps com comprimentos diferentes")
    if n > CFG.MAX_WINDOWS_PER_FILE_WARN:
        log.warning(f"{os.path.basename(path)} tem {n} janelas (> {CFG.MAX_WINDOWS_PER_FILE_WARN}). Revise parâmetros.")

    end_gps = start_gps.astype(np.float64) + float(window_sec)

    snr_rms_safe = np.where(np.abs(snr_rms) < 1e-12, 1e-12, snr_rms).astype(np.float32, copy=False)
    crest = np.clip((snr_peak.astype(np.float32) / snr_rms_safe), a_min=0.0, a_max=None)

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
        "crest_factor": crest.astype(np.float32, copy=False)
    })
    return df

# =========================
# Rotulagem
# =========================
def label_windows(df: pd.DataFrame, ev2tc: Dict[str, float], tc_sorted: np.ndarray,
                  t_margin_pre: float, t_margin_pos: float) -> pd.DataFrame:
    df = df.copy()
    df["label"] = np.uint8(0)

    # 1) por hint
    mask_have_hint = df["event_hint"].ne("")
    if mask_have_hint.any() and ev2tc:
        for ev, sub in df[mask_have_hint].groupby("event_hint"):
            tc = ev2tc.get(ev.upper())
            if tc is None:
                continue
            lo, hi = float(tc - t_margin_pre), float(tc + t_margin_pos)
            hit = (sub["end_gps"].to_numpy() >= lo) & (sub["start_gps"].to_numpy() <= hi)
            df.loc[sub.index[hit], "label"] = 1

    # 2) por arquivo
    mask_unlabeled = df["label"].eq(0)
    if mask_unlabeled.any() and tc_sorted.size > 0:
        for _, sub in df[mask_unlabeled].groupby("file_id"):
            sg_min = float(sub["start_gps"].min())
            eg_max = float(sub["end_gps"].max())
            lo_all = sg_min - t_margin_pre
            hi_all = eg_max + t_margin_pos
            tcs = tcs_in_range(tc_sorted, lo_all, hi_all)
            if tcs.size == 0:
                continue
            starts = sub["start_gps"].to_numpy()
            ends   = sub["end_gps"].to_numpy()
            for tc in tcs:
                lo, hi = float(tc - t_margin_pre), float(tc + t_margin_pos)
                hit = (ends >= lo) & (starts <= hi)
                if hit.any():
                    df.loc[sub.index[hit], "label"] = 1

    return df

# =========================
# Split por grupo -> 'subset' com garantias e fallback
# =========================
def _split_by_group(idx_all, groups_all, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(idx_all, groups=groups_all))
    return tr_idx, te_idx

def _random_split(n: int, test_frac: float, val_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * (n - n_test)))
    te = idx[:n_test]
    va = idx[n_test:n_test+n_val]
    tr = idx[n_test+n_val:]
    return tr, va, te

def _enforce_minimums(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cnt = df["subset"].value_counts().to_dict()
    need_tr = max(0, CFG.MIN_TRAIN_ROWS - cnt.get("train", 0))
    need_va = max(0, CFG.MIN_VAL_ROWS - cnt.get("val", 0))
    need_te = max(0, CFG.MIN_TEST_ROWS - cnt.get("test", 0))
    if need_tr <= 0 and need_va <= 0 and need_te <= 0:
        return df

    log.warning(f"Ajustando mínimos: falta train={need_tr} val={need_va} test={need_te}")
    # Estratégia simples: mover do maior para o menor
    def move(src, dst, k):
        if k <= 0:
            return
        take = df.index[df["subset"] == src][:k]
        df.loc[take, "subset"] = dst

    # loop pequeno de balanceamento
    for _ in range(10):
        cnt = df["subset"].value_counts().to_dict()
        need = {"train": max(0, CFG.MIN_TRAIN_ROWS - cnt.get("train", 0)),
                "val":   max(0, CFG.MIN_VAL_ROWS   - cnt.get("val", 0)),
                "test":  max(0, CFG.MIN_TEST_ROWS  - cnt.get("test", 0))}
        if sum(need.values()) == 0:
            break
        # escolhe destino com maior necessidade
        dst = max(need, key=lambda k: need[k])
        if need[dst] == 0:
            break
        # escolhe origem com maior sobra
        over = {k: cnt.get(k, 0) - getattr(CFG, f"MIN_{k.upper()}_ROWS") for k in ["train","val","test"]}
        src = max(over, key=lambda k: over[k])
        if over[src] <= 0:
            # move por tamanho absoluto
            src = max(cnt, key=lambda k: cnt[k])
            if src == dst:
                break
        move(src, dst, min(need[dst], max(1, over.get(src, 1))))
    return df

def make_group_splits(df: pd.DataFrame,
                      test_size: float, val_size: float,
                      seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = df["event_hint"].where(df["event_hint"].ne(""), df["file_id"]).astype(str)

    idx_all = np.arange(len(df))
    groups_all = df["group_id"].to_numpy()

    # TEST
    try:
        trv_idx, te_idx = _split_by_group(idx_all, groups_all, test_size, seed)
    except Exception as e:
        log.warning(f"Split por grupo falhou em TEST: {e}. Usando split aleatório.")
        tr, va, te = _random_split(len(df), test_size, val_size, seed)
        out = df.copy()
        out["subset"] = "train"
        out.loc[va, "subset"] = "val"
        out.loc[te, "subset"] = "test"
        return _enforce_minimums(out.drop(columns=["group_id"]))

    train_val = df.iloc[trv_idx].copy()
    test = df.iloc[te_idx].copy()
    test["subset"] = "test"

    # VAL do restante
    rel_val = val_size / max(1e-12, (1.0 - test_size))
    idx_trv = np.arange(len(train_val))
    groups_trv = train_val["group_id"].astype(str).to_numpy()
    try:
        tr_idx, va_idx = _split_by_group(idx_trv, groups_trv, rel_val, seed + 1)
    except Exception as e:
        log.warning(f"Split por grupo falhou em VAL: {e}. Usando split aleatório no restante.")
        tr, va, _ = _random_split(len(train_val), 0.0, rel_val, seed + 1)
        train = train_val.iloc[tr].copy()
        val   = train_val.iloc[va].copy()
    else:
        train = train_val.iloc[tr_idx].copy()
        val   = train_val.iloc[va_idx].copy()

    train["subset"] = "train"
    val["subset"]   = "val"

    out = pd.concat([train, val, test], ignore_index=True)
    out["subset"] = pd.Categorical(out["subset"], categories=["train", "val", "test"])

    # Garantir mínimos
    out = _enforce_minimums(out)

    # Reordena para estabilidade
    return out.drop(columns=["group_id"]).reset_index(drop=True)

# =========================
# Escrita Parquet em row groups
# =========================
def write_parquet(df: pd.DataFrame, out_path: str, compression: str, rowgroup_size: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = df.copy()

    # 1) Sanitize colunas de texto para evitar None/NaN
    for col, default in [
        ("file_id", ""),
        ("source_file", ""),
        ("event_hint", ""),
        ("detector", "UNK"),
        ("subset", "train"),
    ]:
        if col in df.columns:
            # usa dtype "string" do pandas que preserva nulos, mas já preenchemos para garantir
            df[col] = df[col].astype("string").fillna(default)

    # 2) Tipos numéricos padronizados
    if "label" in df: df["label"] = df["label"].astype(np.uint8)
    for col in ("fs", "window_sec", "stride_sec", "snr_rms", "snr_peak", "crest_factor"):
        if col in df: df[col] = df[col].astype(np.float32)
    if "start_idx" in df: df["start_idx"] = df["start_idx"].astype(np.int64)
    for col in ("start_gps", "end_gps"):
        if col in df: df[col] = df[col].astype(np.float64)

    # 3) Categories opcionais só depois de garantir strings válidas
    for col in ("detector", "event_hint", "subset"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 4) Schema fixo e escrita por row groups
    schema = pa.Table.from_pandas(df.head(0), preserve_index=False).schema
    writer = pq.ParquetWriter(
        out_path, schema,
        compression=compression,
        version="2.6",
        use_dictionary=True
    )
    try:
        step = max(1, int(rowgroup_size))
        for start in range(0, len(df), step):
            chunk = df.iloc[start:start + step]
            table = pa.Table.from_pandas(chunk, preserve_index=False, schema=schema)
            writer.write_table(table)
    finally:
        writer.close()
    log.info(f"Parquet escrito: {out_path} (linhas: {len(df):,})")

# =========================
# MAIN
# =========================
def main():
    os.makedirs(CFG.PROCESSED_DIR, exist_ok=True)

    # 1) localizar janelas
    paths = sorted(glob.glob(os.path.join(CFG.PROCESSED_DIR, "*_windows.hdf5")))
    if not paths:
        log.error("Nenhum *_windows.hdf5 em data/processed. Rode o gerador de janelas antes.")
        return
    log.info(f"Arquivos de janelas encontrados: {len(paths)}")

    # 2) mapa de t_c com cache e fallback
    ev2tc, tc_sorted = fetch_event_coalescence_map()
    log.info(f"Eventos com tc disponíveis: {len(ev2tc)}")

    # 3) ler e rotular por arquivo
    frames: List[pd.DataFrame] = []
    for p in tqdm(paths, desc="Lendo e rotulando", unit="file"):
        try:
            df_file = load_windows_h5(p)
            df_file = label_windows(
                df_file, ev2tc, tc_sorted,
                t_margin_pre=CFG.T_MARGIN_PRE,
                t_margin_pos=CFG.T_MARGIN_POS
            )
            frames.append(df_file)
        except Exception as e:
            log.error(f"Falha em {os.path.basename(p)}: {e}")

    if not frames:
        log.error("Nenhum dado válido após leitura e rotulagem.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # 4) splits com garantias
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
        "label", "subset"
    ]
    missing = [c for c in keep_cols if c not in df_all.columns]
    if missing:
        raise RuntimeError(f"Faltando colunas no dataset final: {missing}")

    df_all = df_all[keep_cols].copy()

    # 6) preview
    try:
        df_all.head(2000).to_csv(CFG.OUT_PREVIEW, index=False)
        log.info(f"Preview salvo: {CFG.OUT_PREVIEW}")
    except Exception as e:
        log.warning(f"Falha ao salvar preview: {e}")

    # 7) parquet
    # Blindagem extra contra nulos em strings
    for col, default in [("file_id",""), ("source_file",""), ("event_hint",""), ("detector","UNK"), ("subset","train")]:
        if col in df_all.columns:
            df_all[col] = df_all[col].astype("string").fillna(default)

    write_parquet(
        df=df_all,
        out_path=CFG.OUT_PARQUET,
        compression=CFG.PARQUET_COMPRESSION,
        rowgroup_size=CFG.PARQUET_ROWGROUP_SIZE
    )

    # 8) meta.json
    meta = {
        "t_margin_pre": CFG.T_MARGIN_PRE,
        "t_margin_pos": CFG.T_MARGIN_POS,
        "test_size": CFG.TEST_SIZE,
        "val_size": CFG.VAL_SIZE,
        "min_rows": {
            "train": CFG.MIN_TRAIN_ROWS,
            "val": CFG.MIN_VAL_ROWS,
            "test": CFG.MIN_TEST_ROWS
        },
        "parquet_compression": CFG.PARQUET_COMPRESSION,
        "rowgroup_size": CFG.PARQUET_ROWGROUP_SIZE,
        "n_rows": int(len(df_all)),
        "n_pos": int(df_all["label"].sum()),
        "by_subset": {
            k: {
                "n": int(len(v)),
                "pos": int(v["label"].sum()),
                "pos_pct": float(100.0 * v["label"].sum() / max(len(v), 1))
            }
            for k, v in df_all.groupby("subset")
        }
    }
    with open(CFG.META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Meta salvo: {CFG.META_JSON}")

    # 9) resumo
    tot = len(df_all)
    pos = int(df_all["label"].sum())
    dist = df_all["subset"].value_counts().to_dict()
    log.info("===== RESUMO DATASET =====")
    log.info(f"Linhas totais: {tot:,}")
    log.info(f"Positivas: {pos:,}  ({100.0*pos/max(tot,1):.2f}%)")
    log.info(f"Dist: {dist}")
    log.info(f"Parquet: {CFG.OUT_PARQUET}")
    log.info("===========================")

if __name__ == "__main__":
    main()