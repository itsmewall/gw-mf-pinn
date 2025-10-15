#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Silencia ruído do pykerr
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# MF1 baseline
from eval.mf_baseline import (
    CFG as MF1CFG,
    build_template_bank,
    maybe_load_template_cache,
    score_subset,
    summarize_and_plot,
    build_windows_index,
)

# MF2 treino
from training.train_mf import run_mf_stage2, MFCFG as MF2CFG


def fmtf(x):
    try:
        return "NA" if x is None or not np.isfinite(x) else f"{float(x):.4f}"
    except Exception:
        return "NA"


def safe_auc_ap(y, s):
    y = np.asarray(y).astype(int)
    if y.size == 0 or len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def take_smoke_subset(df, n=200):
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    pos = df[df["label"].astype(int) == 1]
    neg = df[df["label"].astype(int) == 0]
    if len(df) == 0:
        return df
    if len(pos) == 0:
        return df.head(min(n, len(df))).copy()
    k_pos = min(max(1, n // 10), len(pos))   # tenta até 10% de positivos no smoke
    k_neg = min(n - k_pos, len(neg))
    mix = pd.concat([pos.head(k_pos), neg.head(k_neg)], axis=0)
    if len(mix) < n:
        rest = df.drop(mix.index, errors="ignore").head(n - len(mix))
        mix = pd.concat([mix, rest], axis=0)
    return mix.sample(frac=1.0, random_state=123).reset_index(drop=True)


def test_mf1_smoke(out_root: Path):
    print("\n=== MF1: smoke test ===")

    df = pd.read_parquet(MF1CFG.PARQUET_PATH)
    split_col = "subset" if "subset" in df.columns else ("split" if "split" in df.columns else None)
    if split_col is None:
        raise KeyError("dataset.parquet precisa da coluna subset ou split")

    df = df[[split_col, "file_id", "start_gps", "label"]].copy()
    df_val_full = df[df[split_col] == "val"].copy()
    df_test_full = df[df[split_col] == "test"].copy()

    df_val = take_smoke_subset(df_val_full, n=200)
    df_test = take_smoke_subset(df_test_full, n=200)

    idx_map = build_windows_index(MF1CFG.WINDOWS_DIR)
    if not idx_map:
        raise RuntimeError("nenhum *_windows.hdf5 encontrado em WINDOWS_DIR")

    # patch temporário do CFG para rodar rápido
    from contextlib import contextmanager

    @contextmanager
    def patch_cfg(**over):
        keep = {}
        for k, v in over.items():
            keep[k] = getattr(MF1CFG, k)
            setattr(MF1CFG, k, v)
        try:
            yield
        finally:
            for k, v in keep.items():
                setattr(MF1CFG, k, v)

    out_dir = out_root / "mf1"
    out_dir.mkdir(parents=True, exist_ok=True)

    with patch_cfg(
        TEMPLATES_N=16,
        TEMPLATE_SUBSAMPLE=1,
        TOPK_POOL=1,
        MAX_SHIFT_SEC=0.010,
        LAG_STEP=8,
        N_JOBS=-1,
        USE_GPU=False,
    ):
        bank = build_template_bank(MF1CFG.TEMPLATES_N)
        tmpl_cache = maybe_load_template_cache(str(out_dir), bank)

        print(f"[MF1] Scoring VAL n={len(df_val)}")
        s_val = score_subset("VAL", df_val, idx_map, tmpl_cache)
        yv = df_val["label"].to_numpy(int)
        auc_v, ap_v = safe_auc_ap(yv, s_val)
        if auc_v is not None:
            summarize_and_plot(yv, s_val, str(out_dir), "val_smoke")

        print(f"[MF1] Scoring TEST n={len(df_test)}")
        s_test = score_subset("TEST", df_test, idx_map, tmpl_cache)
        yt = df_test["label"].to_numpy(int)
        auc_t, ap_t = safe_auc_ap(yt, s_test)
        if auc_t is not None:
            summarize_and_plot(yt, s_test, str(out_dir), "test_smoke")

    print(f"[MF1] OK. AUC_val={fmtf(auc_v)} AP_val={fmtf(ap_v)} | AUC_test={fmtf(auc_t)} AP_test={fmtf(ap_t)}")
    return {"val": {"auc": auc_v, "ap": ap_v}, "test": {"auc": auc_t, "ap": ap_t}, "out_dir": str(out_dir)}


def test_mf2_smoke(out_root: Path):
    print("\n=== MF2: smoke test ===")
    # reduzir custo do smoke
    MF2CFG.EPOCHS = 1
    MF2CFG.BATCH = 64
    MF2CFG.MIXED_PREC = False
    MF2CFG.OUT_ROOT = str(out_root / "mf2")
    Path(MF2CFG.OUT_ROOT).mkdir(parents=True, exist_ok=True)
    run_mf_stage2()
    return {"out_dir": MF2CFG.OUT_ROOT}


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path("reports/tests") / ts
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saída de testes: {out_root}")

    res1 = test_mf1_smoke(out_root)
    res2 = test_mf2_smoke(out_root)

    with open(out_root / "summary.json", "w") as f:
        json.dump({"mf1": res1, "mf2": res2, "ts": ts}, f, indent=2)

    print("\n[OK] Smoke MF1+MF2 finalizado")


if __name__ == "__main__":
    main()
