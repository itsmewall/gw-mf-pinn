# src/exp/sensitivity.py
# ======================================================================================
# Sensitivity 2.0 - impacto real de variáveis em MF2 (e registro de MF1/ML)
#
# Ideia:
# - ML roda 1 vez (controle).
# - MF1 roda 1 vez (controle).
# - MF2 roda várias vezes com combinações que mudam:
#     BATCH, NUM_WORKERS, EPOCHS, AMP, DEVICE, AUG_TIMESHIFT_SEC,
#     AUG_GAUSS_STD, TIME_DROPOUT_P.
#
# Existem 2 grades:
#   1) --wsl-safe  -> pouca variação, workers baixos, não mata o WSL.
#   2) --gpu-full  -> mais variação, para rodar quando puder deixar a máquina trabalhar.
#
# Saída:
#   results/sensitivity/<ts>/
#       trials.csv
#       impact_quality.png
#       impact_speed.png
#
# Observação:
# - MF2 sempre roda com train(hparams=overrides), sem args de CLI, igual quando o run.py chama.
# - Força GPU quando possível.
# ======================================================================================

from __future__ import annotations
import os
import sys
import time
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Pathing
# -------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
PROJ_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

RESULTS = os.path.join(PROJ_ROOT, "results")
REPORTS = os.path.join(PROJ_ROOT, "reports")
OUTROOT = os.path.join(RESULTS, "sensitivity")


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _latest_subdir(base: str) -> Optional[str]:
    if not os.path.isdir(base):
        return None
    subs = []
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p):
            try:
                if os.listdir(p):
                    subs.append(p)
            except Exception:
                pass
    return max(subs, key=os.path.getmtime) if subs else None


def _json_load_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _timeit(fn, *args, **kwargs) -> Tuple[any, float]:
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0


def _imp(*names):
    import importlib
    last_err = None
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise last_err


# -------------------------------------------------------------------------
# Leitura de métricas
# -------------------------------------------------------------------------
def _read_baseline_ml_metrics(dirpath: str) -> Tuple[float, float]:
    m = _json_load_safe(os.path.join(dirpath, "metrics.json"))
    ap = None
    auc = None
    if "val" in m:
        ap = m["val"].get("ap")
        auc = m["val"].get("auc")
    if ap is None or auc is None:
        if "test" in m:
            ap = ap if ap is not None else m["test"].get("ap")
            auc = auc if auc is not None else m["test"].get("auc")
    return float(ap) if ap is not None else float("nan"), float(auc) if auc is not None else float("nan")


def _read_mf1_summary(dirpath: str) -> Tuple[float, float]:
    for name in ["summary.json", "metrics.json", "report.json"]:
        p = os.path.join(dirpath, name)
        if os.path.exists(p):
            s = _json_load_safe(p)
            for key in ["val", "validation", "eval", "metrics"]:
                if key in s and isinstance(s[key], dict):
                    ap = s[key].get("ap") or s[key].get("AP") or s[key].get("average_precision")
                    auc = s[key].get("auc") or s[key].get("AUC") or s[key].get("roc_auc")
                    if ap is not None or auc is not None:
                        return float(ap) if ap is not None else float("nan"), float(auc) if auc is not None else float("nan")
    return float("nan"), float("nan")


def _read_mf2_summary(dirpath: str) -> Tuple[float, float]:
    s = _json_load_safe(os.path.join(dirpath, "summary.json"))
    ap = s.get("best_val_ap")
    auc = s.get("best_val_auc")
    return float(ap) if ap is not None else float("nan"), float(auc) if auc is not None else float("nan")


# -------------------------------------------------------------------------
# Runners
# -------------------------------------------------------------------------
@dataclass
class TrialResult:
    phase: str
    variant: str
    params: dict
    wall_s: float
    quality_ap: float
    quality_auc: float
    out_dir: Optional[str]


def run_ml(params: dict) -> TrialResult:
    bml = _imp("src.eval.baseline_ml", "eval.baseline_ml")
    out_before = _latest_subdir(os.path.join(REPORTS, "baseline_ml"))
    _, wall = _timeit(bml.main)
    out_after = _latest_subdir(os.path.join(REPORTS, "baseline_ml"))
    out_dir = out_after if out_after else out_before
    ap, auc = (float("nan"), float("nan"))
    if out_dir:
        ap, auc = _read_baseline_ml_metrics(out_dir)
    return TrialResult("ML", params.get("_name", "ml_default"), params, wall, ap, auc, out_dir)


def run_mf1(params: dict) -> TrialResult:
    mbf = _imp("src.eval.mf_baseline", "eval.mf_baseline")
    out_before = _latest_subdir(os.path.join(REPORTS, "mf_baseline"))
    if hasattr(mbf, "main"):
        _, wall = _timeit(mbf.main)
    else:
        return TrialResult("MF1", params.get("_name", "mf1_skip"), params, 0.0, float("nan"), float("nan"), None)
    out_after = _latest_subdir(os.path.join(REPORTS, "mf_baseline"))
    out_dir = out_after if out_after else out_before
    ap, auc = (float("nan"), float("nan"))
    if out_dir:
        ap, auc = _read_mf1_summary(out_dir)
    return TrialResult("MF1", params.get("_name", "mf1_default"), params, wall, ap, auc, out_dir)


def run_mf2(params: dict) -> TrialResult:
    tmf = _imp("src.training.train_mf", "training.train_mf")

    # Remove None
    overrides = {k: v for k, v in params.items() if not k.startswith("_") and v is not None}

    def _train():
        return tmf.train(hparams=overrides)

    summary, wall = _timeit(_train)

    out_dir = None
    ap = auc = float("nan")

    try:
        runs_mf = os.path.join(PROJ_ROOT, "runs", "mf")
        latest = _latest_subdir(runs_mf)
        if latest and os.path.exists(os.path.join(latest, "summary.json")):
            out_dir = latest
            ap, auc = _read_mf2_summary(latest)
        if isinstance(summary, dict):
            maybe_best = summary.get("best_ap_path") or summary.get("best_auc_path")
            if maybe_best:
                out_dir = os.path.dirname(maybe_best)
                ap = float(summary.get("best_val_ap", ap))
                auc = float(summary.get("best_val_auc", auc))
    except Exception:
        pass

    return TrialResult("MF2", params.get("_name", "mf2_default"), params, wall, ap, auc, out_dir)


# -------------------------------------------------------------------------
# Grades
# -------------------------------------------------------------------------
def grid_wsl_safe() -> List[dict]:
    """
    Varredura que não mata o WSL.
    Varia poucas coisas, mas já produz mais colunas.
    """
    g: List[dict] = []

    # controles
    g.append(dict(_phase="ML", _name="ml_safe"))
    g.append(dict(_phase="MF1", _name="mf1_safe"))

    # MF2 em GPU, poucos workers, 1 época
    base = dict(
        DEVICE="cuda",
        AMP=True,
        EPOCHS=1,
        PIN_MEMORY=True,
    )

    # batch e workers
    g.append(dict(_phase="MF2", _name="mf2_safe_small",
                  **base,
                  BATCH=160,
                  NUM_WORKERS=2,
                  TIME_DROPOUT_P=0.02,
                  AUG_TIMESHIFT_SEC=0.0,
                  AUG_GAUSS_STD=0.0))

    g.append(dict(_phase="MF2", _name="mf2_safe_med",
                  **base,
                  BATCH=224,
                  NUM_WORKERS=2,
                  TIME_DROPOUT_P=0.02,
                  AUG_TIMESHIFT_SEC=0.005,
                  AUG_GAUSS_STD=0.01))

    # muda só workers para ver impacto real
    g.append(dict(_phase="MF2", _name="mf2_safe_workers4",
                  **base,
                  BATCH=224,
                  NUM_WORKERS=4,
                  TIME_DROPOUT_P=0.02,
                  AUG_TIMESHIFT_SEC=0.005,
                  AUG_GAUSS_STD=0.01))

    # desliga AMP para ver impacto
    g.append(dict(_phase="MF2", _name="mf2_safe_noamp",
                  **base,
                  AMP=False,
                  BATCH=224,
                  NUM_WORKERS=2,
                  TIME_DROPOUT_P=0.02,
                  AUG_TIMESHIFT_SEC=0.005,
                  AUG_GAUSS_STD=0.01))

    # aumenta aumento de dados para ver se qualidade mexe
    g.append(dict(_phase="MF2", _name="mf2_safe_augstrong",
                  **base,
                  BATCH=224,
                  NUM_WORKERS=2,
                  TIME_DROPOUT_P=0.05,
                  AUG_TIMESHIFT_SEC=0.010,
                  AUG_GAUSS_STD=0.020))

    return g


def grid_gpu_full() -> List[dict]:
    """
    Grade mais ampla para ver impacto de verdade.
    Esta aqui vai demorar bem mais.
    """
    g: List[dict] = []
    g.append(dict(_phase="ML", _name="ml_full"))
    g.append(dict(_phase="MF1", _name="mf1_full"))

    base = dict(
        DEVICE="cuda",
        AMP=True,
        PIN_MEMORY=True,
    )

    # 1) variação de batch
    for b in [128, 192, 256, 320]:
        g.append(dict(_phase="MF2", _name=f"mf2_b{b}",
                      **base,
                      EPOCHS=1,
                      BATCH=b,
                      NUM_WORKERS=4,
                      TIME_DROPOUT_P=0.02,
                      AUG_TIMESHIFT_SEC=0.005,
                      AUG_GAUSS_STD=0.01))

    # 2) variação de workers
    for w in [2, 4, 6, 8]:
        g.append(dict(_phase="MF2", _name=f"mf2_w{w}",
                      **base,
                      EPOCHS=1,
                      BATCH=224,
                      NUM_WORKERS=w,
                      TIME_DROPOUT_P=0.02,
                      AUG_TIMESHIFT_SEC=0.005,
                      AUG_GAUSS_STD=0.01))

    # 3) variação de augment que afeta CPU
    aug_cfgs = [
        (0.0, 0.0, 0.0),
        (0.005, 0.01, 0.02),
        (0.010, 0.02, 0.03),
    ]
    for i, (ts, gs, td) in enumerate(aug_cfgs, 1):
        g.append(dict(_phase="MF2", _name=f"mf2_aug{i}",
                      **base,
                      EPOCHS=1,
                      BATCH=224,
                      NUM_WORKERS=4,
                      TIME_DROPOUT_P=td,
                      AUG_TIMESHIFT_SEC=ts,
                      AUG_GAUSS_STD=gs))

    # 4) epochs para ver impacto em qualidade
    for e in [1, 2]:
        g.append(dict(_phase="MF2", _name=f"mf2_ep{e}",
                      **base,
                      EPOCHS=e,
                      BATCH=224,
                      NUM_WORKERS=4,
                      TIME_DROPOUT_P=0.03,
                      AUG_TIMESHIFT_SEC=0.010,
                      AUG_GAUSS_STD=0.015))

    # 5) CPU de controle
    g.append(dict(_phase="MF2", _name="mf2_cpu_control",
                  DEVICE="cpu",
                  AMP=False,
                  EPOCHS=1,
                  BATCH=128,
                  NUM_WORKERS=2,
                  PIN_MEMORY=False,
                  TIME_DROPOUT_P=0.0,
                  AUG_TIMESHIFT_SEC=0.0,
                  AUG_GAUSS_STD=0.0))

    return g


# -------------------------------------------------------------------------
# Análise
# -------------------------------------------------------------------------
def _effect_per_param(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    met = df[metric].astype(float)
    effects = []
    for col in sorted([c for c in df.columns if c.startswith("param_")]):
        if df[col].nunique(dropna=True) <= 1:
            continue
        vals = df[col]
        global_mean = np.nanmean(met.values)
        part = []
        for v in sorted(vals.dropna().unique(), key=lambda x: str(x)):
            m = np.nanmean(met[vals == v].values)
            if not np.isnan(m):
                part.append((v, m, m - global_mean))
        if not part:
            continue
        deltas = [abs(x[2]) for x in part]
        mean_abs_delta = float(np.mean(deltas)) if deltas else 0.0
        effects.append(dict(param=col, mean_abs_delta=mean_abs_delta, global_mean=float(global_mean)))
    eff = pd.DataFrame(effects).sort_values("mean_abs_delta", ascending=False)
    return eff


def plot_impacts(df: pd.DataFrame, out_dir: str):
    eff_q = _effect_per_param(df, "quality_ap")
    if not eff_q.empty:
        plt.figure(figsize=(8, max(3, 0.4*len(eff_q))))
        plt.barh(eff_q["param"][::-1], eff_q["mean_abs_delta"][::-1], color="#2ecc71")
        plt.xlabel("Mean |delta(AP)| vs media global")
        plt.title("Variaveis com maior impacto em qualidade")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "impact_quality.png"), dpi=160)
        plt.close()

    df = df.copy()
    df["speed"] = -df["wall_s"].astype(float)
    eff_s = _effect_per_param(df, "speed")
    if not eff_s.empty:
        plt.figure(figsize=(8, max(3, 0.4*len(eff_s))))
        plt.barh(eff_s["param"][::-1], eff_s["mean_abs_delta"][::-1], color="#3498db")
        plt.xlabel("Mean |delta(speed)| vs media global  (speed = -wall_s)")
        plt.title("Variaveis com maior impacto em velocidade")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "impact_speed.png"), dpi=160)
        plt.close()


# -------------------------------------------------------------------------
# Execução
# -------------------------------------------------------------------------
def run_trials(out_dir: str, mode: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    if mode == "wsl-safe":
        grid = grid_wsl_safe()
    else:
        grid = grid_gpu_full()

    rows: List[dict] = []

    # Configurações de memória para evitar matar o WSL
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for spec in grid:
        phase = spec["_phase"]
        name = spec.get("_name", f"{phase}_trial")
        print(f"[RUN] {phase} :: {name}")

        if phase == "ML":
            tr = run_ml(spec)
        elif phase == "MF1":
            tr = run_mf1(spec)
        elif phase == "MF2":
            tr = run_mf2(spec)
        else:
            print(f"[WARN] phase desconhecida: {phase}")
            continue

        row = dict(
            phase=tr.phase,
            variant=tr.variant,
            wall_s=tr.wall_s,
            quality_ap=tr.quality_ap,
            quality_auc=tr.quality_auc,
            out_dir=str(tr.out_dir) if tr.out_dir else None,
        )
        for k, v in spec.items():
            if k.startswith("_"):
                continue
            row[f"param_{k}"] = v
        rows.append(row)

        # flush parcial
        df_tmp = pd.DataFrame(rows)
        df_tmp.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

        # evita esbarrar no WSL
        time.sleep(3)

        # tenta limpar GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    print(f"[RUN] Sensitivity output: {out_dir}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Sensitivity speed vs quality")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wsl-safe", action="store_true", help="Grade segura para WSL")
    g.add_argument("--gpu-full", action="store_true", help="Grade maior, mais pesada")
    args = ap.parse_args()

    mode = "wsl-safe" if args.wsl_safe else "gpu-full"
    out_dir = os.path.join(OUTROOT, _ts())
    df = run_trials(out_dir, mode)
    plot_impacts(df, out_dir)


if __name__ == "__main__":
    main()