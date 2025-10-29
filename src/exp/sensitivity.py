# src/exp/sensitivity.py
# ======================================================================================
# Sensitivity sweep de velocidade vs qualidade para ML, MF1 e MF2
# - Roda variações agressivas de parâmetros que afetam desempenho
# - Coleta métricas de qualidade e tempo de execução
# - Gera CSV e gráficos de impacto por variável (qualidade e velocidade)
#
# Execução típica:
#   python -m src.exp.sensitivity --quick
#   python -m src.exp.sensitivity --full
#
# Saídas em: results/sensitivity/<timestamp>/
#   - trials.csv
#   - impact_quality.png
#   - impact_speed.png
#   - logs/...
# ======================================================================================

from __future__ import annotations
import os, sys, time, json, argparse, math, glob
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------
# Pathing robusto para rodar como módulo
# ---------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)          # .../src
PROJ_ROOT = os.path.dirname(SRC_DIR)     # repo root
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

RESULTS = os.path.join(PROJ_ROOT, "results")
REPORTS = os.path.join(PROJ_ROOT, "reports")
OUTROOT = os.path.join(RESULTS, "sensitivity")

def _imp(*names):
    import importlib
    last_err = None
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise last_err

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
                # ignora diretórios vazios
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

# ---------------------------------------------
# Medidas de qualidade genéricas
# ---------------------------------------------
@dataclass
class TrialResult:
    phase: str              # "ML" | "MF1" | "MF2"
    variant: str            # nome curto do experimento
    params: dict            # dict com alterações
    wall_s: float           # walltime
    quality_ap: float       # métrica principal (AP)
    quality_auc: float      # métrica auxiliar (AUC)
    out_dir: Optional[str]  # pasta de saída do run (se houver)

def _read_baseline_ml_metrics(dirpath: str) -> Tuple[float, float]:
    # metrics.json com {"val": {"auc":..., "ap":...}, "test": {...}}
    m = _json_load_safe(os.path.join(dirpath, "metrics.json"))
    ap = None
    auc = None
    if "val" in m:
        ap = m["val"].get("ap")
        auc = m["val"].get("auc")
    if ap is None or auc is None:
        # fallback: tenta test
        if "test" in m:
            ap = ap if ap is not None else m["test"].get("ap")
            auc = auc if auc is not None else m["test"].get("auc")
    return float(ap) if ap is not None else float("nan"), float(auc) if auc is not None else float("nan")

def _read_mf2_summary(dirpath: str) -> Tuple[float, float]:
    # training.train_mf: summary.json tem best_val_ap / best_val_auc
    s = _json_load_safe(os.path.join(dirpath, "summary.json"))
    ap = s.get("best_val_ap", None)
    auc = s.get("best_val_auc", None)
    return float(ap) if ap is not None else float("nan"), float(auc) if auc is not None else float("nan")

def _read_mf1_summary(dirpath: str) -> Tuple[float, float]:
    # Se mf_baseline escrever algo; caso não, NaN
    # Tenta alguns nomes comuns:
    for name in ["summary.json", "metrics.json", "report.json"]:
        p = os.path.join(dirpath, name)
        if os.path.exists(p):
            s = _json_load_safe(p)
            # tenta achar AP/AUC
            for key in ["val", "validation", "eval", "metrics"]:
                if key in s and isinstance(s[key], dict):
                    ap = s[key].get("ap") or s[key].get("AP") or s[key].get("average_precision")
                    auc = s[key].get("auc") or s[key].get("AUC") or s[key].get("roc_auc")
                    if ap is not None or auc is not None:
                        return float(ap) if ap is not None else float("nan"), float(auc) if auc is not None else float("nan")
    return float("nan"), float("nan")

# ---------------------------------------------
# Runners dos três estágios
# ---------------------------------------------
def run_ml(params: dict) -> TrialResult:
    """
    Roda src.eval.baseline_ml.main() com config default.
    Variações aqui são mínimas (o baseline_ml já é CPU estavel).
    """
    bml = _imp("src.eval.baseline_ml", "eval.baseline_ml")
    out_before = _latest_subdir(os.path.join(REPORTS, "baseline_ml"))
    _, wall = _timeit(bml.main)  # usa defaults; o script já grava em reports/baseline_ml/<ts>/
    out_after = _latest_subdir(os.path.join(REPORTS, "baseline_ml"))
    out_dir = out_after if out_after != out_before else out_after
    ap, auc = (float("nan"), float("nan"))
    if out_dir:
        ap, auc = _read_baseline_ml_metrics(out_dir)
    return TrialResult("ML", params.get("_name", "ml_default"), params, wall, ap, auc, out_dir)

def run_mf1(params: dict) -> TrialResult:
    """
    Roda src.eval.mf_baseline.main() se existir.
    """
    mbf = _imp("src.eval.mf_baseline", "eval.mf_baseline")
    # Muitos scripts de baseline expõem main() sem parâmetros.
    out_before = _latest_subdir(os.path.join(REPORTS, "mf_baseline"))
    if hasattr(mbf, "main"):
        _, wall = _timeit(mbf.main)
    else:
        # Se não houver main(), não falha: retorna NaN rápido
        return TrialResult("MF1", params.get("_name", "mf1_skip"), params, 0.0, float("nan"), float("nan"), None)
    out_after = _latest_subdir(os.path.join(REPORTS, "mf_baseline"))
    out_dir = out_after if out_after != out_before else out_after
    ap, auc = (float("nan"), float("nan"))
    if out_dir:
        ap, auc = _read_mf1_summary(out_dir)
    return TrialResult("MF1", params.get("_name", "mf1_default"), params, wall, ap, auc, out_dir)

def run_mf2(params: dict) -> TrialResult:
    """
    Roda src.training.train_mf.train(hparams=...) com overrides agressivos.
    """
    tmf = _imp("src.training.train_mf", "training.train_mf")
    # HParams defaults; vamos sobrescrever itens críticos
    overrides = dict(
        EPOCHS=params.get("EPOCHS"),
        BATCH=params.get("BATCH"),
        NUM_WORKERS=params.get("NUM_WORKERS"),
        PIN_MEMORY=params.get("PIN_MEMORY"),
        AMP=params.get("AMP"),
        DEVICE=params.get("DEVICE"),
        TIME_DROPOUT_P=params.get("TIME_DROPOUT_P"),
        AUG_TIMESHIFT_SEC=params.get("AUG_TIMESHIFT_SEC"),
        AUG_GAUSS_STD=params.get("AUG_GAUSS_STD")
    )
    # remove None para não quebrar hasattr/assignment
    overrides = {k: v for k, v in overrides.items() if v is not None}
    def _train():
        return tmf.train(hparams=overrides)

    summary, wall = _timeit(_train)
    out_dir = None
    ap = auc = float("nan")
    try:
        # O train() imprime mas também salvou summary.json no runs/mf/<ts>/
        # Precisamos localizá-lo. Procura o mais recente em runs/mf.
        runs_mf = os.path.join(PROJ_ROOT, "runs", "mf")
        latest = _latest_subdir(runs_mf)
        if latest and os.path.exists(os.path.join(latest, "summary.json")):
            out_dir = latest
            ap, auc = _read_mf2_summary(latest)
        # Se o retorno já vier com paths, respeita
        if isinstance(summary, dict):
            maybe_best = summary.get("best_ap_path") or summary.get("best_auc_path")
            if maybe_best:
                out_dir = os.path.dirname(maybe_best)
                ap = float(summary.get("best_val_ap", ap))
                auc = float(summary.get("best_val_auc", auc))
    except Exception:
        pass
    return TrialResult("MF2", params.get("_name", "mf2_default"), params, wall, ap, auc, out_dir)

# ---------------------------------------------
# Varreduras
# ---------------------------------------------
def grid_quick() -> List[dict]:
    """
    Varredura curta, focada em velocidade e diagnósticos rápidos.
    """
    g: List[dict] = []

    # ML baseline: sempre roda 1
    g.append(dict(_phase="ML", _name="ml_default"))

    # MF1 baseline: 1 passada
    g.append(dict(_phase="MF1", _name="mf1_default"))

    # MF2: três variações chave
    g += [
        dict(_phase="MF2", _name="mf2_cpu_small",
             DEVICE="cpu", AMP=False, EPOCHS=1, BATCH=128, NUM_WORKERS=2, PIN_MEMORY=False,
             TIME_DROPOUT_P=0.02, AUG_TIMESHIFT_SEC=0.0, AUG_GAUSS_STD=0.0),
        dict(_phase="MF2", _name="mf2_cuda_fast",
             DEVICE="cuda", AMP=True, EPOCHS=2, BATCH=256, NUM_WORKERS=8, PIN_MEMORY=True,
             TIME_DROPOUT_P=0.02, AUG_TIMESHIFT_SEC=0.005, AUG_GAUSS_STD=0.01),
        dict(_phase="MF2", _name="mf2_cuda_ultra",
             DEVICE="cuda", AMP=True, EPOCHS=2, BATCH=384, NUM_WORKERS=12, PIN_MEMORY=True,
             TIME_DROPOUT_P=0.03, AUG_TIMESHIFT_SEC=0.010, AUG_GAUSS_STD=0.015),
    ]
    return g

def grid_full() -> List[dict]:
    """
    Varredura maior, ainda focada em velocidade, mas com mais combinações.
    """
    g: List[dict] = []
    g.append(dict(_phase="ML", _name="ml_default"))
    g.append(dict(_phase="MF1", _name="mf1_default"))
    mf2_variants = [
        # CPU baseline curto
        dict(_name="mf2_cpu_small", DEVICE="cpu", AMP=False, EPOCHS=2, BATCH=128, NUM_WORKERS=4, PIN_MEMORY=False,
             TIME_DROPOUT_P=0.02, AUG_TIMESHIFT_SEC=0.0, AUG_GAUSS_STD=0.0),
        # CUDA combos
        dict(_name="mf2_cuda_xs", DEVICE="cuda", AMP=True, EPOCHS=2, BATCH=192, NUM_WORKERS=6, PIN_MEMORY=True,
             TIME_DROPOUT_P=0.02, AUG_TIMESHIFT_SEC=0.005, AUG_GAUSS_STD=0.01),
        dict(_name="mf2_cuda_s", DEVICE="cuda", AMP=True, EPOCHS=3, BATCH=256, NUM_WORKERS=8, PIN_MEMORY=True,
             TIME_DROPOUT_P=0.02, AUG_TIMESHIFT_SEC=0.005, AUG_GAUSS_STD=0.01),
        dict(_name="mf2_cuda_m", DEVICE="cuda", AMP=True, EPOCHS=3, BATCH=320, NUM_WORKERS=12, PIN_MEMORY=True,
             TIME_DROPOUT_P=0.03, AUG_TIMESHIFT_SEC=0.010, AUG_GAUSS_STD=0.015),
        dict(_name="mf2_cuda_l", DEVICE="cuda", AMP=True, EPOCHS=4, BATCH=384, NUM_WORKERS=16, PIN_MEMORY=True,
             TIME_DROPOUT_P=0.03, AUG_TIMESHIFT_SEC=0.010, AUG_GAUSS_STD=0.015),
    ]
    for v in mf2_variants:
        v2 = dict(_phase="MF2", **v)
        g.append(v2)
    return g

# ---------------------------------------------
# Execução das varreduras
# ---------------------------------------------
def run_trials(out_dir: str, quick: bool) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    grid = grid_quick() if quick else grid_full()
    rows: List[dict] = []

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
        # adiciona colunas com os params variáveis
        for k, v in spec.items():
            if k.startswith("_"):  # meta
                continue
            row[f"param_{k}"] = v
        rows.append(row)

        # salva parcial
        df_tmp = pd.DataFrame(rows)
        df_tmp.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    print(f"[RUN] Sensitivity output: {out_dir}")
    return df

# ---------------------------------------------
# Análise de impacto
# ---------------------------------------------
def _effect_per_param(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Mede efeito médio de cada parâmetro sobre 'metric' comparando subgrupos.
    Retorna ranking por |delta_medio|, ignorando NaN.
    """
    met = df[metric].astype(float)
    effects = []
    # considera apenas colunas param_*
    for col in sorted([c for c in df.columns if c.startswith("param_")]):
        # só aceita colunas com variação
        if df[col].nunique(dropna=True) <= 1:
            continue
        # normaliza representações
        vals = df[col]
        # calcula média por valor e delta relativo à média global
        global_mean = np.nanmean(met.values)
        part = []
        for v in sorted(vals.dropna().unique(), key=lambda x: str(x)):
            m = np.nanmean(met[vals == v].values)
            if not np.isnan(m):
                part.append((v, m, m - global_mean))
        if not part:
            continue
        # magnitude média do delta
        deltas = [abs(x[2]) for x in part]
        mean_abs_delta = float(np.mean(deltas)) if deltas else 0.0
        effects.append(dict(param=col, mean_abs_delta=mean_abs_delta, global_mean=float(global_mean)))
    eff = pd.DataFrame(effects).sort_values("mean_abs_delta", ascending=False)
    return eff

def plot_impacts(df: pd.DataFrame, out_dir: str):
    # Impacto sobre qualidade (usa quality_ap por padrão)
    eff_q = _effect_per_param(df, "quality_ap")
    if not eff_q.empty:
        plt.figure(figsize=(8, max(3, 0.4*len(eff_q))))
        plt.barh(eff_q["param"][::-1], eff_q["mean_abs_delta"][::-1], color="#2ecc71")
        plt.xlabel("Mean |delta(AP)| vs média global")
        plt.title("Variáveis com maior impacto em qualidade")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "impact_quality.png"), dpi=160)
        plt.close()

    # Impacto sobre velocidade (negativo do tempo)
    df = df.copy()
    df["speed"] = -df["wall_s"].astype(float)
    eff_s = _effect_per_param(df, "speed")
    if not eff_s.empty:
        plt.figure(figsize=(8, max(3, 0.4*len(eff_s))))
        plt.barh(eff_s["param"][::-1], eff_s["mean_abs_delta"][::-1], color="#3498db")
        plt.xlabel("Mean |delta(speed)| vs média global  (speed = -wall_s)")
        plt.title("Variáveis com maior impacto em velocidade")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "impact_speed.png"), dpi=160)
        plt.close()

# ---------------------------------------------
# CLI
# ---------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sensitivity sweep speed vs quality")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="Executa grid curto")
    g.add_argument("--full", action="store_true", help="Executa grid maior")
    args = ap.parse_args()

    out_dir = os.path.join(OUTROOT, _ts())
    df = run_trials(out_dir, quick=args.quick)
    plot_impacts(df, out_dir)

if __name__ == "__main__":
    main()
