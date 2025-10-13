# src/eval/baseline_ml.py
# -------------------------------------------------------------------------------------------------
# Baseline ML (CPU por padrão) com opção de GPU via XGBoost:
# - Gera 'split' por grupos (file_id) se não existir (70/15/15) sem vazamento entre janelas
# - Downsampling de negativos no treino
# - Calibração opcional (sigmoid)
# - Estratégias de limiar: best_f1 e FAR grid
# - Salva métricas, plots, scores_test e model.joblib (com asdict(CFG))
# - Se criar split, escreve data/processed/dataset_with_split.parquet
#
# Requisitos:
#   pip install pandas numpy scikit-learn matplotlib seaborn pyarrow joblib tqdm
#   (opcional, GPU) pip install xgboost>=2.0
#
# Para usar GPU: defina CFG.MODEL="xgb". O XGBoost usará CUDA com tree_method="gpu_hist".
# -------------------------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_context("talk"); sns.set_style("whitegrid")
except Exception:
    pass

from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GroupShuffleSplit
import joblib

# XGBoost (opcional; só necessário se MODEL="xgb")
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Cfg:
    DATASET: str = os.path.join("data", "processed", "dataset.parquet")
    DATASET_WITH_SPLIT: str = os.path.join("data", "processed", "dataset_with_split.parquet")
    OUT_DIR: str = os.path.join("reports", "baseline_ml")

    # MODEL: "logreg" | "rf" | "xgb" (GPU via XGBoost)
    MODEL: str = "xgb"
    SEED: int = 42
    VERBOSE: bool = True

    # ---- LogReg
    LOGREG_C: float = 2.0
    LOGREG_PENALTY: str = "l2"        # "l1"|"l2"
    LOGREG_MAX_ITER: int = 800

    # ---- RandomForest
    RF_N_EST: int = 600
    RF_MAX_DEPTH: Optional[int] = None
    RF_MIN_SAMPLES_LEAF: int = 1
    RF_N_JOBS: int = -1
    RF_CLASS_WEIGHT: Optional[str] = "balanced_subsample"

    # ---- XGBoost (GPU)
    XGB_N_EST: int = 800
    XGB_MAX_DEPTH: int = 8
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE: float = 0.8
    XGB_REG_L1: float = 0.0
    XGB_REG_L2: float = 1.0
    XGB_MAX_BIN: int = 256
    XGB_EVAL_METRIC: str = "auc"
    XGB_TREE_METHOD: str = "gpu_hist"    # "gpu_hist" para GPU, "hist" para CPU
    XGB_PREDICTOR: str = "gpu_predictor" # "gpu_predictor" ou "auto"
    XGB_N_JOBS: int = 0                  # 0 = deixe a GPU trabalhar

    # ---- Features
    NUM_COLS: tuple = ("snr_rms", "snr_peak", "crest_factor", "window_sec", "stride_sec")
    CAT_COLS: tuple = ("detector",)

    # aplicar log1p apenas nestas numéricas (por nome)
    APPLY_LOG1P_ON: tuple = ("snr_rms", "snr_peak", "crest_factor")

    # ---- Estratégias de threshold
    THRESH_STRATEGY: str = "best_f1"  # "best_f1" | "target_far"
    TARGET_FAR: float = 0.01          # se usar "target_far": FPR alvo (1%)
    FAR_GRID: tuple = (0.001, 0.005, 0.01, 0.02, 0.05)

    # ---- Calibração
    CALIBRATE_PROBA: bool = True      # bom p/ RF e às vezes p/ LogReg/XGB

    # ---- Downsampling de negativos no TREINO
    ENABLE_NEG_DOWNSAMPLE: bool = True
    MAX_NEG_PER_POS: int = 50

CFG = Cfg()


# =========================
# Logging helpers
# =========================
def _log(msg: str):
    if CFG.VERBOSE:
        print(msg)

def ts_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Split helpers
# =========================
REQUIRED_COLS = {"label", *CFG.NUM_COLS, *CFG.CAT_COLS}

def _need_split(df: pd.DataFrame) -> bool:
    return "split" not in df.columns

def _check_required_cols(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. "
                         f"Columns present: {list(df.columns)[:20]}...")

def _ensure_group_col(df: pd.DataFrame) -> pd.Series:
    # preferir file_id; se não houver, use event_hint; senão, cria grupo por arquivo lógico (start_gps+detector)
    if "file_id" in df.columns:
        g = df["file_id"].astype(str)
    elif "event_hint" in df.columns:
        g = df["event_hint"].astype(str)
    else:
        base = (df.get("start_gps", pd.Series(index=df.index, data=-1)).astype(str) +
                "_" + df.get("detector", pd.Series(index=df.index, data="UNK")).astype(str))
        g = base
    return g

def _assign_split_groups(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Cria uma coluna 'split' por grupos (sem vazamento entre janelas)."""
    groups = _ensure_group_col(df)
    idx = np.arange(len(df))

    # 1) train vs temp (val+test)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    tr_idx, temp_idx = next(gss.split(idx, groups=groups))
    # 2) temp -> val vs test (50/50 do temp -> 15/15 geral)
    temp_groups = groups.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed+1)
    va_rel, te_rel = next(gss2.split(temp_idx, groups=temp_groups))
    va_idx = temp_idx[va_rel]
    te_idx = temp_idx[te_rel]

    split = pd.Series(index=df.index, data="", dtype="object")
    split.iloc[tr_idx] = "train"
    split.iloc[va_idx] = "val"
    split.iloc[te_idx] = "test"
    return split

def ensure_splits(df: pd.DataFrame, save_sidecar: bool = True) -> pd.DataFrame:
    _check_required_cols(df)
    if not _need_split(df):
        return df
    _log("[SPLIT] Coluna 'split' ausente — gerando via GroupShuffleSplit (por grupo/file_id).")
    df = df.copy()
    df["split"] = _assign_split_groups(df, seed=CFG.SEED).values
    ct = df["split"].value_counts().to_dict()
    _log(f"[SPLIT] Feito: train={ct.get('train',0):,}  val={ct.get('val',0):,}  test={ct.get('test',0):,}")
    if save_sidecar:
        try:
            df.to_parquet(CFG.DATASET_WITH_SPLIT, index=False)
            _log(f"[SPLIT] Sidecar salvo em: {CFG.DATASET_WITH_SPLIT}")
        except Exception as e:
            _log(f"[SPLIT] Aviso: falha ao salvar sidecar: {e}")
    return df


# =========================
# Data utils
# =========================
def split_sets(df: pd.DataFrame):
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()
    return tr, va, te

def to_xy(df: pd.DataFrame):
    X = df[list(CFG.NUM_COLS) + list(CFG.CAT_COLS)].copy()
    y = df["label"].astype(np.uint8).to_numpy()
    for c in CFG.CAT_COLS:
        X[c] = X[c].astype(str)
    return X, y

def downsample_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """Mantém todos os positivos e amostra negativos até MAX_NEG_PER_POS."""
    if not CFG.ENABLE_NEG_DOWNSAMPLE:
        return train_df
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0]
    n_pos = len(pos)
    if n_pos == 0 or len(neg) == 0:
        return train_df
    max_neg = min(len(neg), CFG.MAX_NEG_PER_POS * n_pos)
    neg_sample = neg.sample(n=max_neg, random_state=CFG.SEED, replace=False)
    out = pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1.0, random_state=CFG.SEED)
    return out


# =========================
# Transformações numéricas (TOP-LEVEL, picklável)
# =========================
def _log1p_indices(X: np.ndarray, indices: tuple) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_out = X.copy()
    if len(indices) > 0:
        X_out[:, indices] = np.log1p(np.clip(X_out[:, indices], 0.0, None))
    return X_out

def make_numeric_transform():
    idx = tuple(i for i, c in enumerate(CFG.NUM_COLS) if c in CFG.APPLY_LOG1P_ON)
    log_step = FunctionTransformer(
        func=_log1p_indices,
        kw_args={"indices": idx},
        feature_names_out="one-to-one",
        validate=False
    )
    num = Pipeline(steps=[
        ("log1p_sel", log_step),
        ("scaler", StandardScaler())
    ])
    return num


# =========================
# Model building
# =========================
def build_estimator(model_name: str):
    if model_name == "logreg":
        base = LogisticRegression(
            C=CFG.LOGREG_C,
            penalty=CFG.LOGREG_PENALTY,
            solver="liblinear" if CFG.LOGREG_PENALTY == "l1" else "lbfgs",
            max_iter=CFG.LOGREG_MAX_ITER,
            class_weight="balanced",
            random_state=CFG.SEED,
        )
        return CalibratedClassifierCV(base, cv=3, method="sigmoid") if CFG.CALIBRATE_PROBA else base

    if model_name == "rf":
        base = RandomForestClassifier(
            n_estimators=CFG.RF_N_EST,
            max_depth=CFG.RF_MAX_DEPTH,
            min_samples_leaf=CFG.RF_MIN_SAMPLES_LEAF,
            n_jobs=CFG.RF_N_JOBS,
            class_weight=CFG.RF_CLASS_WEIGHT,
            random_state=CFG.SEED,
        )
        return CalibratedClassifierCV(base, cv=3, method="sigmoid") if CFG.CALIBRATE_PROBA else base

    if model_name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("XGBoost não instalado. Instale com: pip install xgboost>=2.0")
        base = xgb.XGBClassifier(
            n_estimators=CFG.XGB_N_EST,
            max_depth=CFG.XGB_MAX_DEPTH,
            learning_rate=CFG.XGB_LR,
            subsample=CFG.XGB_SUBSAMPLE,
            colsample_bytree=CFG.XGB_COLSAMPLE,
            reg_alpha=CFG.XGB_REG_L1,
            reg_lambda=CFG.XGB_REG_L2,
            max_bin=CFG.XGB_MAX_BIN,
            objective="binary:logistic",
            tree_method=CFG.XGB_TREE_METHOD,     # "gpu_hist" => GPU
            predictor=CFG.XGB_PREDICTOR,         # "gpu_predictor"
            eval_metric=CFG.XGB_EVAL_METRIC,
            random_state=CFG.SEED,
            n_jobs=CFG.XGB_N_JOBS
        )
        # Calibração ajuda a calibrar probas do XGB (opcional)
        return CalibratedClassifierCV(base, cv=3, method="sigmoid") if CFG.CALIBRATE_PROBA else base

    raise ValueError(f"MODEL inválido: {model_name}")

def build_pipeline(model_name: str) -> Pipeline:
    num = make_numeric_transform()
    # Compat: sklearn>=1.2 usa 'sparse_output'; versões antigas usam 'sparse'
    try:
        cat = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    except TypeError:
        cat = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num, list(CFG.NUM_COLS)),
            ("cat", cat, list(CFG.CAT_COLS)),
        ],
        remainder="drop"
    )
    clf = build_estimator(model_name)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

def safe_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return np.clip(p, 0.0, 1.0)
    if hasattr(clf, "decision_function"):
        d = model.decision_function(X).astype(float)
        d = (d - d.min()) / max(d.max() - d.min(), 1e-12)
        return d
    y = model.predict(X).astype(float)
    return np.clip(y, 0.0, 1.0)


# =========================
# Thresholding
# =========================
def find_threshold_best_f1(y_true: np.ndarray, y_score: np.ndarray):
    y_true = y_true.astype(np.uint8, copy=False)
    yq = np.round(np.clip(y_score, 0.0, 1.0), 4)
    thr = np.unique(np.concatenate(([0.0], yq, [1.0])))

    order = np.argsort(-yq, kind="mergesort")
    yt = y_true[order]
    scores_sorted = yq[order]
    tp_pref = np.cumsum(yt == 1)
    fp_pref = np.cumsum(yt == 0)
    n_pos = int((y_true == 1).sum())

    best_f1, best_thr, best_stats = -1.0, 0.5, {}
    for t in tqdm(thr, desc="[VAL] threshold scan (best_f1)", unit="thr", leave=False):
        i = np.searchsorted(-scores_sorted, -t, side="right") - 1
        if i < 0:
            tp = 0; fp = 0
        else:
            tp = int(tp_pref[i]); fp = int(fp_pref[i])
        fn = n_pos - tp
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
            best_stats = {"f1": f1, "precision": prec, "recall": rec,
                          "tn": None, "fp": fp, "fn": fn, "tp": tp}
    return float(best_thr), best_stats

def find_threshold_by_far(y_true: np.ndarray, y_score: np.ndarray, far: float):
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    ys = y_score[order]
    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())

    fp = 0; tp = 0
    best = {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0, "precision": 1.0, "recall": 0.0}
    for i in tqdm(range(len(ys)), desc=f"[VAL] FAR≤{far:.3%}", unit="cut", leave=False):
        thr = ys[i]
        if yt[i] == 1: tp += 1
        else: fp += 1
        fpr = fp / max(n0, 1); tpr = tp / max(n1, 1)
        prec = tp / max(tp + fp, 1); rec = tpr
        if fpr <= far:
            best = {"threshold": float(thr), "fpr": float(fpr), "tpr": float(tpr),
                    "precision": float(prec), "recall": float(rec)}
            break
    return float(best["threshold"]), best


# =========================
# Plots + Features
# =========================
def plot_roc_pr(y_true, y_score, out_dir: str, tag: str):
    ensure_dir(out_dir)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_roc = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.4f}")
        plt.plot([0,1], [0,1], 'k--', alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {tag}"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"), dpi=160); plt.close()
    except Exception:
        pass
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(rec, prec, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {tag}"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"pr_{tag}.png"), dpi=160); plt.close()
    except Exception:
        pass

def plot_conf_mat(tn, fp, fn, tp, out_dir: str, tag: str):
    ensure_dir(out_dir)
    cm = np.array([[tn, fp],[fn, tp]], dtype=int)
    plt.figure(figsize=(4.6,4.2))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="black", fontsize=12)
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    plt.title(f"Confusion — {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cm_{tag}.png"), dpi=160); plt.close()

def feature_names_after_pre(pipe: Pipeline, X_sample: pd.DataFrame) -> List[str]:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    out_names: List[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            out_names += list(cols)
        elif name == "cat":
            oh: OneHotEncoder = trans.named_steps["onehot"]
            out_names += list(oh.get_feature_names_out(cols))
    return out_names


# =========================
# Main
# =========================
def main():
    t_all = time.time()
    assert os.path.exists(CFG.DATASET), f"Dataset não encontrado: {CFG.DATASET}"
    _log("[1/6] Lendo dataset…")
    df = pd.read_parquet(CFG.DATASET)

    # --- Garante 'split' se faltar ---
    df = ensure_splits(df, save_sidecar=True)

    tr, va, te = split_sets(df)
    _log(f"   • train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    _log("[2/6] Downsampling do treino…")
    tr_ds = downsample_train(tr)
    _log(f"   • treino (após downsample): {len(tr_ds):,}  (pos={int((tr_ds['label']==1).sum()):,}, neg={int((tr_ds['label']==0).sum()):,})")

    _log("[3/6] Preparando X/y…")
    Xtr, ytr = to_xy(tr_ds)
    Xva, yva = to_xy(va)
    Xte, yte = to_xy(te)

    _log("[4/6] Treinando modelo…")
    pipe = build_pipeline(CFG.MODEL)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    train_time = time.time() - t0
    _log(f"   • tempo de treino: {train_time:.2f}s")

    _log("[5/6] Avaliando e escolhendo limiares (val)…")
    p_va = safe_proba(pipe, Xva)
    print(f"   • val: y=1 -> {(yva==1).sum()} | y=0 -> {(yva==0).sum()} | scores[min/med/max]=({p_va.min():.4g}/{np.median(p_va):.4g}/{p_va.max():.4g})")
    thr_best_f1, stats_best = find_threshold_best_f1(yva, p_va)
    far_thresholds = {}
    for far in CFG.FAR_GRID:
        thr_far, info = find_threshold_by_far(yva, p_va, far=far)
        far_thresholds[str(far)] = info
    thr_main = thr_best_f1 if CFG.THRESH_STRATEGY == "best_f1" else far_thresholds[str(CFG.TARGET_FAR)]["threshold"]

    _log("[6/6] Avaliando no TEST…")
    p_te = safe_proba(pipe, Xte)
    yhat_te = (p_te >= thr_main).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(yte, yhat_te, labels=[0,1]).ravel()

    metrics = {
        "model": CFG.MODEL,
        "train_rows": int(len(tr_ds)),
        "val_rows": int(len(va)),
        "test_rows": int(len(te)),
        "train_time_sec": round(train_time, 3),
        "threshold_strategy": CFG.THRESH_STRATEGY,
        "chosen_threshold": float(thr_main),
        "downsampling": {"enabled": bool(CFG.ENABLE_NEG_DOWNSAMPLE), "max_neg_per_pos": int(CFG.MAX_NEG_PER_POS)},
        "val": {
            "roc_auc": float(roc_auc_score(yva, p_va)) if len(np.unique(yva))>1 else None,
            "pr_ap":  float(average_precision_score(yva, p_va)) if len(np.unique(yva))>1 else None,
            "best_f1": stats_best
        },
        "val_far_thresholds": far_thresholds,
        "test": {
            "roc_auc": float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else None,
            "pr_ap":  float(average_precision_score(yte, p_te)) if len(np.unique(yte))>1 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(tp / max(tp+fp, 1)),
            "recall":    float(tp / max(tp+fn, 1)),
            "fpr":       float(fp / max((te['label']==0).sum(), 1))
        }
    }

    out_dir = os.path.join(CFG.OUT_DIR, ts_now()); ensure_dir(out_dir)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_f1": {"threshold": float(thr_best_f1), **stats_best},
            "far_grid_val": far_thresholds,
            "strategy_used": CFG.THRESH_STRATEGY,
            "threshold_used": float(thr_main)
        }, f, indent=2)

    # Plots + relatório + scores
    plot_roc_pr(yva, p_va, out_dir, tag="val")
    plot_roc_pr(yte, p_te, out_dir, tag="test")
    plot_conf_mat(tn, fp, fn, tp, out_dir, tag="test")

    try:
        report = classification_report(yte, yhat_te, digits=4)
        with open(os.path.join(out_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
            f.write(report)
    except Exception:
        pass

    try:
        dump_cols = ["file_id", "event_hint", "detector", "start_gps",
                     "window_sec", "stride_sec", "snr_rms", "snr_peak", "crest_factor", "label"]
        subset = te[dump_cols].copy()
        subset["score"] = p_te
        subset["pred"] = (p_te >= thr_main).astype(np.uint8)
        subset.to_parquet(os.path.join(out_dir, "scores_test.parquet"), index=False)
    except Exception:
        pass

    # salva modelo (picklável)
    joblib.dump({"pipeline": pipe, "threshold": float(thr_main), "cfg": asdict(CFG)},
                os.path.join(out_dir, "model.joblib"),
                compress=3)

    _log("\n[OK] Baseline otimizado concluído.")
    _log(f" Saída: {out_dir}")
    _log(f" Train rows (após downsampling): {metrics['train_rows']}")
    _log(f" Test AUC-ROC: {metrics['test']['roc_auc']}")
    _log(f" Test AP (PR): {metrics['test']['pr_ap']}")
    _log(f" Test confusion: tn={metrics['test']['tn']} fp={metrics['test']['fp']} fn={metrics['test']['fn']} tp={metrics['test']['tp']}")
    _log(f" Threshold usado: {metrics['chosen_threshold']:.6f}  | estratégia: {CFG.THRESH_STRATEGY}")
    _log(f" Tempo total: {time.time()-t_all:.2f}s")


if __name__ == "__main__":
    main()
