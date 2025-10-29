# src/eval/baseline_ml.py
# -------------------------------------------------------------------------------------------------
# Baseline ML (CPU por padrao) com opcao de GPU via XGBoost 2.x:
# - Cria split por grupos (file_id/evento) se nao existir 70/15/15 sem vazamento
# - Downsampling de negativos no treino
# - Calibracao opcional sigmoid
# - Thresholds best_f1 ou target_far e reporte adicional por recall alvo
# - Salva metricas, plots, scores_test e model.joblib com asdict(CFG)
# - Se criar split, escreve data/processed/dataset_with_split.parquet
#
# Requisitos:
#   pip install pandas numpy scikit-learn matplotlib pyarrow joblib tqdm
#   GPU      pip install xgboost>=2.0
#
# Para usar GPU: CFG.MODEL="xgb" e CFG.XGB_DEVICE="cuda"
# -------------------------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# XGBoost GPU opcional
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

    # MODEL: "logreg" | "rf" | "xgb"
    MODEL: str = "xgb"
    SEED: int = 42
    VERBOSE: bool = True

    # LogReg
    LOGREG_C: float = 2.0
    LOGREG_PENALTY: str = "l2"        # "l1" | "l2"
    LOGREG_MAX_ITER: int = 800

    # RandomForest
    RF_N_EST: int = 600
    RF_MAX_DEPTH: Optional[int] = None
    RF_MIN_SAMPLES_LEAF: int = 1
    RF_N_JOBS: int = -1
    RF_CLASS_WEIGHT: Optional[str] = "balanced_subsample"

    # XGBoost 2.x
    XGB_N_EST: int = 800
    XGB_MAX_DEPTH: int = 8
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE: float = 0.8
    XGB_REG_L1: float = 0.0
    XGB_REG_L2: float = 1.0
    XGB_MAX_BIN: int = 256
    XGB_EVAL_METRIC: str = "aucpr"    # classe rara
    XGB_TREE_METHOD: str = "hist"
    XGB_DEVICE: str = "cuda"          # "cuda" | "cpu"
    XGB_N_JOBS: int = 0               # 0 deixa a GPU trabalhar
    AUTO_SCALE_POS_WEIGHT: bool = True

    # Features
    NUM_COLS: tuple = ("snr_rms", "snr_peak", "crest_factor", "window_sec", "stride_sec")
    CAT_COLS: tuple = ("detector",)
    APPLY_LOG1P_ON: tuple = ("snr_rms", "snr_peak", "crest_factor")

    # Thresholds
    THRESH_STRATEGY: str = "best_f1"  # "best_f1" | "target_far"
    TARGET_FAR: float = 0.01
    FAR_REPORT_GRID: tuple = (1e-6, 5e-6, 1e-5, 5e-4, 1e-3, 5e-3, 1e-2)
    RECALL_TARGET_REPORT: float = 0.30

    # Calibracao
    CALIBRATE_PROBA: bool = True

    # Downsampling
    ENABLE_NEG_DOWNSAMPLE: bool = True
    MAX_NEG_PER_POS: int = 50

CFG = Cfg()


# =========================
# Logging
# =========================
def _log(msg: str):
    if CFG.VERBOSE:
        print(msg, flush=True)

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
    if "file_id" in df.columns:
        g = df["file_id"].astype(str)
    elif "event_hint" in df.columns:
        g = df["event_hint"].astype(str)
    else:
        base = (df.get("start_gps", pd.Series(index=df.index, data=-1)).astype(str)
                + "_" + df.get("detector", pd.Series(index=df.index, data="UNK")).astype(str))
        g = base
    return g

def _assign_split_groups(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    groups = _ensure_group_col(df)
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    tr_idx, temp_idx = next(gss.split(idx, groups=groups))
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
    _log("[SPLIT] 'split' ausente - criando por GroupShuffleSplit por grupo file_id.")
    df = df.copy()
    df["split"] = _assign_split_groups(df, seed=CFG.SEED).values
    ct = df["split"].value_counts().to_dict()
    _log(f"[SPLIT] Feito: train={ct.get('train',0):,}  val={ct.get('val',0):,}  test={ct.get('test',0):,}")
    if save_sidecar:
        try:
            df.to_parquet(CFG.DATASET_WITH_SPLIT, index=False)
            _log(f"[SPLIT] Sidecar salvo: {CFG.DATASET_WITH_SPLIT}")
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
# Transformacoes numericas
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
        ("scaler", StandardScaler()),
        ("to32", FunctionTransformer(lambda X: X.astype("float32"), accept_sparse=True))
    ])
    return num


# =========================
# Model building
# =========================
def build_estimator(model_name: str, extra_params: Optional[Dict] = None):
    extra_params = extra_params or {}

    if model_name == "logreg":
        base = LogisticRegression(
            C=CFG.LOGREG_C,
            penalty=CFG.LOGREG_PENALTY,
            solver="liblinear" if CFG.LOGREG_PENALTY == "l1" else "lbfgs",
            max_iter=CFG.LOGREG_MAX_ITER,
            class_weight="balanced",
            random_state=CFG.SEED,
        )
        if extra_params:
            base.set_params(**extra_params)
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
        if extra_params:
            base.set_params(**extra_params)
        return CalibratedClassifierCV(base, cv=3, method="sigmoid") if CFG.CALIBRATE_PROBA else base

    if model_name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("XGBoost nao instalado. Instale com: pip install xgboost>=2.0")
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
            eval_metric=CFG.XGB_EVAL_METRIC,
            tree_method=CFG.XGB_TREE_METHOD,
            device=CFG.XGB_DEVICE,
            random_state=CFG.SEED,
            n_jobs=CFG.XGB_N_JOBS,
            # evita warning de fallback ao usar inplace_predict com entrada CPU
            use_inplace_predict=False
        )
        if extra_params:
            base.set_params(**extra_params)
        return CalibratedClassifierCV(base, cv=3, method="sigmoid") if CFG.CALIBRATE_PROBA else base

    raise ValueError(f"MODEL invalido: {model_name}")

def build_pipeline(model_name: str, estimator=None) -> Pipeline:
    num = make_numeric_transform()
    try:
        cat = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore",
                                                       sparse_output=True,
                                                       dtype="float32"))])
    except TypeError:
        cat = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore",
                                                       sparse=True,
                                                       dtype="float32"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num, list(CFG.NUM_COLS)),
            ("cat", cat, list(CFG.CAT_COLS)),
        ],
        remainder="drop"
    )
    clf = estimator if estimator is not None else build_estimator(model_name)
    to32 = FunctionTransformer(lambda X: X.astype("float32"), accept_sparse=True)
    pipe = Pipeline(steps=[("pre", pre), ("to32", to32), ("clf", clf)])
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

def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos < 1:
        return 1.0
    return max(n_neg / max(n_pos, 1.0), 1.0)


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
    for t in tqdm(thr, desc="[VAL] threshold scan best_f1", unit="thr", leave=False):
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
    for i in range(len(ys)):
        thr = ys[i]
        if yt[i] == 1: tp += 1
        else: fp += 1
        fpr = fp / max(n0, 1); tpr = tp / max(n1, 1)
        prec = tp / max(tp + fp, 1); rec = tpr
        if fpr <= far:
            best = {"threshold": float(thr), "fpr": float(fpr), "tpr": float(tpr),
                    "precision": float(prec), "recall": float(rec)}
            break

    neg = y_score[y_true == 0]
    if neg.size:
        neg_max = float(np.max(neg))
        if best["threshold"] <= neg_max:
            best["threshold"] = float(np.nextafter(neg_max, np.float32(np.inf)))
            best["fpr"] = float((y_score[y_true==0] >= best["threshold"]).mean())

    return float(best["threshold"]), best

def find_threshold_by_recall(y_true: np.ndarray, y_score: np.ndarray, recall_target: float = 0.30):
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    ys = y_score[order]
    n1 = int((y_true == 1).sum())
    tp = 0
    thr = 1.0
    for i in range(len(ys)):
        if yt[i] == 1:
            tp += 1
        if n1 > 0 and tp / n1 >= recall_target:
            thr = float(ys[i])
            break
    return thr, {"recall_target": float(recall_target), "achieved_recall": float(tp / max(n1,1))}


# =========================
# Plots
# =========================
def plot_roc_pr(y_true, y_score, out_dir: str, tag: str):
    ensure_dir(out_dir)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else float("nan")
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.4f}")
        plt.plot([0,1], [0,1], 'k--', alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {tag}"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"), dpi=160); plt.close()
    except Exception:
        pass
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else float("nan")
        plt.figure(figsize=(6,5))
        plt.plot(rec, prec, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {tag}"); plt.legend()
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
    plt.title(f"Confusion - {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cm_{tag}.png"), dpi=160); plt.close()


# =========================
# Main
# =========================
def main():
    t_all = time.time()
    assert os.path.exists(CFG.DATASET), f"Dataset nao encontrado: {CFG.DATASET}"
    _log("[1/6] Lendo dataset...")
    df = pd.read_parquet(CFG.DATASET)

    df = ensure_splits(df, save_sidecar=True)
    tr, va, te = split_sets(df)
    _log(f"   • train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    _log("[2/6] Downsampling do treino...")
    tr_ds = downsample_train(tr)
    _log(f"   • treino apos downsample: {len(tr_ds):,}  pos={(tr_ds['label']==1).sum():,}  neg={(tr_ds['label']==0).sum():,}")

    _log("[3/6] Preparando X/y...")
    Xtr, ytr = to_xy(tr_ds)
    Xva, yva = to_xy(va)
    Xte, yte = to_xy(te)

    _log("[4/6] Treinando modelo...")
    extra = {}
    if CFG.MODEL == "xgb" and _HAS_XGB and CFG.AUTO_SCALE_POS_WEIGHT:
        extra["scale_pos_weight"] = compute_scale_pos_weight(ytr)

    est = build_estimator(CFG.MODEL, extra_params=extra)
    pipe = build_pipeline(CFG.MODEL, estimator=est)

    t0 = time.time()
    pipe.fit(Xtr, ytr)
    train_time = time.time() - t0
    _log(f"   • tempo de treino: {train_time:.2f}s")

    _log("[5/6] Avaliando e escolhendo limiar val...")
    p_va = safe_proba(pipe, Xva)
    print(f"   • val: y=1 -> {(yva==1).sum()} | y=0 -> {(yva==0).sum()} | scores[min/med/max]=({p_va.min():.4g}/{np.median(p_va):.4g}/{p_va.max():.4g})")

    thr_best_f1, stats_best = find_threshold_best_f1(yva, p_va)
    thr_far, info_far = find_threshold_by_far(yva, p_va, far=float(CFG.TARGET_FAR))

    far_report = {}
    for far in CFG.FAR_REPORT_GRID:
        ttmp, iinfo = find_threshold_by_far(yva, p_va, far=float(far))
        far_report[str(far)] = iinfo

    thr_rec, info_rec = find_threshold_by_recall(yva, p_va, recall_target=float(CFG.RECALL_TARGET_REPORT))

    thr_main = thr_best_f1 if CFG.THRESH_STRATEGY == "best_f1" else thr_far

    _log("[6/6] Avaliando no TEST...")
    p_te = safe_proba(pipe, Xte)
    yhat_te = (p_te >= thr_main).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(yte, yhat_te, labels=[0,1]).ravel()

    def _safe_auc_ap(y, s) -> Tuple[Optional[float], Optional[float]]:
        if np.unique(y).size < 2:
            return None, None
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

    test_auc, test_ap = _safe_auc_ap(yte, p_te)

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
            "best_f1": stats_best,
            "target_far": {"target": float(CFG.TARGET_FAR), **info_far},
            "recall_target": {"target": float(CFG.RECALL_TARGET_REPORT), "threshold": float(thr_rec), **info_rec},
        },
        "val_far_report": far_report,
        "test": {
            "roc_auc": test_auc,
            "pr_ap":  test_ap,
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
            "target_far": {"target": float(CFG.TARGET_FAR), **info_far},
            "recall_target": {"target": float(CFG.RECALL_TARGET_REPORT), "threshold": float(thr_rec), **info_rec},
            "far_report_grid": far_report,
            "strategy_used": CFG.THRESH_STRATEGY,
            "threshold_used": float(thr_main)
        }, f, indent=2)

    plot_roc_pr(yva, p_va, out_dir, tag="val")
    plot_roc_pr(yte, p_te, out_dir, tag="test")
    plot_conf_mat(tn, fp, fn, tp, out_dir, tag="test")

    try:
        report = classification_report(yte, yhat_te, digits=4, zero_division=0)
        with open(os.path.join(out_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
            f.write(report)
    except Exception:
        pass

    try:
        cols_keep = [c for c in ["file_id", "event_hint", "detector", "start_gps",
                                 "window_sec", "stride_sec", "snr_rms", "snr_peak", "crest_factor", "label"]
                     if c in te.columns]
        subset = te[cols_keep].copy()
        subset["score"] = p_te
        subset["pred"] = yhat_te
        subset.to_parquet(os.path.join(out_dir, "scores_test.parquet"), index=False)
    except Exception:
        pass

    joblib.dump({"pipeline": pipe, "threshold": float(thr_main), "cfg": asdict(CFG)},
                os.path.join(out_dir, "model.joblib"),
                compress=3)

    _log("\n[OK] Baseline ML concluido.")
    _log(f" Saida: {out_dir}")
    _log(f" Train rows apos downsampling: {metrics['train_rows']}")
    _log(f" Test AUC-ROC: {metrics['test']['roc_auc']}")
    _log(f" Test AP PR: {metrics['test']['pr_ap']}")
    _log(f" Test confusion: tn={metrics['test']['tn']} fp={metrics['test']['fp']} fn={metrics['test']['fn']} tp={metrics['test']['tp']}")
    _log(f" Threshold usado: {metrics['chosen_threshold']:.6f}  | estrategia: {CFG.THRESH_STRATEGY}")
    _log(f" Tempo total: {time.time()-t_all:.2f}s")


if __name__ == "__main__":
    main()
