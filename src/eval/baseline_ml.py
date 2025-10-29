# src/eval/baseline_ml.py
# Baseline ML para detecção binária em janelas GW
# - Lê data/processed/dataset.parquet
# - Se existir data/processed/dataset_with_split.parquet usa o split salvo
# - Caso contrário, cria split por grupo (file_id) com GroupShuffleSplit
# - Downsample no treino para reduzir desbalanceamento
# - Pipeline: imputação + padronização + XGBoost (CPU por padrão, estável no pickle)
# - Estratégias de threshold: best_f1 | target_far | target_recall
# - Salva modelo e métricas em reports/baseline_ml/YYYYMMDD-HHMMSS

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost é opcional, mas esperado pelo pipeline
from xgboost import XGBClassifier


# ============================== Config ==============================

@dataclass
class Cfg:
    SEED: int = 42
    DATA_DIR: str = "data/processed"
    PARQUET_NAME: str = "dataset.parquet"
    PARQUET_SPLIT_NAME: str = "dataset_with_split.parquet"
    OUT_DIR: str = "reports/baseline_ml"

    # Frações de split por grupo
    TEST_SIZE: float = 0.18
    VAL_SIZE: float = 0.18  # do remanescente após remover test

    # Downsample no treino
    DOWNSAMPLE_POS_MAX: int = 200     # máximo de positivos no treino
    DOWNSAMPLE_NEG_PER_POS: int = 50  # razão de negativos por positivo

    # Estratégia de limiar: "best_f1" | "target_far" | "target_recall"
    THRESH_STRATEGY: str = "best_f1"
    TARGET_FAR: float = 1e-4           # FPR alvo para target_far
    RECALL_TARGET_REPORT: float = 0.30 # Recall alvo para target_recall

    # XGBoost estável para CPU
    XGB_PARAMS: dict = None

    def __post_init__(self):
        if self.XGB_PARAMS is None:
            self.XGB_PARAMS = dict(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                # Força CPU estável e evita avisos de predictor
                predictor="cpu_predictor",
                random_state=self.SEED,
                n_jobs=0,
            )

CFG = Cfg()


# ============================== Utilidades ==============================

def _load_dataset() -> pd.DataFrame:
    base = Path(CFG.DATA_DIR)
    p = base / CFG.PARQUET_NAME
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    df = pd.read_parquet(p)
    # Normaliza nomes usuais
    if "label" in df.columns and "y" not in df.columns:
        df = df.rename(columns={"label": "y"})
    if "target" in df.columns and "y" not in df.columns:
        df = df.rename(columns={"target": "y"})
    if "y" not in df.columns:
        raise ValueError("Coluna de rótulo 'y' ausente")
    # Assegura split se existir previamente
    ps = base / CFG.PARQUET_SPLIT_NAME
    if ps.exists():
        df_split = pd.read_parquet(ps)
        # Confere chave de junção
        # Espera-se que df_split tenha mesmas linhas e índice compatível
        if "split" in df_split.columns and len(df_split) == len(df):
            df["split"] = df_split["split"].values
    return df


def _create_group_split(df: pd.DataFrame) -> pd.DataFrame:
    """Cria split por grupo usando file_id, salva sidecar parquet."""
    if "file_id" not in df.columns:
        # fallback: todo mundo no mesmo grupo pelo índice inteiro
        groups = df.index.values
    else:
        groups = df["file_id"].values

    rs = np.random.RandomState(CFG.SEED)

    # Primeiro, separa test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=CFG.TEST_SIZE, random_state=CFG.SEED)
    idx_trainval, idx_test = next(gss1.split(df, groups=groups))
    mask = np.full(len(df), fill_value="train", dtype=object)
    mask[idx_test] = "test"

    # Depois, separa val a partir do restante
    df_trainval = df.iloc[idx_trainval]
    groups_tv = groups[idx_trainval]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=CFG.VAL_SIZE, random_state=CFG.SEED + 1)
    idx_train, idx_val = next(gss2.split(df_trainval, groups=groups_tv))
    mask[idx_trainval[idx_val]] = "val"
    mask[idx_trainval[idx_train]] = "train"

    df = df.copy()
    df["split"] = mask

    # Salva o sidecar com split fixado
    out = Path(CFG.DATA_DIR) / CFG.PARQUET_SPLIT_NAME
    df[["split"]].to_parquet(out, index=False)
    return df


def _ensure_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" in df.columns:
        return df
    return _create_group_split(df)


def _select_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Seleciona colunas numéricas e ignora as que não são features."""
    drop_like = {
        "y", "label", "split", "file_id", "detector", "event_id", "tc", "t0", "t1",
        "window_idx", "window_id", "path", "subset"
    }
    cols_feat = [c for c in df.columns if c not in drop_like]
    # Numéricas apenas, sem lambdas, sem FunctionTransformer
    num_cols = [c for c in cols_feat if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols: List[str] = []  # baseline só com numéricas
    return num_cols, cat_cols


def _downsample_train(df_train: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    pos = df_train[df_train["y"] == 1]
    neg = df_train[df_train["y"] == 0]

    if len(pos) == 0:
        # Nada a fazer, mantém como está
        return df_train

    # Limita positivos
    pos_sample = pos
    if len(pos) > CFG.DOWNSAMPLE_POS_MAX:
        pos_idx = rng.choice(pos.index.values, size=CFG.DOWNSAMPLE_POS_MAX, replace=False)
        pos_sample = pos.loc[pos_idx]

    # Seleciona negativos proporcionais
    n_neg = min(len(neg), len(pos_sample) * CFG.DOWNSAMPLE_NEG_PER_POS)
    neg_idx = rng.choice(neg.index.values, size=n_neg, replace=False)
    neg_sample = neg.loc[neg_idx]

    out = pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1.0, random_state=CFG.SEED)
    return out


# ============================== Thresholds ==============================

def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pr, rc, thr = metrics.precision_recall_curve(y_true, y_score)
    # thr tem tamanho len(rc) - 1
    f1 = (2 * pr * rc) / np.clip(pr + rc, 1e-12, None)
    k = int(np.nanargmax(f1))
    # alinhar índice com thr
    if 0 < k <= len(thr) - 1:
        return float(thr[k])
    if len(thr) > 0:
        return float(np.median(thr))
    return float(np.median(y_score))


def _target_far_threshold(y_true: np.ndarray, y_score: np.ndarray, far_target: float) -> float:
    fpr, tpr, thr = metrics.roc_curve(y_true, y_score)
    # penaliza tpr == 0 para evitar degenerate
    score = np.abs(fpr - far_target) + (tpr == 0) * 1e3
    k = int(np.argmin(score))
    return float(thr[k])


def _target_recall_threshold(y_true: np.ndarray, y_score: np.ndarray, recall_target: float) -> float:
    pr, rc, thr = metrics.precision_recall_curve(y_true, y_score)
    idx = np.where(rc[:-1] >= recall_target)[0]
    if len(idx) == 0:
        # fallback conservador
        return float(np.percentile(y_score, 99.9))
    k = int(idx[-1])
    return float(thr[k])


def _select_threshold(strategy: str, y_val: np.ndarray, p_val: np.ndarray) -> float:
    s = strategy.lower().strip()
    if s == "best_f1":
        return _best_f1_threshold(y_val, p_val)
    if s == "target_far":
        return _target_far_threshold(y_val, p_val, CFG.TARGET_FAR)
    if s == "target_recall":
        return _target_recall_threshold(y_val, p_val, CFG.RECALL_TARGET_REPORT)
    raise ValueError(f"Estratégia de limiar inválida: {strategy}")


def _report_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    yhat = (y_score >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    prec = metrics.precision_score(y_true, yhat, zero_division=0)
    rec = metrics.recall_score(y_true, yhat, zero_division=0)
    fpr = fp / max(fp + tn, 1)
    return dict(
        threshold=float(thr), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        precision=float(prec), recall=float(rec), fpr=float(fpr)
    )


# ============================== Treino/Avaliação ==============================

def _build_pipeline(num_cols: List[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols)],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=None,
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(**CFG.XGB_PARAMS)
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", model),
    ])
    return pipe


def _split_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        # Se algo veio vazio por desequilíbrio de grupos, refaz split
        df2 = _create_group_split(df.drop(columns=["split"]))
        tr = df2[df2["split"] == "train"]
        va = df2[df2["split"] == "val"]
        te = df2[df2["split"] == "test"]
    return tr, va, te


def _prepare_xy(df: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[num_cols].astype(np.float32)
    y = df["y"].astype(int).values
    return X, y


def main():
    t0 = time.time()
    rng = np.random.RandomState(CFG.SEED)

    # 1) Carregar dataset e garantir split
    print("[1/6] Lendo dataset...")
    df = _load_dataset()
    if "split" not in df.columns:
        df = _ensure_split(df)

    # 2) Selecionar colunas numéricas
    num_cols, _ = _select_columns(df)
    if len(num_cols) == 0:
        raise ValueError("Nenhuma coluna numérica disponível para features")

    # 3) Separar splits
    tr, va, te = _split_sets(df)

    # 4) Downsample no treino
    print("[2/6] Downsampling do treino...")
    tr_ds = _downsample_train(tr, rng)
    print(f"   • treino (após downsample): {len(tr_ds):,}  (pos={int((tr_ds['y']==1).sum()):,}, neg={int((tr_ds['y']==0).sum()):,})")

    # 5) Preparar X/y
    print("[3/6] Preparando X/y...")
    X_train, y_train = _prepare_xy(tr_ds, num_cols)
    X_val, y_val = _prepare_xy(va, num_cols)
    X_test, y_test = _prepare_xy(te, num_cols)

    # 6) Treinar
    print("[4/6] Treinando modelo...")
    pipe = _build_pipeline(num_cols)
    pipe.fit(X_train, y_train)
    train_time = time.time()

    # 7) Avaliar e escolher limiar usando VAL
    print("[5/6] Avaliando e escolhendo limiar (val)...")
    p_val = pipe.predict_proba(X_val)[:, 1]
    auc_val = metrics.roc_auc_score(y_val, p_val)
    ap_val = metrics.average_precision_score(y_val, p_val)
    thr_main = _select_threshold(CFG.THRESH_STRATEGY, y_val, p_val)
    rep_val = _report_at_threshold(y_val, p_val, thr_main)

    # 8) Avaliar no TEST
    print("[6/6] Avaliando no TEST...")
    p_test = pipe.predict_proba(X_test)[:, 1]
    auc_test = metrics.roc_auc_score(y_test, p_test)
    ap_test = metrics.average_precision_score(y_test, p_test)
    rep_test = _report_at_threshold(y_test, p_test, thr_main)

    # 9) Saída e logs
    tag = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(CFG.OUT_DIR) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    # 9.a) Escrever scores para o viz
    # Colunas opcionais para alinhar com o pipeline de viz
    def _maybe_cols(df, cols):
        have = [c for c in cols if c in df.columns]
        return df[have].reset_index(drop=True) if have else pd.DataFrame()

    val_meta = _maybe_cols(va, ["file_id", "start_gps"])
    tst_meta = _maybe_cols(te, ["file_id", "start_gps"])

    df_val_scores = pd.concat(
        [val_meta,
         pd.DataFrame({"label": y_val.astype(int), "score": p_val.astype(float)})],
        axis=1
    )
    df_tst_scores = pd.concat(
        [tst_meta,
         pd.DataFrame({"label": y_test.astype(int), "score": p_test.astype(float)})],
        axis=1
    )

    # formatos que o teu viz costuma aceitar
    df_val_scores.to_csv(outdir / "scores_val.csv", index=False)
    df_tst_scores.to_csv(outdir / "scores_test.csv", index=False)
    df_val_scores.to_parquet(outdir / "scores_val.parquet", index=False)
    df_tst_scores.to_parquet(outdir / "scores_test.parquet", index=False)


    # 10) Persistir artefatos
    joblib.dump(
        {
            "pipeline": pipe,
            "threshold": float(thr_main),
            "cfg": asdict(CFG),
            "columns": {"numeric": num_cols},
            "metrics": {
                "val": {"auc": float(auc_val), "ap": float(ap_val), **rep_val},
                "test": {"auc": float(auc_test), "ap": float(ap_test), **rep_test},
            },
        },
        outdir / "model.joblib"
    )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(
            {
                "val": {"auc": float(auc_val), "ap": float(ap_val), **rep_val},
                "test": {"auc": float(auc_test), "ap": float(ap_test), **rep_test},
            },
            f, indent=2
        )

    # Sidecar simples com tags
    with open(outdir / "run_info.json", "w") as f:
        json.dump(
            {
                "tag": tag,
                "train_rows_after_downsample": int(len(X_train)),
                "train_time_s": float(train_time - t0),
                "total_time_s": float(time.time() - t0),
            },
            f, indent=2
        )


if __name__ == "__main__":
    main()
