# Shim: adapta src/train/train_mf.train() ao contrato run_mf_stage2()
import os, json, time, torch
import numpy as np
import pandas as pd

# importa tudo do seu MF2 real
from train.train_mf import (
    HParams, MFModel, WindowsDS, build_windows_index, eval_epoch, train as _train
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS_ROOT   = os.path.join(ROOT, "reports")
OUT_ROOT       = os.path.join(REPORTS_ROOT, "mf_stage2")

def _ts(): return time.strftime("%Y%m%d-%H%M%S", time.localtime())
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _choose_threshold_by_far(labels, scores, target_fpr=1e-4):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    order  = np.argsort(-scores, kind="mergesort")
    y = labels[order]; s = scores[order]
    n0 = int((labels == 0).sum()); n1 = int((labels == 1).sum())
    fp = 0; tp = 0
    best = {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0, "precision": 1.0, "recall": 0.0}
    for i in range(len(s)):
        if y[i] == 1: tp += 1
        else: fp += 1
        fpr = fp / max(n0, 1); tpr = tp / max(n1, 1)
        prec = tp / max(tp + fp, 1); rec = tpr
        if fpr <= target_fpr:
            best = {"threshold": float(s[i]), "fpr": float(fpr), "tpr": float(tpr),
                    "precision": float(prec), "recall": float(rec)}
            break
    neg = scores[labels == 0]
    if neg.size:
        neg_max = float(np.max(neg))
        if best["threshold"] <= neg_max:
            best["threshold"] = float(np.nextafter(neg_max, np.float32(np.inf)))
            best["fpr"] = float((scores[labels == 0] >= best["threshold"]).mean())
    return best

def _confusion_at_thr(labels, scores, thr):
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    pred = (s >= float(thr)).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return tn, fp, fn, tp

@torch.no_grad()
def _predict_scores(model, dl, device):
    model.eval()
    rows = []
    for xb, yb, _, fid, sgps in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.float().cpu().numpy()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
            _, y2, _, _ = model(xb)
            p = torch.sigmoid(y2).detach().float().cpu().numpy()
        for i in range(len(p)):
            rows.append((str(fid[i]), float(sgps[i]), int(yb[i]), float(p[i])))
    df = pd.DataFrame(rows, columns=["file_id","start_gps","label","score"])
    return df

def run_mf_stage2(config_path: str | None = None):
    # 1) treina usando seu MF2 real e pega melhor checkpoint
    sum_train = _train()  # usa defaults do seu HParams
    best_path = sum_train.get("best_ap_path") or sum_train.get("best_auc_path")
    if not best_path or not os.path.exists(best_path):
        raise RuntimeError("MF2 treinou, mas não encontrei best_*_path no resumo.")

    # 2) prepara diretório padrão do pipeline
    out_dir = os.path.join(OUT_ROOT, _ts()); _ensure_dir(out_dir)

    # 3) carrega dataset e index
    hp = HParams()
    df = pd.read_parquet(hp.PARQUET_PATH)
    if "subset" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split":"subset"})
    need = {"file_id","start_gps","label","subset"}
    if not need.issubset(df.columns):
        raise KeyError(f"dataset.parquet sem colunas {need - set(df.columns)}")

    idx_map = build_windows_index(hp.WINDOWS_DIR)

    # 4) DataLoaders val/test para gerar scores
    df_val  = df[df["subset"]=="val"].copy()
    df_test = df[df["subset"]=="test"].copy()
    ds_va = WindowsDS(df_val,  idx_map, hp.FS_TARGET, hp.WINDOW_SEC, teacher_map=None, aug=False)
    ds_te = WindowsDS(df_test, idx_map, hp.FS_TARGET, hp.WINDOW_SEC, teacher_map=None, aug=False)

    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=max(256, hp.BATCH), shuffle=False,
                                        num_workers=hp.NUM_WORKERS, pin_memory=hp.PIN_MEMORY)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=max(256, hp.BATCH), shuffle=False,
                                        num_workers=hp.NUM_WORKERS, pin_memory=hp.PIN_MEMORY)

    # 5) carrega modelo e checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() and hp.DEVICE.startswith("cuda") else "cpu")
    model = MFModel(ch_mf1=hp.CH_MF1, ch_mf2=hp.CH_MF2, dropout=hp.DROPOUT).to(device)
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)

    # 6) gera scores CSVs
    val_csv  = os.path.join(out_dir, "scores_val.csv")
    test_csv = os.path.join(out_dir, "scores_test.csv")
    _predict_scores(model, dl_va, device).to_csv(val_csv, index=False)
    _predict_scores(model, dl_te, device).to_csv(test_csv, index=False)

    # 7) escolhe limiar por FAR alvo e monta summaries
    val_df  = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    info_thr = _choose_threshold_by_far(val_df["label"].values, val_df["score"].values, target_fpr=1e-4)
    thr = float(info_thr["threshold"])
    tn, fp, fn, tp = _confusion_at_thr(test_df["label"].values, test_df["score"].values, thr)
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1); fpr = fp / max(tn+fp, 1)

    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen_threshold": thr, "mode": "constrained_roc", "target_fpr": 1e-4, "info_val": info_thr}, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "cfg": {"MODE": "constrained_roc", "TARGET_FAR": 1e-4},
            "gpu": {"requested": True, "available": torch.cuda.is_available()},
            "val": {},  # métricas detalhadas não são obrigatórias aqui
            "test": {
                "auc": None, "ap": None,
                "confusion_at_thr": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                "precision": float(prec), "recall": float(rec), "fpr": float(fpr)
            },
            "threshold": thr,
            "threshold_info": info_thr
        }, f, indent=2)

    print(f"[MF2] OK. Saída: {out_dir}")
    return out_dir
