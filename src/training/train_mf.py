# Loop de treino MF-PINN 
# src/mf/train_stage2_mf.py
# Multi-Fidelity Stage 2 (PyTorch): LF↔HF consistência + classificação
# Sem argumentos de linha de comando. Tudo configurado abaixo.

import os, json, math, time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Reuso dos utilitários já existentes no seu projeto
from eval.mf_baseline import (
    cached_start_arrays,
    resolve_whitened_path,
    load_window_slice,
    build_windows_index
)

@dataclass
class MFCFG:
    PARQUET_PATH: str = "data/processed/dataset.parquet"
    WINDOWS_DIR: str   = "data/processed"
    OUT_ROOT: str      = "reports/mf_stage2"

    FS_TARGET: int     = 4096
    WINDOW_SEC: float  = 2.0
    N_LF: int          = 3

    # Treino
    EPOCHS: int        = 6
    BATCH: int         = 128
    LR: float          = 2e-3
    WD: float          = 1e-3
    NEG_PER_POS: int   = 50      # downsample de neg para treinar
    NUM_WORKERS: int   = 4
    MIXED_PREC: bool   = True
    GRAD_CLIP: float   = 1.0

    # Perdas
    LAMBDA_CONS: float = 0.2     # logit LF↔HF
    LAMBDA_REP: float  = 0.1     # embedding LF↔HF

    # Avaliação
    SAVE_PREFIX: str   = "mf_stage2"
    MAX_VAL_ROWS: Optional[int] = None   # usa VAL todo
    MAX_TEST_ROWS: int = 60000           # igual ao baseline

    # Determinismo leve
    SEED: int          = 42

CFG = MFCFG()

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PairedMFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, idx_map: Dict[str,str], fs: int, window_sec: float, n_lf: int):
        self.df = df.reset_index(drop=True)
        self.idx_map = idx_map
        self.fs = fs
        self.L = int(round(window_sec*fs))
        self.n_lf = n_lf

    def __len__(self):
        return len(self.df)

    def _load_hf(self, i: int) -> Optional[np.ndarray]:
        r = self.df.iloc[i]
        fid  = r["file_id"]
        sgps = float(r["start_gps"])
        win_path = self.idx_map.get(fid)
        if not win_path:
            return None
        _, _, meta_in = cached_start_arrays(win_path)
        wht = resolve_whitened_path(win_path, meta_in)
        if not wht or not os.path.exists(wht):
            return None
        x = load_window_slice(win_path, wht, sgps)
        if x is None:
            return None
        # x já vem normalizado pelo loader
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _avg_pool_np(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return x
        L = x.shape[-1]
        m = L // k
        x2 = x[:m*k].reshape(m, k).mean(axis=1)
        # volta para o mesmo comprimento via upsample simples
        return np.repeat(x2, k)[:L]

    def _make_lf_stack(self, x: np.ndarray) -> np.ndarray:
        # LF sintético simples: filtros passa-baixa por pooling médio
        # kernels variados para diversidade
        pool_ks = [4, 8, 16, 32, 12, 6]
        out = []
        rng = np.random.default_rng()
        for _ in range(self.n_lf):
            k = int(rng.choice(pool_ks))
            y = self._avg_pool_np(x, k)
            # leve jitter e ganho
            gain = float(rng.uniform(0.8, 1.2))
            noise = rng.normal(0.0, 0.02, size=x.shape).astype(np.float32)
            z = gain*y + noise
            z = (z - z.mean())/(z.std() + 1e-12)
            out.append(z.astype(np.float32, copy=False))
        return np.stack(out, axis=0)  # [n_lf, L]

    def __getitem__(self, i: int):
        x = self._load_hf(i)
        if x is None:
            # devolve zeros para manter robustez do DataLoader
            x = np.zeros(self.L, np.float32)
        x_lf = self._make_lf_stack(x)      # [n_lf, L]
        y = int(self.df.iloc[i]["label"])
        return torch.from_numpy(x_lf), torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class SeparableConv1d(nn.Module):
    def __init__(self, cin, cout, k, s=1, p=None):
        super().__init__()
        if p is None:
            p = k//2
        self.dw = nn.Conv1d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class Encoder1D(nn.Module):
    def __init__(self, in_channels=1, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, 2, 3, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU()
        )
        self.block1 = nn.Sequential(
            SeparableConv1d(width, width, 7, 1),
            SeparableConv1d(width, width*2, 5, 2)
        )
        self.block2 = nn.Sequential(
            SeparableConv1d(width*2, width*2, 5, 1),
            SeparableConv1d(width*2, width*4, 5, 2)
        )
        self.block3 = nn.Sequential(
            SeparableConv1d(width*4, width*4, 3, 1),
            SeparableConv1d(width*4, width*8, 3, 2)
        )
        self.head = nn.AdaptiveAvgPool1d(1)  # global pooling

        self.out_dim = width*8

    def forward(self, x):  # x: [B,1,L]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x).squeeze(-1)  # [B, C]
        return x

class MFNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.enc = Encoder1D(1, width=width)
        self.fc = nn.Linear(self.enc.out_dim, 1)

    def forward(self, x_hf, x_lf):
        # x_hf: [B,1,L]  x_lf: [B,n_lf,1,L]
        B, nlf = x_lf.shape[0], x_lf.shape[1]

        z_hf = self.enc(x_hf)               # [B,C]
        logit_hf = self.fc(z_hf).squeeze(1) # [B]

        z_lfs = []
        logits_lf = []
        for i in range(nlf):
            zi = self.enc(x_lf[:, i, :, :])
            z_lfs.append(zi)
            logits_lf.append(self.fc(zi).squeeze(1))
        z_lf = torch.stack(z_lfs, dim=1).mean(dim=1)         # [B,C]
        logit_lf = torch.stack(logits_lf, dim=1).mean(dim=1) # [B]

        return logit_hf, logit_lf, z_hf, z_lf

def build_train_subset(df_train: pd.DataFrame) -> pd.DataFrame:
    pos = df_train[df_train["label"].astype(int) == 1]
    neg = df_train[df_train["label"].astype(int) == 0]
    if len(pos) == 0:
        return df_train.sample(n=min(len(df_train), 20000), random_state=CFG.SEED)
    n_neg = min(len(neg), CFG.NEG_PER_POS * len(pos))
    neg_ds = neg.sample(n=n_neg, random_state=CFG.SEED) if n_neg > 0 else neg
    mix = pd.concat([pos, neg_ds], axis=0).sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
    return mix

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    return float(auc), float(ap)

def plot_curves(y_true: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, scores):.3f}")
        plt.plot([0,1],[0,1],'k:',alpha=.4)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{prefix}.png"), dpi=140); plt.close()
    except Exception:
        pass
    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(rec, prec, label=f"AP={average_precision_score(y_true, scores):.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_{prefix}.png"), dpi=140); plt.close()
    except Exception:
        pass

def evaluate(model: MFNet, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for x_lf, x_hf, y in loader:
            x_hf = x_hf.unsqueeze(1).to(device, non_blocking=True)         # [B,1,L]
            x_lf = x_lf.unsqueeze(2).to(device, non_blocking=True)         # [B,n_lf,1,L]
            logit_hf, logit_lf, _, _ = model(x_hf, x_lf)
            score = torch.sigmoid(logit_hf).detach().cpu().numpy()
            ss.append(score); ys.append(y.numpy())
    ss = np.concatenate(ss, 0); ys = np.concatenate(ys, 0)
    return ys, ss

def run_mf_stage2():
    set_seed(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(CFG.OUT_ROOT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    # Carrega dataset e splits
    df = pd.read_parquet(CFG.PARQUET_PATH)
    split_col = "subset" if "subset" in df.columns else "split"
    if split_col not in df.columns:
        raise KeyError("dataset.parquet precisa da coluna subset ou split")

    df = df[[split_col, "file_id", "start_gps", "label"]].copy()
    df_train = df[df[split_col] == "train"].copy()
    df_val   = df[df[split_col] == "val"].copy()
    df_test  = df[df[split_col] == "test"].copy()
    if CFG.MAX_VAL_ROWS:
        df_val = df_val.head(CFG.MAX_VAL_ROWS).copy()
    if CFG.MAX_TEST_ROWS:
        df_test = df_test.head(CFG.MAX_TEST_ROWS).copy()

    # Downsample balanceado para treino
    df_train_mix = build_train_subset(df_train)

    # Index dos arquivos de janelas
    idx_map = build_windows_index(CFG.WINDOWS_DIR)
    if not idx_map:
        raise RuntimeError("nenhum *_windows.hdf5 encontrado em WINDOWS_DIR")

    # Datasets
    ds_train = PairedMFDataset(df_train_mix, idx_map, CFG.FS_TARGET, CFG.WINDOW_SEC, CFG.N_LF)
    ds_val   = PairedMFDataset(df_val,       idx_map, CFG.FS_TARGET, CFG.WINDOW_SEC, CFG.N_LF)
    ds_test  = PairedMFDataset(df_test,      idx_map, CFG.FS_TARGET, CFG.WINDOW_SEC, CFG.N_LF)

    loader_train = DataLoader(ds_train, batch_size=CFG.BATCH, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    loader_val   = DataLoader(ds_val, batch_size=CFG.BATCH, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    loader_test  = DataLoader(ds_test, batch_size=CFG.BATCH, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Modelo
    model = MFNet(width=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.MIXED_PREC)

    # pos_weight para BCE por desbalanceamento do batch
    # estimativa por df_train_mix
    n_pos = int((df_train_mix["label"] == 1).sum())
    n_neg = int((df_train_mix["label"] == 0).sum())
    pos_weight = torch.tensor([(n_neg + 1) / max(n_pos, 1)], device=device, dtype=torch.float32)

    history = []
    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        t0 = time.time()
        loss_meter = 0.0
        it = 0
        for x_lf, x_hf, y in loader_train:
            x_hf = x_hf.unsqueeze(1).to(device, non_blocking=True)   # [B,1,L]
            x_lf = x_lf.unsqueeze(2).to(device, non_blocking=True)   # [B,n_lf,1,L]
            y = y.to(device, non_blocking=True).float()              # [B]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG.MIXED_PREC):
                logit_hf, logit_lf, z_hf, z_lf = model(x_hf, x_lf)
                # Losses
                Ldata = F.binary_cross_entropy_with_logits(logit_hf, y, pos_weight=pos_weight)
                Lcons = F.mse_loss(logit_lf, logit_hf)
                Lrep  = F.mse_loss(z_lf, z_hf)
                loss = Ldata + CFG.LAMBDA_CONS*Lcons + CFG.LAMBDA_REP*Lrep

            scaler.scale(loss).backward()
            if CFG.GRAD_CLIP and CFG.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            loss_meter += float(loss.detach().cpu().item())
            it += 1

        sched.step()
        dt = time.time() - t0

        # Avaliações
        yv, sv = evaluate(model, loader_val, device)
        yt, st = evaluate(model, loader_test, device)

        auc_val, ap_val = compute_metrics(yv, sv)
        auc_test, ap_test = compute_metrics(yt, st)

        history.append({
            "epoch": epoch,
            "time_s": dt,
            "train_loss": loss_meter / max(it, 1),
            "val_auc": auc_val, "val_ap": ap_val,
            "test_auc": auc_test, "test_ap": ap_test,
            "lr": float(sched.get_last_lr()[0])
        })

        print(f"[MF2][{epoch}/{CFG.EPOCHS}] loss={history[-1]['train_loss']:.4f} "
              f"val AUC={auc_val:.4f} AP={ap_val:.4f} | test AUC={auc_test:.4f} AP={ap_test:.4f} "
              f"| {dt:.1f}s")

    # Salva saídas
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Curvas finais
    yv, sv = evaluate(model, loader_val, device)
    yt, st = evaluate(model, loader_test, device)
    plot_curves(yv, sv, out_dir, f"{CFG.SAVE_PREFIX}_val")
    plot_curves(yt, st, out_dir, f"{CFG.SAVE_PREFIX}_test")

    # CSVs de scores
    df_val_out = ds_val.df.copy();   df_val_out["score"]  = sv
    df_test_out = ds_test.df.copy(); df_test_out["score"] = st
    df_val_out.to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)
    df_test_out.to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)

    # Summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "cfg": asdict(CFG),
            "val": {"auc": float(roc_auc_score(yv.astype(int), sv)) if len(np.unique(yv))>1 else None,
                    "ap": float(average_precision_score(yv.astype(int), sv)) if len(np.unique(yv))>1 else None},
            "test": {"auc": float(roc_auc_score(yt.astype(int), st)) if len(np.unique(yt))>1 else None,
                     "ap": float(average_precision_score(yt.astype(int), st)) if len(np.unique(yt))>1 else None},
            "out_dir": out_dir
        }, f, indent=2)

    print(f"[OK] MF Stage 2 concluído. Saída: {out_dir}")

# Permite importar e chamar do run.py
if __name__ == "__main__":
    run_mf_stage2()