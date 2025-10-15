# src/training/train_mf.py
# MF Stage 2 turbo: LF on-device, prefetch em stream, TF32/bf16, torch.compile e passo por época opcional

import os, json, time, collections
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

from eval.mf_baseline import (
    cached_start_arrays,
    resolve_whitened_path,
    load_window_slice,
    build_windows_index
)

# Backends rápidos
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class MFCFG:
    PARQUET_PATH: str = "data/processed/dataset.parquet"
    WINDOWS_DIR: str   = "data/processed"
    OUT_ROOT: str      = "reports/mf_stage2"

    FS_TARGET: int    = 4096
    WINDOW_SEC: float = 2.0

    # Treino
    EPOCHS: int      = 6
    BATCH: int       = 128
    LR: float        = 2e-3
    WD: float        = 1e-3
    NEG_PER_POS: int = 50
    NUM_WORKERS: int = 0
    MIXED_PREC: bool = True
    GRAD_CLIP: float = 1.0

    # Multi-fidelity
    N_LF: int = 3
    LF_KERNELS: Tuple[int, ...] = (4, 6, 8, 12, 16, 32)
    LF_NOISE_STD: float = 0.02
    LF_ON_DEVICE: bool = True

    # Avaliação e limites
    MAX_TRAIN_ROWS: Optional[int] = None
    MAX_VAL_ROWS: Optional[int]   = None
    MAX_TEST_ROWS: Optional[int]  = 60000

    SAVE_PREFIX: str = "mf_stage2"
    SEED: int = 42

    # Cache e rede
    DATASET_CACHE_SIZE: int = 256
    NET_WIDTH: int = 64

    # Otimizações extras
    USE_PREFETCHER: bool = True
    MAX_STEPS_PER_EPOCH: Optional[int] = None  # ex: 200 para limitar passos

CFG = MFCFG()

# FAST preset para pipeline rápido
if os.getenv("FAST_PIPELINE", "0") == "1":
    CFG.EPOCHS = 1
    CFG.BATCH = 64
    CFG.NEG_PER_POS = 10
    CFG.N_LF = 2
    CFG.MAX_TRAIN_ROWS = 2048
    CFG.MAX_VAL_ROWS = 200
    CFG.MAX_TEST_ROWS = 200
    CFG.NET_WIDTH = 32
    CFG.MAX_STEPS_PER_EPOCH = 200

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PairedHFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, idx_map: Dict[str,str], fs: int, window_sec: float, cache_size: int = 256):
        self.df = df.reset_index(drop=True)
        self.idx_map = idx_map
        self.fs = fs
        self.L = int(round(window_sec*fs))
        self.cache = collections.OrderedDict()
        self.cache_size = max(8, int(cache_size))

    def __len__(self): return len(self.df)

    def _cache_put(self, key, val):
        self.cache[key] = val
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _cache_get(self, key):
        if key in self.cache:
            v = self.cache[key]
            self.cache.move_to_end(key)
            return v
        return None

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        fid  = r["file_id"]
        sgps = float(r["start_gps"])
        key = (fid, sgps)

        x = self._cache_get(key)
        if x is None:
            win_path = self.idx_map.get(fid)
            if not win_path:
                x = np.zeros(self.L, np.float32)
            else:
                _, _, meta_in = cached_start_arrays(win_path)
                wht = resolve_whitened_path(win_path, meta_in)
                if not wht or not os.path.exists(wht):
                    x = np.zeros(self.L, np.float32)
                else:
                    x2 = load_window_slice(win_path, wht, sgps)
                    x = x2.astype(np.float32, copy=False) if x2 is not None else np.zeros(self.L, np.float32)
            self._cache_put(key, x)
        y = int(r["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class SeparableConv1d(nn.Module):
    def __init__(self, cin, cout, k, s=1, p=None):
        super().__init__()
        if p is None: p = k//2
        self.dw = nn.Conv1d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm1d(cout); self.act = nn.SiLU()
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        return self.act(x)

class Encoder1D(nn.Module):
    def __init__(self, in_channels=1, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, width, 7, 2, 3, bias=False),
                                  nn.BatchNorm1d(width), nn.SiLU())
        self.block1 = nn.Sequential(SeparableConv1d(width, width, 7, 1),
                                    SeparableConv1d(width, width*2, 5, 2))
        self.block2 = nn.Sequential(SeparableConv1d(width*2, width*2, 5, 1),
                                    SeparableConv1d(width*2, width*4, 5, 2))
        self.block3 = nn.Sequential(SeparableConv1d(width*4, width*4, 3, 1),
                                    SeparableConv1d(width*4, width*8, 3, 2))
        self.head = nn.AdaptiveAvgPool1d(1)
        self.out_dim = width*8
    def forward(self, x):
        x = self.stem(x); x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.head(x).squeeze(-1)

class MFNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.enc = Encoder1D(1, width=width)
        self.fc = nn.Linear(self.enc.out_dim, 1)
    def forward(self, x_hf, x_lf):
        B, nlf = x_lf.shape[0], x_lf.shape[1]
        z_hf = self.enc(x_hf); logit_hf = self.fc(z_hf).squeeze(1)
        z_lfs, logits_lf = [], []
        for i in range(nlf):
            zi = self.enc(x_lf[:, i, :, :]); z_lfs.append(zi)
            logits_lf.append(self.fc(zi).squeeze(1))
        z_lf = torch.stack(z_lfs, dim=1).mean(dim=1)
        logit_lf = torch.stack(logits_lf, dim=1).mean(dim=1)
        return logit_hf, logit_lf, z_hf, z_lf

def make_lf_stack_torch(x_hf: torch.Tensor, n_lf: int, kernels: Tuple[int, ...], add_noise: bool,
                        noise_std: float) -> torch.Tensor:
    B, _, L = x_hf.shape; device = x_hf.device
    ks = list(kernels)
    if n_lf <= len(ks): sel = ks[:n_lf]
    else:
        sel = ks[:]
        while len(sel) < n_lf: sel += ks[:n_lf-len(sel)]
    outs = []
    for k in sel:
        y = F.avg_pool1d(x_hf, kernel_size=k, stride=k)
        y = y.repeat_interleave(k, dim=-1)
        if y.shape[-1] > L: y = y[..., :L]
        elif y.shape[-1] < L: y = F.pad(y, (L - y.shape[-1], 0))
        if add_noise and noise_std > 0: y = y + noise_std * torch.randn_like(y)
        y = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-12)
        outs.append(y)
    return torch.stack(outs, dim=0).permute(1,0,2,3).contiguous()

# Prefetcher para overlap H2D
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader; self.device = device; self.stream = torch.cuda.Stream(device=device)
    def __iter__(self):
        it = iter(self.loader)
        next_batch = None
        def preload():
            nonlocal next_batch
            try:
                batch = next(it)
            except StopIteration:
                next_batch = None; return
            with torch.cuda.stream(self.stream):
                x, y = batch
                x = x.unsqueeze(1).to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                next_batch = (x, y)
        preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = next_batch
            preload()
            yield batch

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2: return None, None
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))

def plot_curves(y_true: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str):
    try:
        y_true = y_true.astype(int)
        if len(np.unique(y_true)) < 2: return
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, scores):.3f}")
        plt.plot([0,1],[0,1],'k:',alpha=.4)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{prefix}.png"), dpi=140); plt.close()

        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(4.8,4.6))
        plt.plot(rec, prec, label=f"AP={average_precision_score(y_true, scores):.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_{prefix}.png"), dpi=140); plt.close()
    except Exception:
        pass

@torch.no_grad()
def evaluate(model: MFNet, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    for x_cpu, y in loader:
        x_hf = x_cpu.unsqueeze(1).to(device, non_blocking=True)
        x_lf = make_lf_stack_torch(x_hf, CFG.N_LF, CFG.LF_KERNELS, add_noise=False, noise_std=0.0)
        logit_hf, _, _, _ = model(x_hf, x_lf)
        ss.append(torch.sigmoid(logit_hf).detach().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys, 0), np.concatenate(ss, 0)

def run_mf_stage2():
    set_seed(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(CFG.OUT_ROOT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("============= MF Stage 2 =============")
    print(f"Device: {device.type}")
    print(f"Mixed precision: {CFG.MIXED_PREC}  AMP dtype: {'bf16' if amp_dtype==torch.bfloat16 else 'fp16'}")
    print(f"LF kernels: {CFG.LF_KERNELS}")
    print("======================================")

    df = pd.read_parquet(CFG.PARQUET_PATH)
    split_col = "subset" if "subset" in df.columns else "split"
    if split_col not in df.columns: raise KeyError("dataset.parquet precisa da coluna subset ou split")

    df = df[[split_col, "file_id", "start_gps", "label"]].copy()
    df_train = df[df[split_col]=="train"].copy()
    df_val   = df[df[split_col]=="val"].copy()
    df_test  = df[df[split_col]=="test"].copy()

    if CFG.MAX_TRAIN_ROWS: df_train = df_train.head(CFG.MAX_TRAIN_ROWS).copy()
    if CFG.MAX_VAL_ROWS:   df_val   = df_val.head(CFG.MAX_VAL_ROWS).copy()
    if CFG.MAX_TEST_ROWS:  df_test  = df_test.head(CFG.MAX_TEST_ROWS).copy()

    pos = df_train[df_train["label"].astype(int)==1]
    neg = df_train[df_train["label"].astype(int)==0]
    if len(pos) > 0:
        n_neg = min(len(neg), CFG.NEG_PER_POS * len(pos))
        neg = neg.sample(n=n_neg, random_state=CFG.SEED) if n_neg > 0 else neg.head(0)
        df_train = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)

    idx_map = build_windows_index(CFG.WINDOWS_DIR)
    if not idx_map: raise RuntimeError("nenhum *_windows.hdf5 encontrado em WINDOWS_DIR")

    ds_train = PairedHFDataset(df_train, idx_map, CFG.FS_TARGET, CFG.WINDOW_SEC, cache_size=CFG.DATASET_CACHE_SIZE)
    ds_val   = PairedHFDataset(df_val,   idx_map, CFG.FS_TARGET, CFG.WINDOW_SEC, cache_size=CFG.DATASET_CACHE_SIZE)
    ds_test  = PairedHFDataset(df_test,  idx_map, CFG.FS_TARGET, CFG.WINDOW_SEC, cache_size=CFG.DATASET_CACHE_SIZE)

    loader_train = DataLoader(ds_train, batch_size=CFG.BATCH, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
                              prefetch_factor=4 if CFG.NUM_WORKERS>0 else None,
                              persistent_workers=(CFG.NUM_WORKERS>0))
    loader_val   = DataLoader(ds_val, batch_size=CFG.BATCH, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              prefetch_factor=4 if CFG.NUM_WORKERS>0 else None,
                              persistent_workers=(CFG.NUM_WORKERS>0))
    loader_test  = DataLoader(ds_test, batch_size=CFG.BATCH, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              prefetch_factor=4 if CFG.NUM_WORKERS>0 else None,
                              persistent_workers=(CFG.NUM_WORKERS>0))

    model = MFNet(width=CFG.NET_WIDTH).to(device)

    # torch.compile para acelerar convs
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    except Exception:
        pass

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=(CFG.MIXED_PREC and device.type=="cuda"))

    n_pos = int((df_train["label"]==1).sum())
    n_neg = int((df_train["label"]==0).sum())
    pos_weight = torch.tensor([(n_neg + 1)/max(n_pos,1)], device=device, dtype=torch.float32)

    # Prefetcher
    prefetch = CUDAPrefetcher(loader_train, device) if (device.type=="cuda" and CFG.USE_PREFETCHER) else None

    history = []
    for epoch in range(1, CFG.EPOCHS+1):
        model.train(); t0 = time.time(); loss_meter = 0.0; it = 0
        iterator = prefetch if prefetch is not None else loader_train
        for batch_idx, batch in enumerate(iterator):
            if prefetch is not None: x_hf, y = batch
            else:
                x_cpu, y = batch
                x_hf = x_cpu.unsqueeze(1).to(device, non_blocking=True)

            x_lf = make_lf_stack_torch(x_hf, CFG.N_LF, CFG.LF_KERNELS, add_noise=True, noise_std=CFG.LF_NOISE_STD)
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(CFG.MIXED_PREC and device.type=="cuda")):
                logit_hf, logit_lf, z_hf, z_lf = model(x_hf, x_lf)
                Ldata = F.binary_cross_entropy_with_logits(logit_hf, y, pos_weight=pos_weight)
                Lcons = F.mse_loss(logit_lf, logit_hf)
                Lrep  = F.mse_loss(z_lf, z_hf)
                loss = Ldata + 0.2*Lcons + 0.1*Lrep

            scaler.scale(loss).backward()
            if CFG.GRAD_CLIP and CFG.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt); scaler.update()

            loss_meter += float(loss.detach().cpu().item()); it += 1

            if CFG.MAX_STEPS_PER_EPOCH and it >= CFG.MAX_STEPS_PER_EPOCH:
                break

        sched.step(); dt = time.time() - t0

        yv, sv = evaluate(model, loader_val, device)
        yt, st = evaluate(model, loader_test, device)
        auc_val, ap_val = compute_metrics(yv, sv)
        auc_test, ap_test = compute_metrics(yt, st)

        print(f"[MF2][{epoch}/{CFG.EPOCHS}] loss={(loss_meter/max(it,1)):.4f} "
              f"val AUC={auc_val if auc_val is not None else 'NA'} AP={ap_val if ap_val is not None else 'NA'} | "
              f"test AUC={auc_test if auc_test is not None else 'NA'} AP={ap_test if ap_test is not None else 'NA'} "
              f"| {dt:.1f}s")

        history.append({
            "epoch": epoch, "time_s": dt,
            "train_loss": loss_meter / max(it, 1),
            "val_auc": auc_val, "val_ap": ap_val,
            "test_auc": auc_test, "test_ap": ap_test,
            "lr": float(sched.get_last_lr()[0])
        })

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    yv, sv = evaluate(model, loader_val, device)
    yt, st = evaluate(model, loader_test, device)
    plot_curves(yv, sv, out_dir, f"{CFG.SAVE_PREFIX}_val")
    plot_curves(yt, st, out_dir, f"{CFG.SAVE_PREFIX}_test")

    df_val_out = ds_val.df.copy();   df_val_out["score"]  = sv
    df_test_out = ds_test.df.copy(); df_test_out["score"] = st
    df_val_out.to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)
    df_test_out.to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)

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

if __name__ == "__main__":
    run_mf_stage2()
