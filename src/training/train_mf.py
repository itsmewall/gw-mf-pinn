# src/training/train_mf.py
# ======================================================================================
# Treinador Multi-Fidelity (MF1 + MF2) para detecção binária em janelas GW
# Otimizado para throughput: TF32, AMP fp16, AdamW fused, torch.compile, DataLoader agressivo,
# cache LRU de janelas HDF5, avaliação intercalada e autotune de batch via variável de ambiente.
# Interface sem argumentos. Use as funções:
#   - train()                  -> treina e salva checkpoints
#   - run_mf_stage2()          -> treina e exporta scores/thresholds/summary no padrão do pipeline
#   - export_latest()          -> só exporta usando o melhor checkpoint do último train
# Execução direta do módulo chama run_mf_stage2().
# Opcional: exporte MF_AUTOBATCH=1 para ativar autotune do maior batch que cabe na VRAM.
# ======================================================================================

from __future__ import annotations
import os, time, json, random, warnings, math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import h5py
from fractions import Fraction
from scipy.signal import resample_poly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint_sequential

from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# =========================
# ETA helpers
# =========================
def _fmt_hms(seconds: float) -> str:
    s = int(max(0, seconds)); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class ETAMeter:
    def __init__(self, alpha: float = 0.12):
        self.alpha = float(alpha); self.ema = None; self.last = None; self.steps = 0
    def start_step(self): self.last = time.time()
    def end_step(self):
        if self.last is None: return
        dt = time.time() - self.last; self.steps += 1
        self.ema = dt if self.ema is None else self.alpha*dt + (1.0-self.alpha)*self.ema
    def eta(self, remaining_steps: int) -> float:
        if self.ema is None: return float("inf")
        return max(0.0, self.ema * max(0, int(remaining_steps)))

# =========================
# Defaults
# =========================
@dataclass
class HParams:
    # dados
    PARQUET_PATH: str = os.path.join("data", "processed", "dataset.parquet")
    WINDOWS_DIR: str  = os.path.join("data", "processed")
    FS_TARGET: int    = 4096
    WINDOW_SEC: float = 2.0
    SUBSETS: Tuple[str, ...] = ("train","val","test")

    # treino
    EPOCHS: int       = 12
    BATCH: int        = 128
    LR: float         = 2e-4
    WD: float         = 1e-4
    OPT: str          = "adamw"
    AMP: bool         = True
    GRAD_CLIP: float  = 1.0
    NUM_WORKERS: int  = 8
    PIN_MEMORY: bool  = True
    PREFETCH: int     = 8
    PERSISTENT: bool  = True
    SEED: int         = 2025
    DEVICE: str       = "cuda"
    CHECKPOINTING: bool = False
    COMPILE: bool       = True
    EVAL_EVERY: int     = 3          # reduz overhead de validação

    # perdas
    W_BCE: float      = 1.0
    W_CONSIST: float  = 0.2
    W_DISTIL: float   = 0.3

    # distil opcional
    TEACHER_CSV_VAL: Optional[str] = None
    TEACHER_CSV_TST: Optional[str] = None
    TEACHER_COL: str = "score"

    # augment
    AUG_TIMESHIFT_SEC: float = 0.010
    AUG_GAUSS_STD: float    = 0.02
    TIME_DROPOUT_P: float   = 0.03

    # modelo
    CH_MF1: int       = 32
    CH_MF2: int       = 48
    DROPOUT: float    = 0.05

    # saída
    OUT_DIR: str      = os.path.join("runs", "mf")
    SAVE_EVERY: int   = 1

# =========================
# Utilidades de dados
# =========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        import torch.backends.cuda as tbc, torch.backends.cudnn as tbn
        tbc.matmul.allow_tf32 = True; tbn.allow_tf32 = True

def build_windows_index(windows_dir: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for root, _, files in os.walk(windows_dir):
        for f in files:
            if f.endswith("_windows.hdf5"):
                idx[f] = os.path.abspath(os.path.join(root, f))
    if not idx: raise FileNotFoundError("Nenhum *_windows.hdf5 encontrado")
    return idx

def _read_attrs(obj) -> Dict[str, float | str]:
    out = {}
    try:
        for k, v in obj.attrs.items():
            if hasattr(v, "item"):
                try: v = v.item()
                except Exception: pass
            out[str(k)] = v
    except Exception: pass
    return out

_cached_sgps: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str,str]]] = {}

def _cached_start_arrays(win_path: str):
    if win_path in _cached_sgps: return _cached_sgps[win_path]
    with h5py.File(win_path, "r") as f:
        sgps = f["/windows/start_gps"][()].astype(float)
        sidx = f["/windows/start_idx"][()].astype(int)
        meta = _read_attrs(f["/meta_in"]) if "/meta_in" in f else {}
    _cached_sgps[win_path] = (sgps, sidx, meta); return _cached_sgps[win_path]

def _resolve_whitened_path(win_path: str, meta_in: Dict[str,str]) -> Optional[str]:
    src = meta_in.get("source_path") or meta_in.get("source_file") or meta_in.get("source")
    candidates = []
    if src and os.path.isabs(src): candidates.append(src)
    if src:
        candidates += [
            os.path.join(os.path.dirname(win_path), src),
            os.path.join("data","interim", os.path.basename(src)),
            os.path.join("data","processed", os.path.basename(src)),
        ]
    bn = os.path.basename(win_path).replace("_windows.hdf5", "_whitened.hdf5")
    candidates += [
        os.path.join(os.path.dirname(win_path), bn),
        os.path.join("data","interim", bn),
        os.path.join("data","processed", bn),
    ]
    for c in candidates:
        if c and os.path.exists(c): return os.path.abspath(c)
    return None

def _load_window_slice(win_path: str, wht_path: str, start_gps: float, fs_tgt: int, win_sec: float) -> Optional[np.ndarray]:
    try:
        sgps, sidx, _ = _cached_start_arrays(win_path)
        idxs = np.where(np.isclose(sgps, float(start_gps), atol=5e-4))[0]
        if idxs.size == 0: return None
        i0 = int(sidx[int(idxs[0])])

        with h5py.File(wht_path, "r") as g:
            cand = ["/strain/StrainWhitened","/strain/whitened","/data/whitened","/whitened","/strain/Strain"]
            dset = None
            for c in cand:
                if c in g and isinstance(g[c], h5py.Dataset):
                    dset = g[c]; break
            if dset is None: return None
            a_g = _read_attrs(g) | _read_attrs(dset)
            fs_in = a_g.get("fs") or a_g.get("sample_rate")
            if not fs_in:
                xs = a_g.get("xspacing") or a_g.get("Xspacing")
                if not xs: return None
                fs_in = 1.0/float(xs)
            fs_in = int(round(float(fs_in)))
            wlen_in = int(round(win_sec*fs_in))
            if i0 < 0 or (i0 + wlen_in) > int(dset.shape[0]): return None
            x = dset[i0:i0+wlen_in].astype(np.float32, copy=True)

        if fs_in != fs_tgt:
            frac = Fraction(fs_tgt, fs_in).limit_denominator(64)
            x = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32, copy=False)

        wlen = int(round(win_sec*fs_tgt))
        if x.size > wlen: x = x[-wlen:]
        elif x.size < wlen:
            buf = np.zeros(wlen, np.float32); buf[-x.size:] = x; x = buf

        if not np.any(np.isfinite(x)): return None
        x -= x.mean(); x /= (x.std() + 1e-12)
        return x
    except Exception:
        return None

# =========================
# Dataset com cache LRU de janelas
# =========================
class WindowsDS(Dataset):
    def __init__(self, df: pd.DataFrame, win_index: Dict[str,str], fs_tgt: int, win_sec: float,
                 teacher_map: Optional[Dict[Tuple[str,float], float]] = None, aug: bool = False,
                 time_dropout_p: float = 0.0, timeshift_sec: float = 0.0, gauss_std: float = 0.0,
                 cache_cap: int = 20000):
        self.df = df.reset_index(drop=True).copy()
        self.idx_map = win_index
        self.fs = fs_tgt; self.sec = win_sec
        self.teacher = teacher_map or {}
        self.aug = aug; self.tdp = float(time_dropout_p)
        self.tshift = float(timeshift_sec); self.gstd = float(gauss_std)
        self._wht_cache: Dict[str,str] = {}
        self._x_cache: Dict[Tuple[str,float], np.ndarray] = {}
        self._cache_cap = int(cache_cap)

    def __len__(self): return len(self.df)

    def _get_wht(self, fid: str) -> Optional[str]:
        if fid in self._wht_cache: return self._wht_cache[fid]
        p = self.idx_map.get(fid)
        if not p:
            self._wht_cache[fid] = None; return None
        _, _, meta = _cached_start_arrays(p)
        wht = _resolve_whitened_path(p, meta)
        self._wht_cache[fid] = wht; return wht

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.tshift > 0.0:
            max_shift = int(round(self.tshift * self.fs))
            if max_shift > 0:
                k = np.random.randint(-max_shift, max_shift+1)
                if k != 0: x = np.roll(x, k)
        if self.gstd > 0.0:
            x = x + np.random.normal(0.0, self.gstd, size=x.shape).astype(np.float32)
        if self.tdp > 0.0:
            mask = np.random.rand(*x.shape) < self.tdp
            x = x.copy(); x[mask] = 0.0
        x = (x - x.mean()) / (x.std() + 1e-12)
        return x

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        fid = str(row["file_id"]); sgps = float(row["start_gps"]); y = int(row["label"])
        key = (fid, sgps)

        x = self._x_cache.get(key)
        if x is None:
            wht = self._get_wht(fid)
            if (not wht) or (not os.path.exists(wht)):
                L = int(round(self.sec*self.fs)); x = np.zeros(L, np.float32)
            else:
                win_path = self.idx_map.get(fid)
                x = _load_window_slice(win_path, wht, sgps, self.fs, self.sec)
                if x is None:
                    L = int(round(self.sec*self.fs)); x = np.zeros(L, np.float32)
            if len(self._x_cache) >= self._cache_cap:
                self._x_cache.pop(next(iter(self._x_cache)))
            self._x_cache[key] = x

        if self.aug: x = self._augment(x)
        t = torch.from_numpy(x).float().unsqueeze(0)
        teach = float(self.teacher.get((fid, sgps), np.nan))
        return t, torch.tensor(y, dtype=torch.float32), teach, fid, sgps

# =========================
# Modelo
# =========================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, p=None, dropout=0.0):
        super().__init__()
        if p is None: p = (k-1)//2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class MFModel(nn.Module):
    def __init__(self, ch_mf1=32, ch_mf2=48, dropout=0.05, checkpointing=False):
        super().__init__()
        self.checkpointing = bool(checkpointing)
        # MF1
        self.mf1 = nn.Sequential(
            ConvBlock(1, ch_mf1, 9, s=2, dropout=dropout),
            ConvBlock(ch_mf1, ch_mf1, 7, s=2, dropout=dropout),
            ConvBlock(ch_mf1, ch_mf1, 5, s=1, dropout=dropout),
        )
        self.mf1_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch_mf1, 1))
        # MF2
        self.proj_z1 = nn.Conv1d(ch_mf1, 16, kernel_size=1)
        self.mf2 = nn.Sequential(
            ConvBlock(1+16, ch_mf2, 9, s=2, dropout=dropout),
            ConvBlock(ch_mf2, ch_mf2, 7, s=2, dropout=dropout),
            ConvBlock(ch_mf2, ch_mf2, 5, s=1, dropout=dropout),
            ConvBlock(ch_mf2, ch_mf2, 3, s=2, dropout=dropout),
        )
        self.mf2_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch_mf2, 1))

    def forward(self, x):
        z1 = self.mf1(x); y1 = self.mf1_head(z1).squeeze(-1)
        L = x.shape[-1]; L1 = z1.shape[-1]
        up = F.interpolate(self.proj_z1(z1), size=max(L//4, L1), mode="nearest")
        if up.shape[-1] != x.shape[-1]:
            up = F.interpolate(up, size=L, mode="nearest")
        inp2 = torch.cat([x, up], dim=1)
        h2 = checkpoint_sequential(self.mf2, segments=3, input=inp2) if self.checkpointing and self.training else self.mf2(inp2)
        y2 = self.mf2_head(h2).squeeze(-1)
        return y1, y2, z1, h2

# =========================
# Métricas
# =========================
@torch.no_grad()
def eval_epoch(model, dl, device, amp=True):
    model.eval(); all_y, all_p = [], []
    for xb, yb, _, _, _ in dl:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, y2, _, _ = model(xb)
        all_y.append(yb.detach().float().cpu().numpy())
        all_p.append(torch.sigmoid(y2).detach().float().cpu().numpy())
    y = np.concatenate(all_y) if all_y else np.zeros(0); p = np.concatenate(all_p) if all_p else np.zeros(0)
    if len(np.unique(y.astype(int))) >= 2:
        auc = float(roc_auc_score(y, p)); ap  = float(average_precision_score(y, p))
    else:
        auc, ap = float("nan"), float("nan")
    return {"auc": auc, "ap": ap, "n": int(y.size)}

def make_teacher_map(csv_path: Optional[str], key_cols=("file_id","start_gps"), val_col="score"):
    if not csv_path or not os.path.exists(csv_path): return {}
    df = pd.read_csv(csv_path); need = set(key_cols) | {val_col}
    if not need.issubset(df.columns): return {}
    out = {}
    for _, r in df.iterrows():
        out[(str(r[key_cols[0]]), float(r[key_cols[1]]))] = float(r[val_col])
    return out

# =========================
# Autotune de batch via env MF_AUTOBATCH=1
# =========================
def _try_forward(model, opt, scaler, xb, yb, tb, hp, device):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=hp.AMP):
        y1, y2, _, _ = model(xb)
        n_pos = (yb > 0.5).sum().item(); n_neg = (yb <= 0.5).sum().item()
        pos_w = float(max(1.0, n_neg / max(1, n_pos))) if n_pos > 0 else 1.0
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        loss_bce = bce(y2, yb)
        loss_consist = F.mse_loss(torch.sigmoid(y1).detach(), torch.sigmoid(y2)) + F.mse_loss(torch.sigmoid(y1), torch.sigmoid(y2).detach())
        has_teacher = torch.isfinite(tb)
        loss_distil = F.mse_loss(torch.sigmoid(y2)[has_teacher], torch.clamp(tb[has_teacher], 0.0, 1.0)) if has_teacher.any() else torch.tensor(0.0, device=device)
        loss = hp.W_BCE*loss_bce + hp.W_CONSIST*loss_consist + hp.W_DISTIL*loss_distil
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return float(loss.detach().cpu().item())

def autotune_batch(hp: HParams, ds_tr: Dataset, device) -> int:
    if not torch.cuda.is_available(): return hp.BATCH
    candidate = max(128, hp.BATCH); max_try = 2048
    model = MFModel(hp.CH_MF1, hp.CH_MF2, hp.DROPOUT, checkpointing=hp.CHECKPOINTING).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.LR, weight_decay=hp.WD, betas=(0.9,0.999), fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(hp.AMP and device.type=="cuda"))
    def mk(b):
        return DataLoader(ds_tr, batch_size=b, shuffle=True, num_workers=hp.NUM_WORKERS,
                          pin_memory=hp.PIN_MEMORY, drop_last=True, persistent_workers=(hp.PERSISTENT and hp.NUM_WORKERS>0),
                          prefetch_factor=hp.PREFETCH if hp.NUM_WORKERS>0 else None)
    best = hp.BATCH; lo, hi = 32, min(max_try, 2048)
    while candidate <= hi:
        try:
            xb, yb, tb, _, _ = next(iter(mk(candidate)))
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True); tb = torch.tensor(tb, dtype=torch.float32, device=device)
            _try_forward(model, opt, scaler, xb, yb, tb, hp, device); torch.cuda.synchronize(); best = candidate; candidate *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); break
            else: raise
    lo = best; hi = min(candidate, hi)
    while lo < hi - 32:
        mid = ((lo + hi) // 64) * 64; mid = max(32, mid)
        try:
            xb, yb, tb, _, _ = next(iter(mk(mid)))
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True); tb = torch.tensor(tb, dtype=torch.float32, device=device)
            _try_forward(model, opt, scaler, xb, yb, tb, hp, device); torch.cuda.synchronize(); lo = mid; best = mid
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); hi = mid
            else: raise
    return int(best)

# =========================
# Treino
# =========================
def _mk_loader(ds, batch, shuffle, hp: HParams, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=hp.NUM_WORKERS, pin_memory=hp.PIN_MEMORY,
                      drop_last=drop_last, persistent_workers=(hp.PERSISTENT and hp.NUM_WORKERS>0),
                      prefetch_factor=hp.PREFETCH if hp.NUM_WORKERS>0 else None)

def train(hparams: Optional[Dict] = None):
    hp = HParams()
    if hparams:
        for k, v in hparams.items():
            if hasattr(hp, k): setattr(hp, k, v)

    set_seed(hp.SEED)
    device = torch.device(hp.DEVICE if torch.cuda.is_available() and hp.DEVICE.startswith("cuda") else "cpu")

    ts = time.strftime("%Y%m%d-%H%M%S"); out_dir = os.path.join(hp.OUT_DIR, ts)
    os.makedirs(out_dir, exist_ok=True); eta_path = os.path.join(out_dir, "eta.json")
    with open(os.path.join(out_dir, "hparams.json"), "w") as f: json.dump(asdict(hp), f, indent=2)

    df = pd.read_parquet(hp.PARQUET_PATH)
    if "subset" not in df.columns:
        if "split" in df.columns: df = df.rename(columns={"split":"subset"})
        else: raise KeyError("dataset.parquet precisa de subset ou split")
    if not {"file_id","start_gps","label","subset"}.issubset(df.columns):
        raise KeyError("dataset.parquet sem colunas requeridas")

    idx_map = build_windows_index(hp.WINDOWS_DIR)
    t_map_val = make_teacher_map(hp.TEACHER_CSV_VAL, val_col=hp.TEACHER_COL)
    t_map_tst = make_teacher_map(hp.TEACHER_CSV_TST, val_col=hp.TEACHER_COL)

    df_train = df[df["subset"]=="train"].copy()
    df_val   = df[df["subset"]=="val"].copy()
    df_test  = df[df["subset"]=="test"].copy()

    ds_tr = WindowsDS(df_train, idx_map, hp.FS_TARGET, hp.WINDOW_SEC, teacher_map=None, aug=True,
                      time_dropout_p=hp.TIME_DROPOUT_P, timeshift_sec=hp.AUG_TIMESHIFT_SEC,
                      gauss_std=hp.AUG_GAUSS_STD, cache_cap=20000)
    ds_va = WindowsDS(df_val, idx_map, hp.FS_TARGET, hp.WINDOW_SEC, teacher_map=t_map_val or None, aug=False, cache_cap=4000)
    ds_te = WindowsDS(df_test, idx_map, hp.FS_TARGET, hp.WINDOW_SEC, teacher_map=t_map_tst or None, aug=False, cache_cap=4000)

    if os.environ.get("MF_AUTOBATCH","0") == "1":  # sem args
        best_b = autotune_batch(hp, ds_tr, device); hp.BATCH = int(best_b)

    dl_tr = _mk_loader(ds_tr, hp.BATCH, True, hp, drop_last=True)
    dl_va = _mk_loader(ds_va, max(256, hp.BATCH), False, hp)
    dl_te = _mk_loader(ds_te, max(256, hp.BATCH), False, hp)

    model = MFModel(hp.CH_MF1, hp.CH_MF2, hp.DROPOUT, checkpointing=hp.CHECKPOINTING).to(device)
    if hp.COMPILE:
        try: model = torch.compile(model, mode="max-autotune", fullgraph=False)
        except Exception: pass

    opt = torch.optim.AdamW(model.parameters(), lr=hp.LR, weight_decay=hp.WD, betas=(0.9,0.999), fused=torch.cuda.is_available())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(hp.EPOCHS, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(hp.AMP and device.type=="cuda"))

    best_ap, best_auc = -1.0, -1.0
    best_ap_path = os.path.join(out_dir, "model_best_ap.pt")
    best_auc_path = os.path.join(out_dir, "model_best_auc.pt")

    total_steps = hp.EPOCHS * max(1, len(dl_tr)); global_step = 0
    pbar = tqdm(total=total_steps, desc="MF2 train", ncols=120); eta_meter = ETAMeter(0.12)

    for epoch in range(1, hp.EPOCHS+1):
        model.train(); t0 = time.time(); loss_meter = 0.0; n_steps = 0
        freeze_mf1 = epoch > hp.EPOCHS - 4
        if freeze_mf1:
            for p in list(model.mf1.parameters()) + list(model.mf1_head.parameters()):
                p.requires_grad = False

        for xb, yb, tb, _, _ in dl_tr:
            eta_meter.start_step()
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            tb = torch.tensor(tb, dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=hp.AMP):
                y1, y2, _, _ = model(xb)
                n_pos = (yb > 0.5).sum().item(); n_neg = (yb <= 0.5).sum().item()
                pos_w = float(max(1.0, n_neg / max(1, n_pos))) if n_pos > 0 else 1.0
                bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
                loss_bce = bce(y2, yb)
                loss_consist = F.mse_loss(torch.sigmoid(y1).detach(), torch.sigmoid(y2)) + F.mse_loss(torch.sigmoid(y1), torch.sigmoid(y2).detach())
                has_teacher = torch.isfinite(tb)
                loss_distil = F.mse_loss(torch.sigmoid(y2)[has_teacher], torch.clamp(tb[has_teacher], 0.0, 1.0)) if has_teacher.any() else torch.tensor(0.0, device=device)
                loss = hp.W_BCE*loss_bce + hp.W_CONSIST*loss_consist + hp.W_DISTIL*loss_distil

            scaler.scale(loss).backward()
            if hp.GRAD_CLIP and hp.GRAD_CLIP > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), hp.GRAD_CLIP)
            scaler.step(opt); scaler.update()

            loss_meter += float(loss.detach().cpu().item()); n_steps += 1
            eta_meter.end_step(); global_step += 1
            remaining = max(0, total_steps - global_step); eta_sec = eta_meter.eta(remaining)
            pbar.set_postfix(epoch=f"{epoch}/{hp.EPOCHS}", loss=f"{loss_meter/max(n_steps,1):.4f}", step=f"{global_step}/{total_steps}", eta=_fmt_hms(eta_sec))
            pbar.update(1)

            if global_step % 20 == 0 or remaining == 0:
                with open(eta_path, "w") as f:
                    json.dump({"epoch": epoch, "epochs_total": hp.EPOCHS, "step": global_step, "steps_total": total_steps,
                               "avg_batch_sec": None if eta_meter.ema is None else float(eta_meter.ema),
                               "eta_seconds": float(eta_sec), "eta_hms": _fmt_hms(eta_sec),
                               "estimated_finish_unix": time.time() + float(eta_sec)}, f, indent=2)

        sched.step(); dt = time.time() - t0

        do_eval = (epoch == 1) or (epoch % max(1, hp.EVAL_EVERY) == 0) or (epoch == hp.EPOCHS)
        if do_eval:
            met_val = eval_epoch(model, dl_va, device, amp=hp.AMP)
            met_tst = eval_epoch(model, dl_te, device, amp=hp.AMP)
        else:
            met_val = {"auc": float("nan"), "ap": float("nan"), "n": len(ds_va)}
            met_tst = {"auc": float("nan"), "ap": float("nan"), "n": len(ds_te)}

        row = {"epoch": epoch, "loss": loss_meter/max(n_steps,1), "lr": float(opt.param_groups[0]["lr"]),
               "val_auc": met_val["auc"], "val_ap": met_val["ap"], "val_n": met_val["n"],
               "test_auc": met_tst["auc"], "test_ap": met_tst["ap"], "test_n": met_tst["n"], "time_s": dt}
        print(f"[MF][{epoch}/{hp.EPOCHS}] loss={row['loss']:.4f} valAUC={row['val_auc']:.4f} valAP={row['val_ap']:.4f} testAUC={row['test_auc']:.4f} testAP={row['test_ap']:.4f} {dt:.1f}s")

        if not math.isnan(row["val_ap"]) and row["val_ap"] > best_ap:
            best_ap = row["val_ap"]; torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": row}, best_ap_path)
        if not math.isnan(row["val_auc"]) and row["val_auc"] > best_auc:
            best_auc = row["val_auc"]; torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": row}, best_auc_path)

        if (epoch % max(1, hp.SAVE_EVERY)) == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": row}, os.path.join(out_dir, f"model_ep{epoch:03d}.pt"))
        with open(os.path.join(out_dir, "train_log.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")

        remaining_epochs = max(0, hp.EPOCHS - epoch)
        est_epoch_sec = dt if eta_meter.ema is None else eta_meter.ema * max(1, len(dl_tr))
        eta_all_sec = remaining_epochs * est_epoch_sec
        with open(eta_path, "w") as f:
            json.dump({"epoch": epoch, "epochs_total": hp.EPOCHS, "avg_epoch_sec": float(est_epoch_sec),
                       "eta_seconds": float(eta_all_sec), "eta_hms": _fmt_hms(eta_all_sec),
                       "estimated_finish_unix": time.time() + float(eta_all_sec)}, f, indent=2)

    pbar.close()
    summary = {"best_val_ap": float(best_ap), "best_val_auc": float(best_auc),
               "best_ap_path": best_ap_path if best_ap >= 0 else None,
               "best_auc_path": best_auc_path if best_auc >= 0 else None, "hparams": asdict(hp)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print("[MF] treino concluído."); print(" Saída:", out_dir)
    print(" Best@AP:", summary["best_ap_path"]); print(" Best@AUC:", summary["best_auc_path"])
    return summary

# =========================
# Export
# =========================
def _ts(): return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def _choose_threshold_by_far(labels, scores, target_fpr=1e-4):
    labels = np.asarray(labels).astype(int); scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores, kind="mergesort"); y = labels[order]; s = scores[order]
    n0 = int((labels == 0).sum()); n1 = int((labels == 1).sum()); fp = 0; tp = 0
    best = {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0, "precision": 1.0, "recall": 0.0}
    for i in range(len(s)):
        if y[i] == 1: tp += 1
        else: fp += 1
        fpr = fp / max(n0, 1); tpr = tp / max(n1, 1)
        prec = tp / max(tp+fp, 1); rec = tpr
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
    y = np.asarray(labels).astype(int); s = np.asarray(scores).astype(float)
    pred = (s >= float(thr)).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return tn, fp, fn, tp

@torch.no_grad()
def _predict_scores(model, dl, device, desc="scoring"):
    model.eval(); rows = []; pbar = tqdm(total=len(dl), desc=desc, ncols=120)
    for xb, yb, _, fid, sgps in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
            _, y2, _, _ = model(xb); p = torch.sigmoid(y2).detach().float().cpu().numpy()
        yb = yb.float().cpu().numpy()
        for i in range(len(p)): rows.append((str(fid[i]), float(sgps[i]), int(yb[i]), float(p[i])))
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows, columns=["file_id","start_gps","label","score"])

def export_latest(out_dir_root: Optional[str] = None):
    """
    Usa o best checkpoint do último treino em runs/mf/<ts> e só exporta scores/thresholds.
    """
    hp = HParams()
    root_runs = hp.OUT_DIR
    if not os.path.exists(root_runs): raise FileNotFoundError("Nenhum run encontrado em runs/mf")
    sub = sorted([d for d in os.listdir(root_runs) if os.path.isdir(os.path.join(root_runs, d))])
    if not sub: raise FileNotFoundError("Nenhum run encontrado em runs/mf")
    last = os.path.join(root_runs, sub[-1])
    for cand in ["model_best_ap.pt", "model_best_auc.pt"]:
        best_path = os.path.join(last, cand)
        if os.path.exists(best_path): break
    else:
        raise FileNotFoundError("Best checkpoint não encontrado no último run")

    # saída
    root = out_dir_root or os.path.join("reports", "mf_stage2")
    os.makedirs(root, exist_ok=True)
    out_dir = os.path.join(root, _ts()); os.makedirs(out_dir, exist_ok=True)

    # dados
    df = pd.read_parquet(hp.PARQUET_PATH)
    if "subset" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split":"subset"})
    if not {"file_id","start_gps","label","subset"}.issubset(df.columns):
        raise KeyError("dataset.parquet sem colunas requeridas")
    idx_map = build_windows_index(hp.WINDOWS_DIR)

    df_val  = df[df["subset"]=="val"].copy()
    df_test = df[df["subset"]=="test"].copy()

    ds_va = WindowsDS(df_val, idx_map, hp.FS_TARGET, hp.WINDOW_SEC, aug=False, cache_cap=8000)
    ds_te = WindowsDS(df_test, idx_map, hp.FS_TARGET, hp.WINDOW_SEC, aug=False, cache_cap=8000)

    dl_va = _mk_loader(ds_va, 1024, False, hp)
    dl_te = _mk_loader(ds_te, 1024, False, hp)

    device = torch.device("cuda" if torch.cuda.is_available() and hp.DEVICE.startswith("cuda") else "cpu")
    model = MFModel(hp.CH_MF1, hp.CH_MF2, hp.DROPOUT).to(device)
    state = torch.load(best_path, map_location=device); model.load_state_dict(state["state_dict"], strict=False)

    val_csv  = os.path.join(out_dir, "scores_val.csv")
    test_csv = os.path.join(out_dir, "scores_test.csv")
    _predict_scores(model, dl_va, device, desc="MF2 scoring VAL").to_csv(val_csv, index=False)
    _predict_scores(model, dl_te, device, desc="MF2 scoring TEST").to_csv(test_csv, index=False)

    val_df  = pd.read_csv(val_csv); test_df = pd.read_csv(test_csv)
    info_thr = _choose_threshold_by_far(val_df["label"].values, val_df["score"].values, target_fpr=1e-4)
    thr = float(info_thr["threshold"])
    tn, fp, fn, tp = _confusion_at_thr(test_df["label"].values, test_df["score"].values, thr)
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1); fpr = fp / max(tn+fp, 1)

    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen_threshold": thr, "mode": "constrained_roc", "target_fpr": 1e-4, "info_val": info_thr}, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"cfg": {"MODE": "constrained_roc", "TARGET_FAR": 1e-4},
                   "gpu": {"requested": True, "available": torch.cuda.is_available()},
                   "test": {"auc": None, "ap": None, "confusion_at_thr": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                            "precision": float(prec), "recall": float(rec), "fpr": float(fpr)},
                   "threshold": thr, "threshold_info": info_thr}, f, indent=2)
    print(f"[MF2] OK. Saída: {out_dir}")
    return out_dir

def run_mf_stage2(out_dir_root: Optional[str] = None):
    """
    Caminho de pipeline: treina e exporta MF2.
    """
    train_summary = train()
    # usa o melhor do run recém criado
    best_path = train_summary.get("best_ap_path") or train_summary.get("best_auc_path")
    if not best_path or not os.path.exists(best_path):
        raise RuntimeError("MF2 treinou, mas não encontrei best_*_path no resumo.")
    return export_latest(out_dir_root)

# Execução direta do módulo segue o caminho do pipeline
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_mf_stage2()
