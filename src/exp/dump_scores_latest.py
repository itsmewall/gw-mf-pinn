# src/exp/dump_scores_latest.py
import os, time, json
import pandas as pd
import torch

from src.training.train_mf import HParams, MFModel, build_windows_index, WindowsDS
from torch.utils.data import DataLoader

from src.utils.prof import log_step

def _latest(path):
    subs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return max(subs, key=os.path.getmtime) if subs else None

def main():
    hp = HParams()
    latest_run = _latest(hp.OUT_DIR)
    if not latest_run:
        raise RuntimeError("nenhum run em runs/mf/")
    ckpt = os.path.join(latest_run, "model_best_auc.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(latest_run, "model_best_ap.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFModel(ch_mf1=hp.CH_MF1, ch_mf2=hp.CH_MF2, dropout=hp.DROPOUT).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    df = pd.read_parquet(hp.PARQUET_PATH)
    if "subset" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split": "subset"})
    idx = build_windows_index(hp.WINDOWS_DIR)

    ds = WindowsDS(df, idx, hp.FS_TARGET, hp.WINDOW_SEC, teacher_map=None, aug=False)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    rows = []
    for xb, yb, _, fid, sgps in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            _, y2, _, _ = model(xb)
        p = torch.sigmoid(y2).detach().cpu().numpy()
        yb = yb.numpy()
        for f, s, y, sc in zip(fid, sgps, yb, p):
            rows.append((str(f), float(s), int(y), float(sc)))
    out_dir = os.path.join("results", "scores_dump")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dump_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    pd.DataFrame(rows, columns=["file_id","start_gps","label","score"]).to_csv(out_path, index=False)
    print("scores salvos em", out_path)

if __name__ == "__main__":
    with log_step("dump_scores", tag="mf2_weekend"):
        main()