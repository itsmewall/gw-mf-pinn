# viz/insight_views.py
# Visualizações de valor a partir dos artefatos já gerados
from __future__ import annotations
import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# --------------------------------------------------
# helpers
# --------------------------------------------------
def _latest_subdir(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    subs = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            subs.append(p)
    if not subs:
        return None
    subs.sort(key=os.path.getmtime)
    return subs[-1]

def _det_from_file_id(fid: str) -> str:
    if "H1" in fid: return "H1"
    if "L1" in fid: return "L1"
    if "V1" in fid: return "V1"
    return "UNK"

def _try_load_scores(reports_root: str) -> Optional[pd.DataFrame]:
    """
    Tenta em ordem:
      1) reports/mf_stage2/latest/scores_test.csv
      2) reports/mf_stage2/latest/scores_val.csv
      3) reports/mf_baseline/latest/scores_test.csv
    """
    mf2_root = os.path.join(reports_root, "mf_stage2")
    mf1_root = os.path.join(reports_root, "mf_baseline")

    # MF2
    latest_mf2 = _latest_subdir(mf2_root)
    if latest_mf2:
        for name in ("scores_test.csv", "scores_val.csv"):
            p = os.path.join(latest_mf2, name)
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    df["__source_run__"] = latest_mf2
                    return df
                except Exception:
                    pass

    # MF1
    latest_mf1 = _latest_subdir(mf1_root)
    if latest_mf1:
        p = os.path.join(latest_mf1, "scores_test.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df["__source_run__"] = latest_mf1
                return df
            except Exception:
                pass

    return None

# --------------------------------------------------
# 1) Painel de coincidência H1 x L1
# Pergunta que responde: os maiores scores estão aparecendo nos dois detectores perto um do outro?
# --------------------------------------------------
def plot_coincidence_panel(reports_root: str, out_dir: str, dt_sec: float = 0.015):
    df = _try_load_scores(reports_root)
    if df is None or "file_id" not in df.columns or "start_gps" not in df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    df["det"] = df["file_id"].astype(str).map(_det_from_file_id)
    # pega top 500 por score
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False).head(500)
    else:
        df = df.head(500)

    tH = df[df["det"]=="H1"]["start_gps"].astype(float).values
    tL = df[df["det"]=="L1"]["start_gps"].astype(float).values

    if len(tH) == 0 or len(tL) == 0:
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # scatter tempo x detector
    ax[0].scatter(tH, np.zeros_like(tH)+1, s=14, c="tab:red", label="H1")
    ax[0].scatter(tL, np.zeros_like(tL)-1, s=14, c="tab:blue", label="L1")
    ax[0].set_yticks([-1, 1])
    ax[0].set_yticklabels(["L1", "H1"])
    ax[0].set_xlabel("tempo GPS")
    ax[0].set_title("Eventos com maior score")
    ax[0].legend(loc="upper right")

    # histograma de diferenças
    diffs = []
    for th in tH:
        close = np.abs(tL - th)
        m = close.min()
        diffs.append(m)
    diffs = np.array(diffs)

    ax[1].hist(diffs, bins=30, color="gray", edgecolor="black")
    ax[1].axvline(dt_sec, color="red", linestyle="--", label=f"janela {dt_sec}s")
    ax[1].set_xlabel("|H1 - L1| (s)")
    ax[1].set_ylabel("contagem")
    ax[1].set_title("Distância temporal H1 x L1")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coincidence_panel.png"), dpi=160)
    plt.close(fig)

# --------------------------------------------------
# 2) Overlay da janela real com a waveform do template
# Pergunta que responde: o que o detector viu parece com o que o banco de templates espera?
# --------------------------------------------------
def plot_waveform_overlay(data_processed: str, reports_root: str, out_dir: str, idx: int = 0):
    """
    Pega a janela de maior score e plota junto com uma IMRPhenomD sintética.
    Não precisa de args no pipeline, só usa a primeira janela que achar.
    """
    try:
        from pycbc.waveform import get_td_waveform
    except Exception:
        return

    df = _try_load_scores(reports_root)
    if df is None:
        return

    if "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if idx >= len(df):
        return

    row = df.iloc[idx]
    file_id = str(row["file_id"])
    start_gps = float(row["start_gps"])
    win_path = os.path.join(data_processed, f"{file_id}_windows.hdf5")
    if not os.path.exists(win_path):
        return

    import h5py
    with h5py.File(win_path, "r") as h5:
        # janela mais simples: pega um dataset que tenha start_gps
        # muitos hdf5 desse tipo usam algo como /windows/data e /windows/meta
        # vamos tentar padrão
        if "windows" in h5:
            grp = h5["windows"]
            # tenta encontrar o índice com start_gps mais perto
            gps_arr = grp["start_gps"][:]
            idx_sel = int(np.argmin(np.abs(gps_arr - start_gps)))
            sig = grp["data"][idx_sel, :]
            fs = grp.attrs.get("fs", 4096.0)
        else:
            return

    # waveform sintética
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=30, mass2=30,
        delta_t=1.0/fs,
        f_lower=20.0,
    )
    wv = hp.numpy()
    # recorta ou interpola para o tamanho da janela
    if len(wv) > len(sig):
        wv = wv[-len(sig):]
    elif len(wv) < len(sig):
        pad = len(sig) - len(wv)
        wv = np.pad(wv, (pad, 0))
    # normaliza os dois
    sig_n = sig / (np.max(np.abs(sig)) + 1e-9)
    wv_n = wv / (np.max(np.abs(wv)) + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(len(sig_n)) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, sig_n, label="sinal detectado", alpha=0.85)
    plt.plot(t, wv_n, label="waveform IMRPhenomD (30,30)", alpha=0.6)
    plt.title(f"Overlay sinal x template esperado\nfile_id={file_id} gps={start_gps}")
    plt.xlabel("tempo (s)")
    plt.ylabel("amplitude normalizada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overlay_signal_template.png"), dpi=160)
    plt.close()

# --------------------------------------------------
# 3) Mapa de calor de score ao longo do tempo
# Pergunta que responde: há períodos com mais atividade? há faixas de score mostrando burst?
# --------------------------------------------------
def plot_score_heatmap(reports_root: str, out_dir: str, bins: int = 48):
    df = _try_load_scores(reports_root)
    if df is None or "start_gps" not in df.columns:
        return
    os.makedirs(out_dir, exist_ok=True)

    # normaliza o tempo para 0..1
    t = df["start_gps"].astype(float).values
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return
    t_norm = (t - t_min) / (t_max - t_min + 1e-9)

    if "score" in df.columns:
        s = df["score"].astype(float).values
    else:
        s = np.ones_like(t) * 0.5

    # bin temporal
    edges = np.linspace(0.0, 1.0, bins+1)
    heat = np.zeros((2, bins), float)  # linha 0 H1, linha 1 L1
    for i in range(len(t_norm)):
        b = np.searchsorted(edges, t_norm[i], side="right") - 1
        b = max(0, min(b, bins-1))
        det = _det_from_file_id(str(df.iloc[i]["file_id"]))
        if det == "H1":
            heat[0, b] = max(heat[0, b], s[i])
        elif det == "L1":
            heat[1, b] = max(heat[1, b], s[i])

    fig, ax = plt.subplots(figsize=(10, 2.6))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks([0,1])
    ax.set_yticklabels(["H1","L1"])
    ax.set_xticks(np.linspace(0, bins-1, 5))
    ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(t_min, t_max, 5)])
    ax.set_xlabel("tempo GPS (normalizado)")
    ax.set_title("Mapa de calor de score por detector")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score max no bin")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_heatmap.png"), dpi=160)
    plt.close(fig)

# --------------------------------------------------
# 4) Projeção 2D de features do MF2
# Pergunta que responde: os positivos ficam agrupados em um canto do espaço de features?
# Requer que o run do MF2 tenha salvo runs/mf/<ts>/features.npy e runs/mf/<ts>/labels.npy
# --------------------------------------------------
def plot_feature_projection(project_root: str, out_dir: str):
    runs_mf = os.path.join(project_root, "runs", "mf")
    latest = _latest_subdir(runs_mf)
    if not latest:
        return
    feat_p = os.path.join(latest, "features.npy")
    lab_p = os.path.join(latest, "labels.npy")
    if not (os.path.exists(feat_p) and os.path.exists(lab_p)):
        return

    feats = np.load(feat_p)  # shape [N, D]
    labs = np.load(lab_p)    # shape [N]
    if feats.ndim != 2:
        return

    # redução bem simples para não trazer sklearn extra
    # pega as duas primeiras componentes da covariância
    X = feats - feats.mean(axis=0, keepdims=True)
    cov = X.T @ X / max(len(X)-1, 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    w = vecs[:, idx[:2]]   # [D, 2]
    proj = X @ w           # [N, 2]

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5,5))
    for lab, color, name in [(0, "gray", "neg"), (1, "red", "pos")]:
        sel = labs == lab
        if not np.any(sel):
            continue
        plt.scatter(proj[sel,0], proj[sel,1], s=6, c=color, alpha=0.4, label=name)
    plt.legend()
    plt.title("Projeção 2D das features do MF2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mf2_feature_projection.png"), dpi=160)
    plt.close()

# --------------------------------------------------
# função única chamada pelo run.py
# --------------------------------------------------
def generate_all_views(project_root: str):
    reports_root = os.path.join(project_root, "reports")
    data_processed = os.path.join(project_root, "data", "processed")
    viz_out = os.path.join(project_root, "reports", "viz_insights")
    os.makedirs(viz_out, exist_ok=True)

    plot_coincidence_panel(reports_root, viz_out, dt_sec=0.015)
    plot_waveform_overlay(data_processed, reports_root, viz_out, idx=0)
    plot_score_heatmap(reports_root, viz_out, bins=48)
    plot_feature_projection(project_root, viz_out)
