# src/viz/waveview.py
from __future__ import annotations
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib

# usa backend sem janela
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# vamos reaproveitar o que já existe no projeto
from eval import mf_baseline as mbf


def _now_str():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _latest_subdir(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    subs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not subs:
        return None
    return max(subs, key=os.path.getmtime)


def _try_load_scores_from_report(report_dir: str) -> pd.DataFrame | None:
    """
    Procura scores_test.csv ou scores_val.csv dentro de um report.
    Retorna DataFrame com, no mínimo, file_id, start_gps e score.
    """
    if not report_dir or not os.path.isdir(report_dir):
        return None

    cand_files = [
        os.path.join(report_dir, "scores_test.csv"),
        os.path.join(report_dir, "scores_val.csv"),
    ]
    for p in cand_files:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # normaliza nomes mais prováveis
                cols = {c.lower(): c for c in df.columns}
                # precisa ter pelo menos file_id e start_gps
                need = []
                if "file_id" in df.columns:
                    need.append("file_id")
                elif "fileid" in cols:
                    df = df.rename(columns={cols["fileid"]: "file_id"})
                    need.append("file_id")

                if "start_gps" in df.columns:
                    need.append("start_gps")
                elif "start" in cols:
                    df = df.rename(columns={cols["start"]: "start_gps"})
                    need.append("start_gps")

                if "score" not in df.columns and "ncc" in cols:
                    df = df.rename(columns={cols["ncc"]: "score"})

                if not {"file_id", "start_gps", "score"}.issubset(df.columns):
                    # estrutura inesperada
                    continue

                return df
            except Exception:
                continue
    return None


def _find_best_scored_windows(reports_dir: str, max_rows: int = 5) -> pd.DataFrame | None:
    """
    1. tenta o último reports/mf_baseline/*
    2. se não tiver, tenta o último reports/mf_stage2/*
    """
    # 1) mf_baseline
    base_root = os.path.join(reports_dir, "mf_baseline")
    latest_base = _latest_subdir(base_root)
    if latest_base:
        df = _try_load_scores_from_report(latest_base)
        if df is not None:
            return df.sort_values("score", ascending=False).head(max_rows).reset_index(drop=True)

    # 2) mf_stage2
    stage2_root = os.path.join(reports_dir, "mf_stage2")
    latest_stage2 = _latest_subdir(stage2_root)
    if latest_stage2:
        df = _try_load_scores_from_report(latest_stage2)
        if df is not None:
            return df.sort_values("score", ascending=False).head(max_rows).reset_index(drop=True)

    return None


def _fallback_from_dataset(data_processed: str, max_rows: int = 5) -> pd.DataFrame | None:
    """
    Se não tiver scores, cai pro dataset.parquet e pega as primeiras ondas de val ou test.
    """
    p = os.path.join(data_processed, "dataset.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    # dataset pode ter subset ou split
    if "subset" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split": "subset"})
    if "subset" in df.columns:
        # prefere positivos de val
        df_pos = df[df["label"] == 1] if "label" in df.columns else df
        df_val = df_pos[df_pos["subset"].astype(str).str.lower().eq("val")]
        if df_val.empty:
            df_val = df[df["subset"].astype(str).str.lower().eq("val")]
        take = df_val.head(max_rows).copy()
    else:
        take = df.head(max_rows).copy()

    # normaliza colunas
    if "file_id" not in take.columns and "fileid" in take.columns:
        take = take.rename(columns={"fileid": "file_id"})
    if "start_gps" not in take.columns and "start" in take.columns:
        take = take.rename(columns={"start": "start_gps"})

    if not {"file_id", "start_gps"}.issubset(take.columns):
        return None

    # adiciona score fake so a pipeline ficar igual
    take["score"] = 1.0
    return take.reset_index(drop=True)


def _ensure_out_dir(base: str = "results/viz_wave") -> str:
    out_dir = os.path.join(base, _now_str())
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _plot_time_series(x: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(10, 3))
    t = np.arange(len(x))
    plt.plot(t, x, lw=1.0, color="#0b7285")
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("strain (whitened)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_spec2d(x: np.ndarray, fs: float, out_path: str, title: str):
    plt.figure(figsize=(6, 4))
    # usa uma janela pequena para não matar o WSL
    nperseg = min(256, len(x))
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=nperseg, Fs=fs, noverlap=nperseg // 2, cmap="viridis")
    plt.title(title)
    plt.ylabel("freq [Hz]")
    plt.xlabel("tempo [s]")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_spec3d(x: np.ndarray, fs: float, out_path: str, title: str):
    # espectrograma 3D simplificado
    nperseg = min(256, len(x))
    Pxx, freqs, bins = _specgram_array(x, fs, nperseg)
    if Pxx is None:
        return
    T, F = np.meshgrid(bins, freqs)
    Z = 10.0 * np.log10(Pxx + 1e-10)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, F, Z, cmap="plasma")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("freq [Hz]")
    ax.set_zlabel("dB")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def _specgram_array(x: np.ndarray, fs: float, nperseg: int):
    try:
        from matplotlib.mlab import specgram
        Pxx, freqs, bins = specgram(
            x,
            NFFT=nperseg,
            Fs=fs,
            noverlap=nperseg // 2,
            scale_by_freq=True,
            mode="psd",
        )
        return Pxx, freqs, bins
    except Exception:
        return None, None, None


def generate_from_pipeline(
    data_processed: str,
    reports_dir: str,
    max_plots: int = 3,
    fs_fallback: float = 4096.0,
) -> str:
    """
    Gera figuras usando os dados realmente analisados.
    1. tenta pegar top scores de mf_baseline ou mf_stage2
    2. carrega as janelas via funções do próprio mf_baseline
    3. salva PNGs
    """
    out_dir = _ensure_out_dir()

    # 1) pega candidatos
    df = _find_best_scored_windows(reports_dir, max_rows=max_plots)
    if df is None or df.empty:
        # cai pro dataset
        df = _fallback_from_dataset(data_processed, max_rows=max_plots)
    if df is None or df.empty:
        # nada para plotar
        with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "empty"}, f, indent=2)
        return out_dir

    # 2) índice de janelas
    idx_map = mbf.build_windows_index(data_processed)

    # 3) percorre candidatos
    plotted = []
    for i, row in df.iterrows():
        file_id = str(row["file_id"])
        start_gps = float(row["start_gps"])
        score = float(row.get("score", 0.0))

        win_path = idx_map.get(file_id)
        if not win_path or not os.path.exists(win_path):
            continue

        # pega o whitened associado usando a função do projeto
        sgps_arr, _, meta_in = mbf.cached_start_arrays(win_path)
        whitened_path = mbf.resolve_whitened_path(win_path, meta_in)

        # carrega a janela real
        x = mbf.load_window_slice(win_path, whitened_path, start_gps)
        if x is None:
            continue
        x = np.asarray(x, dtype=np.float32)

        # tenta pegar fs real do meta
        fs = float(meta_in.get("fs", fs_fallback)) if isinstance(meta_in, dict) else fs_fallback

        base_name = f"{i:02d}_{file_id.replace('/', '_')}_{start_gps:.3f}"

        _plot_time_series(
            x,
            os.path.join(out_dir, f"{base_name}_ts.png"),
            title=f"{file_id} @ {start_gps:.3f}  score={score:.5f}",
        )
        _plot_spec2d(
            x,
            fs,
            os.path.join(out_dir, f"{base_name}_spec2d.png"),
            title=f"{file_id} @ {start_gps:.3f}",
        )
        _plot_spec3d(
            x,
            fs,
            os.path.join(out_dir, f"{base_name}_spec3d.png"),
            title=f"{file_id} @ {start_gps:.3f}",
        )

        plotted.append(
            {
                "file_id": file_id,
                "start_gps": start_gps,
                "score": score,
                "win_path": win_path,
                "whitened_path": whitened_path,
            }
        )

    # salva um jsonzinho com o que foi plotado
    with open(os.path.join(out_dir, "plotted.json"), "w", encoding="utf-8") as f:
        json.dump(plotted, f, indent=2)

    return out_dir