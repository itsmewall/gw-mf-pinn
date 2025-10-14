# src/viz/plot_scores_timeline.py
# -----------------------------------------------------------------------------------
# Gera gráficos "score vs tempo" por file_id, combinando:
# - MF (scores_test.csv do último reports/mf_baseline/<tag>/)
# - ML (scores_test.parquet do último reports/baseline_ml/<tag>/)
# Com limiares (se encontrados), marcadores de janelas positivas e split=test.
# Salva em ./results/timeline/<timestamp>/ por arquivo (PNG).
# -----------------------------------------------------------------------------------

from __future__ import annotations
import os, glob, json, time
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS_MF     = os.path.join(ROOT, "reports", "mf_baseline")
REPORTS_ML     = os.path.join(ROOT, "reports", "baseline_ml")
OUT_ROOT       = os.path.join(ROOT, "results", "timeline")


def _latest_dir(base: str, pattern: str="*") -> Optional[str]:
    cand = sorted(glob.glob(os.path.join(base, pattern)))
    return cand[-1] if cand else None

def _load_dataset() -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, "dataset.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset.parquet não encontrado em {path}")
    df = pd.read_parquet(path)
    # harmoniza 'split' (algumas versões usam 'subset')
    if "split" not in df.columns and "subset" in df.columns:
        df = df.rename(columns={"subset": "split"})
    need = {"file_id","start_gps","label","split"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"dataset.parquet sem colunas: {miss}")
    # trabalha só com test
    df = df[df["split"] == "test"].copy()
    return df

def _load_mf_scores() -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
    d = _latest_dir(REPORTS_MF)
    if not d:
        print("[viz] nenhum diretório em reports/mf_baseline")
        return None, None, None
    csv = os.path.join(d, "scores_test.csv")
    thr = None
    try:
        # threshold em summary.json, se existir
        s = os.path.join(d, "summary.json")
        if os.path.exists(s):
            with open(s, "r") as f: js = json.load(f)
            thr = float(js.get("threshold", None)) if js else None
    except Exception:
        pass
    if not os.path.exists(csv):
        print(f"[viz] MF: {d} sem scores_test.csv")
        return None, thr, d
    df = pd.read_csv(csv)
    # Esperado: columns=['subset','file_id','start_gps','label','score']
    # Harmoniza nome da coluna do score se necessário
    if "score" not in df.columns:
        # fallback comum
        score_col = [c for c in df.columns if c.lower().startswith("score")][0]
        df = df.rename(columns={score_col: "score"})
    keep = {"file_id","start_gps","score"}
    df = df[list(keep)].copy()
    df["score_mf"] = df.pop("score").astype(float)
    return df, thr, d

def _load_ml_scores() -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
    d = _latest_dir(REPORTS_ML)
    if not d:
        print("[viz] nenhum diretório em reports/baseline_ml")
        return None, None, None
    pq = os.path.join(d, "scores_test.parquet")
    thr = None
    try:
        tjson = os.path.join(d, "thresholds.json")
        if os.path.exists(tjson):
            with open(tjson, "r") as f: js = json.load(f)
            thr = float(js.get("threshold_used", None))
        else:
            mjson = os.path.join(d, "metrics.json")
            if os.path.exists(mjson):
                with open(mjson, "r") as f: ms = json.load(f)
                thr = float(ms.get("chosen_threshold", None))
    except Exception:
        pass
    if not os.path.exists(pq):
        print(f"[viz] ML: {d} sem scores_test.parquet")
        return None, thr, d
    df = pd.read_parquet(pq)
    # Esperado: tem colunas dumpadas + 'score'
    if "score" not in df.columns:
        # fallback: algumas versões podem usar 'proba' / 'pred_score'
        cand = [c for c in df.columns if c.lower() in ("score","proba","pred_score")]
        if not cand:
            raise KeyError("scores_test.parquet sem coluna de score.")
        df = df.rename(columns={cand[0]: "score"})
    keep = {"file_id","start_gps","score"}
    df = df[list(keep)].copy()
    df["score_ml"] = df.pop("score").astype(float)
    return df, thr, d

def _join_scores(df_base: pd.DataFrame,
                 df_mf: Optional[pd.DataFrame],
                 df_ml: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df_base.copy()
    if df_mf is not None:
        df = df.merge(df_mf, on=["file_id","start_gps"], how="left")
    if df_ml is not None:
        df = df.merge(df_ml, on=["file_id","start_gps"], how="left")
    # ordena temporalmente por file_id, depois start_gps
    df = df.sort_values(["file_id","start_gps"]).reset_index(drop=True)
    return df

def _pick_top_files(df: pd.DataFrame, k: int = 12) -> list:
    """
    Estratégia:
      1) prioriza files com positivos (label=1) no 'test'
      2) completa com maiores picos de score (MF ou ML)
    """
    pos_files = df.loc[df["label"] == 1, "file_id"].unique().tolist()
    # ranking por pico de max(score_mf, score_ml)
    score_cols = [c for c in ["score_mf","score_ml"] if c in df.columns]
    if not score_cols:
        score_cols = []
    df["_max_score"] = df[score_cols].max(axis=1, skipna=True) if score_cols else 0.0
    top_by_score = (df.groupby("file_id")["_max_score"].max()
                      .sort_values(ascending=False).index.tolist())
    seen, out = set(), []
    for fid in pos_files + top_by_score:
        if fid not in seen:
            out.append(fid); seen.add(fid)
        if len(out) >= k: break
    if not out:
        out = df["file_id"].unique().tolist()[:k]
    return out

def _format_ts(gps: float) -> str:
    # GPS puro é ok, mas dá um arredondado pra legibilidade
    return f"{gps:.2f}"

def _plot_file(df_file: pd.DataFrame,
               thr_mf: Optional[float],
               thr_ml: Optional[float],
               out_dir: str,
               fid: str,
               tag_info: Dict[str,str]):
    if df_file.empty:
        return
    x = df_file["start_gps"].to_numpy()
    y_mf = df_file["score_mf"].to_numpy() if "score_mf" in df_file.columns else None
    y_ml = df_file["score_ml"].to_numpy() if "score_ml" in df_file.columns else None
    y_lab = df_file["label"].to_numpy().astype(int)

    plt.figure(figsize=(11, 4.8))
    # Plots
    if y_mf is not None:
        plt.plot(x, y_mf, color="#2563eb", lw=1.3, label="MF score (NCC)")
    if y_ml is not None:
        plt.plot(x, y_ml, color="#16a34a", lw=1.3, label="ML score (proba)")
    # Limiar
    if (thr_mf is not None) and (y_mf is not None):
        plt.axhline(thr_mf, color="#3b82f6", lw=1.0, ls="--", alpha=0.7, label=f"thr MF={thr_mf:.4g}")
    if (thr_ml is not None) and (y_ml is not None):
        plt.axhline(thr_ml, color="#22c55e", lw=1.0, ls="--", alpha=0.7, label=f"thr ML={thr_ml:.4g}")
    # Marcadores de positivos
    pos_mask = y_lab == 1
    if np.any(pos_mask):
        yref = np.nanmax(
            np.vstack([a for a in [y_mf, y_ml] if a is not None]),
            axis=0
        ) if ((y_mf is not None) or (y_ml is not None)) else np.zeros_like(x)
        ymark = np.nanmax(yref) * 1.02 if np.isfinite(np.nanmax(yref)) else 1.0
        plt.vlines(x[pos_mask], ymin=0, ymax=ymark, colors="#ef4444", linestyles=":", lw=1.0, label="label=1")
        # justifica legenda única
        # (se for muita linha, não repete label)
        # ok do jeito simples — matplotlib cuida

    plt.xlabel("start_gps (s)")
    plt.ylabel("score")
    subtitle = []
    if tag_info.get("mf"):
        subtitle.append(f"MF: {tag_info['mf']}")
    if tag_info.get("ml"):
        subtitle.append(f"ML: {tag_info['ml']}")
    st = " | ".join(subtitle) if subtitle else ""
    plt.title(f"{fid}  {st}")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    safe_name = "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in fid)
    out = os.path.join(out_dir, f"{safe_name}.png")
    plt.savefig(out, dpi=160)
    plt.close()

def main():
    t0 = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_dir = os.path.join(OUT_ROOT, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs(out_dir, exist_ok=True)

    print("[viz] carregando dataset test …")
    df_base = _load_dataset()   # file_id, start_gps, label, split=test

    print("[viz] carregando MF (último run) …")
    df_mf, thr_mf, tag_mf = _load_mf_scores()
    print(f"      MF: thr={thr_mf} | tag={tag_mf}")

    print("[viz] carregando ML (último run) …")
    df_ml, thr_ml, tag_ml = _load_ml_scores()
    print(f"      ML: thr={thr_ml} | tag={tag_ml}")

    print("[viz] unindo …")
    df = _join_scores(df_base, df_mf, df_ml)

    print("[viz] escolhendo arquivos-alvo …")
    file_list = _pick_top_files(df, k=12)
    print(f"      {len(file_list)} file_id(s) selecionados")

    # gera um índice rápido para cada file_id
    for fid in file_list:
        dff = df[df["file_id"] == fid].copy()
        _plot_file(
            dff, thr_mf, thr_ml, out_dir, fid,
            tag_info={"mf": os.path.basename(tag_mf) if tag_mf else "",
                      "ml": os.path.basename(tag_ml) if tag_ml else ""}
        )

    # também gera um gráfico geral com distribuição de scores (opcional, leve)
    try:
        sc_cols = [c for c in ["score_mf","score_ml"] if c in df.columns]
        if sc_cols:
            plt.figure(figsize=(8,4))
            for c, col in zip(["#2563eb","#16a34a"], sc_cols):
                plt.hist(df[col].dropna().values, bins=80, alpha=0.5, color=c, label=col, density=True)
            plt.title("Distribuição de scores (TEST)")
            plt.xlabel("score"); plt.ylabel("densidade"); plt.grid(alpha=0.25); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "_scores_hist.png"), dpi=160); plt.close()
    except Exception:
        pass

    print(f"[viz] concluído em {time.time()-t0:.1f}s")
    print(f"[viz] saída: {out_dir}")

if __name__ == "__main__":
    main()