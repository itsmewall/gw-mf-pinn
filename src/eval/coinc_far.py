# src/eval/coinc_far.py
from __future__ import annotations
import numpy as np
import pandas as pd

def coincidence_and_far(scores_df: pd.DataFrame,
                        time_slides: int = 200,
                        max_dt: float = 0.010) -> dict:
    """
    scores_df com colunas: detector in {H1,L1}, start_gps, score
    max_dt em segundos. Retorna thresholds por FAR.
    """
    # 1) junta por tempo
    h1 = scores_df[scores_df.detector == "H1"].copy()
    l1 = scores_df[scores_df.detector == "L1"].copy()
    h1 = h1.sort_values("start_gps")
    l1 = l1.sort_values("start_gps")

    # 2) coincidências verdadeiras
    out = []
    j = 0
    for _, r in h1.iterrows():
        t = r.start_gps
        # move janela em L1
        while j < len(l1) and l1.iloc[j].start_gps < t - max_dt:
            j += 1
        k = j
        while k < len(l1) and l1.iloc[k].start_gps <= t + max_dt:
            sc = np.hypot(r.score, l1.iloc[k].score)  # combinação quadrática
            out.append((t, sc))
            k += 1
    coinc = pd.DataFrame(out, columns=["t", "score_coinc"]).sort_values("score_coinc", ascending=False)

    # 3) background por time-slides
    rng = np.random.default_rng(42)
    bg_scores = []
    if len(l1) and len(h1):
        dt = float(l1.start_gps.max() - l1.start_gps.min() + 100.0)
        for _ in range(time_slides):
            shift = float(rng.uniform(0.1, dt))
            l1_shift = l1.copy()
            l1_shift.start_gps = l1_shift.start_gps + shift
            # repete coincidência
            j = 0
            for _, r in h1.iterrows():
                t = r.start_gps
                while j < len(l1_shift) and l1_shift.iloc[j].start_gps < t - max_dt:
                    j += 1
                k = j
                while k < len(l1_shift) and l1_shift.iloc[k].start_gps <= t + max_dt:
                    bg_scores.append(np.hypot(r.score, l1_shift.iloc[k].score))
                    k += 1
    bg_scores = np.array(bg_scores, dtype=float)
    bg_scores.sort()
    def thr_at_far(far):
        if len(bg_scores) == 0:
            return np.inf
        idx = int(np.ceil((1.0 - far) * len(bg_scores))) - 1
        idx = max(0, min(idx, len(bg_scores) - 1))
        return float(bg_scores[idx])

    return {
        "coinc_top": coinc.head(20).to_dict(orient="records"),
        "thr_far_1e_4": thr_at_far(1e-4),
        "thr_far_1e_5": thr_at_far(1e-5),
        "thr_far_1e_6": thr_at_far(1e-6),
        "bg_count": int(len(bg_scores))
    }
