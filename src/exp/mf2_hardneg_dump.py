# src/exp/mf2_hardneg_dump.py
import os, time
import pandas as pd

from src.utils.prof import log_step

ROOT = os.path.dirname(os.path.dirname(__file__))
REPORTS = os.path.join(ROOT, "reports", "mf_stage2")
OUT = os.path.join(ROOT, "results", "hardneg")

def _latest_report(base):
    subs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    subs = [s for s in subs if os.path.exists(os.path.join(s, "scores_test.csv"))]
    return max(subs, key=os.path.getmtime) if subs else None

def main():
    os.makedirs(OUT, exist_ok=True)
    rep = _latest_report(REPORTS)
    if rep is None:
        raise RuntimeError("nenhum reports/mf_stage2 com scores_test.csv")
    df = pd.read_csv(os.path.join(rep, "scores_test.csv"))
    neg = df[df["label"] == 0].copy()
    neg = neg.sort_values("score", ascending=False)
    topk = neg.head(2000)  # pode aumentar
    out_path = os.path.join(OUT, f"hardneg_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    topk.to_csv(out_path, index=False)
    print("hard negatives salvos em", out_path)

if __name__ == "__main__":
    with log_step("mf2_hardneg", tag="mf2_weekend"):
        main()