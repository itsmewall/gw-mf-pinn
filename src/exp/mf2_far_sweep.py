# src/exp/mf2_far_sweep.py
import os, json, time
import numpy as np
import pandas as pd

from src.utils.prof import log_step

ROOT = os.path.dirname(os.path.dirname(__file__))  # repo
REPORTS = os.path.join(ROOT, "reports", "mf_stage2")
OUTROOT = os.path.join(ROOT, "results", "mf2_far")

def _latest_report(base):
    if not os.path.isdir(base):
        raise RuntimeError("sem reports/mf_stage2")
    subs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    subs = [s for s in subs if os.path.exists(os.path.join(s, "scores_val.csv"))]
    if not subs:
        raise RuntimeError("nenhum scores_val.csv em reports/mf_stage2")
    return max(subs, key=os.path.getmtime)

def main():
    os.makedirs(OUTROOT, exist_ok=True)
    rep = _latest_report(REPORTS)
    val = pd.read_csv(os.path.join(rep, "scores_val.csv"))
    tst = pd.read_csv(os.path.join(rep, "scores_test.csv"))

    targets = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    rows = []

    yv = val["label"].values.astype(int)
    sv = val["score"].values.astype(float)
    yt = tst["label"].values.astype(int)
    st = tst["score"].values.astype(float)

    order = np.argsort(-sv)
    sv_sorted = sv[order]
    yv_sorted = yv[order]

    for tgt in targets:
        # escolhe threshold que entrega FPR <= tgt
        n0 = (yv == 0).sum()
        best_thr = 1.0
        best = dict(fpr=0.0, tpr=0.0, prec=0.0, rec=0.0)
        fp = 0
        tp = 0
        for i in range(len(sv_sorted)):
            if yv_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            fpr = fp / max(n0, 1)
            tpr = tp / max((yv == 1).sum(), 1)
            if fpr <= tgt:
                best_thr = sv_sorted[i]
                prec = tp / max(tp + fp, 1)
                best = dict(fpr=fpr, tpr=tpr, prec=prec, rec=tpr)
                break

        # aplica no test
        pred_t = (st >= best_thr).astype(int)
        tn = int(((yt == 0) & (pred_t == 0)).sum())
        fp = int(((yt == 0) & (pred_t == 1)).sum())
        fn = int(((yt == 1) & (pred_t == 0)).sum())
        tp = int(((yt == 1) & (pred_t == 1)).sum())
        prec_t = tp / max(tp + fp, 1)
        rec_t = tp / max(tp + fn, 1)
        fpr_t = fp / max(tn + fp, 1)
        rows.append(dict(
            target_fpr=tgt,
            thr=float(best_thr),
            val_fpr=float(best["fpr"]),
            val_tpr=float(best["tpr"]),
            val_prec=float(best["prec"]),
            test_tn=tn, test_fp=fp, test_fn=fn, test_tp=tp,
            test_prec=float(prec_t),
            test_rec=float(rec_t),
            test_fpr=float(fpr_t),
            report_dir=rep
        ))

    out = os.path.join(OUTROOT, f"far_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("FAR sweep salvo em", out)

if __name__ == "__main__":
    with log_step("mf2_far_sweep", tag="mf2_weekend"):
        main()