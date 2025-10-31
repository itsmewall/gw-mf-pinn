# src/exp/mf2_multi_seed.py
import os, time, json
from src.training import train_mf
from src.utils.prof import log_step

SEEDS = [2025, 2026, 2027]

def main():
    runs = []
    for s in SEEDS:
        print(f"[MF2/MULTI] seed={s}")
        out = train_mf.train(hparams={
            "SEED": s,
            "EPOCHS": 6,
            "BATCH": 160,
            "NUM_WORKERS": 8,
            "DEVICE": "cuda",
            "AMP": True,
            "COMPILE": True,
            "EVAL_EVERY": 3
        })
        runs.append(out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("results/mf2_multi", exist_ok=True)
    with open(f"results/mf2_multi/summary_{stamp}.json", "w") as f:
        json.dump(runs, f, indent=2)
    print("ok, multi seed salvo em results/mf2_multi/")

if __name__ == "__main__":
    with log_step("mf2_multi_seed", tag="mf2_weekend"):
        main()