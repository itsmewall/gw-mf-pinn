# src/exp/mf2_long_train.py
import time, os, json
from src.training import train_mf
from src.utils.prof import log_step

def run_long_train():
    print("[MF2/LONG] treino longo iniciando")
    out = train_mf.train(hparams={
        "SEED": 2025,
        "EPOCHS": 10,
        "BATCH": 160,
        "NUM_WORKERS": 10,
        "PIN_MEMORY": True,
        "PREFETCH": 8,
        "PERSISTENT": True,
        "DEVICE": "cuda",
        "AMP": True,
        "COMPILE": True,
        "CHECKPOINTING": True,
        "EVAL_EVERY": 2
    })
    os.makedirs("results/mf2_long", exist_ok=True)
    path = os.path.join("results", "mf2_long", f"long_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("[MF2/LONG] concluido. resumo em", path)

def main():
    with log_step("mf2_long_train_inner", tag="mf2_weekend"):
        run_long_train()

if __name__ == "__main__":
    with log_step("mf2_long_train", tag="mf2_weekend"):
        main()