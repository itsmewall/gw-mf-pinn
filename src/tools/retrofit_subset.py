# src/tools/retrofit_subset.py
import os, json, hashlib
import pandas as pd
from pathlib import Path

PARQUET = Path("data/processed/dataset.parquet")
META    = Path("data/processed/dataset_meta.json")  # se existir, usamos val/test ratio daqui

assert PARQUET.exists(), f"Parquet não encontrado: {PARQUET}"

df = pd.read_parquet(PARQUET)

# Se já tem subset, só imprime e sai
if "subset" in df.columns:
    print("[OK] dataset.parquet já tem coluna 'subset'. Nada a fazer.")
    print(df["subset"].value_counts(dropna=False))
    raise SystemExit(0)

# Tenta fontes alternativas
subset = None
if "split" in df.columns:
    # alguns previews antigos usavam 'split'
    subset = df["split"].astype(str)
elif {"is_val","is_test"}.issubset(df.columns):
    subset = df["is_val"].map({True:"val", False:"train"})
    subset[df["is_test"] == True] = "test"

if subset is None:
    # Reconstrução determinística via hash, respeitando proporções do meta (se existir)
    val_ratio, test_ratio = 0.15, 0.15
    if META.exists():
        try:
            meta = json.loads(META.read_text())
            val_ratio  = float(meta.get("val_size", 0.15))
            test_ratio = float(meta.get("test_size", 0.15))
        except Exception:
            pass

    def _bucket(row):
        # string estável
        key = f"{row['file_id']}|{row['start_gps']}"
        h = hashlib.sha1(key.encode("utf-8")).digest()
        # 0..1
        u = int.from_bytes(h[:8], "big") / 2**64
        if u < val_ratio:            return "val"
        elif u < val_ratio+test_ratio:return "test"
        else:                        return "train"

    subset = df.apply(_bucket, axis=1)

df["subset"] = subset.astype("category")

# backup e sobrescreve
bak = PARQUET.with_suffix(".parquet.bak")
if not bak.exists():
    PARQUET.replace(bak)
df.to_parquet(PARQUET)
print("[OK] 'subset' criado e parquet atualizado.")
print(df["subset"].value_counts(dropna=False))
print(f"Backup salvo em: {bak}")
