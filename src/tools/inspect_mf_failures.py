# src/tools/inspect_mf_failures.py
import os, sys, pandas as pd

root = "reports/mf_baseline"
tag  = sys.argv[1] if len(sys.argv) > 1 else sorted(os.listdir(root))[-1]
dbg  = os.path.join(root, tag, "_debug", "fail_rows.csv")

print(f"[info] lendo: {dbg}")
df = pd.read_csv(dbg)

print(df.head(5))

print("\n[contagem por reason]")
print(df["reason"].value_counts())

print("\n[amostras por reason]")
need = [c for c in ["file_id","start_gps","whitened_path","exc_type","exc_msg","msg"] if c in df.columns]
for r in df["reason"].value_counts().index:
    print(f"\n=== {r} ===")
    print(df[df["reason"]==r].head(3)[need])
