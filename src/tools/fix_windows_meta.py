# src/tools/fix_windows_meta.py
import os, h5py, json
from glob import glob

WIN_DIR = "data/processed"    # onde estão os *_windows.hdf5
INTERIM_DIR = "data/interim"  # onde estão os *_whitened.hdf5

def guess_whitened_path(win_path: str) -> str | None:
    bn = os.path.basename(win_path).replace("_windows.hdf5", "_whitened.hdf5")
    # 1) mesmo diretório (caso tenha copiado)
    cand1 = os.path.join(os.path.dirname(win_path), bn)
    # 2) data/interim (layout padrão da pipeline)
    cand2 = os.path.join(INTERIM_DIR, bn)
    for c in (cand1, cand2):
        if os.path.exists(c):
            return os.path.relpath(c, start=os.path.dirname(win_path))
    return None

def ensure_meta(win_path: str) -> bool:
    changed = False
    with h5py.File(win_path, "a") as f:
        # garanta /meta_in
        if "meta_in" not in f:
            f.create_group("meta_in")
            changed = True
        m = f["/meta_in"]

        # garanta source_path
        if "source_path" not in m.attrs:
            rel = guess_whitened_path(win_path)
            if rel:
                m.attrs["source_path"] = rel
                changed = True

        # garanta fs (fallback para attrs de topo)
        if "fs" not in m.attrs:
            fs = f.attrs.get("fs", None)
            if fs is not None:
                m.attrs["fs"] = float(fs); changed = True

        # se não tiver xstart/xspacing, ok — baseline consegue inferir fs pela dset do whitened

    return changed

def main():
    wins = glob(os.path.join(WIN_DIR, "**/*_windows.hdf5"), recursive=True)
    touched = 0
    for w in wins:
        try:
            if ensure_meta(w):
                touched += 1
                print("[fixed]", w)
        except Exception as e:
            print("[skip ]", w, "->", e)
    print(f"Pronto. Corrigidos {touched}/{len(wins)} arquivos.")

if __name__ == "__main__":
    main()
