import os
import sys
import json
import numpy as np
import h5py

DIR = sys.argv[1] if len(sys.argv) > 1 else "data/processed/windows"

def inspect_file(path):
    print(f"\n=== Inspecting {path} ===")
    try:
        with h5py.File(path, "r") as f:
            print("Root keys:", list(f.keys()))
            
            # tenta localizar dset de janelas
            if "windows" in f:
                dset = f["windows"]
            elif "/data/windows" in f:
                dset = f["/data/windows"]
            else:
                print("datasets disponíveis:", list(f.keys()))
                print("Não encontrado dataset 'windows'")
                return

            fs = float(f.attrs.get("fs") or dset.attrs.get("fs") or (1.0 / float(dset.attrs.get("dt", 1.0))))
            win_sec = float(f.attrs.get("window_sec") or dset.attrs.get("window_sec") or (dset.shape[1] / fs))
            whitened = bool(f.attrs.get("whitened", True))
            det = str(f.attrs.get("detector") or dset.attrs.get("detector") or "H1")

            print(f"[INFO] shape={dset.shape} dtype={dset.dtype} fs={fs}Hz window_sec={win_sec} whitened={whitened} det={det}")

            idxs = [0, dset.shape[0] // 2, dset.shape[0] - 1]
            for i in idxs:
                h = np.array(dset[i, :], dtype=float)
                nz = np.count_nonzero(h)
                stats = dict(i=int(i), min=float(np.min(h)), max=float(np.max(h)),
                             mean=float(np.mean(h)), std=float(np.std(h)), nonzero=int(nz))
                print("[WIN]", json.dumps(stats))

            h0 = np.array(dset[0, :], dtype=float)
            print("[HEAD]", h0[:32].round(6).tolist())
    except Exception as e:
        print(f"Erro ao abrir {path}: {e}")

def main():
    print(f"Starting inspection in directory: {DIR}")
    count_files = 0
    for root, _, files in os.walk(DIR):
        for file in files:
            if file.endswith(".hdf5") or file.endswith(".h5"):
                path = os.path.join(root, file)
                print(f"Found file: {path}")
                inspect_file(path)
                count_files += 1
    if count_files == 0:
        print("Nenhum arquivo .hdf5 ou .h5 encontrado no diretório.")

if __name__ == "__main__":
    main()
