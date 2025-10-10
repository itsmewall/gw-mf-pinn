# debug_win.py (adicione na raiz e rode: python debug_win.py)
import os, h5py, numpy as np
from fractions import Fraction
from glob import glob

FS_TARGET = 4096
WINDOW_SEC = 2.0

# Pegue um dos IDs do quick_probe
FID = "H-H1_GWOSC_O4a_4KHZ_R1-1388625920-4096_windows.hdf5"
WIN_PATH = glob(f"data/processed/**/{FID}", recursive=True)[0]

def read_attrs(obj):
    out = {}
    try:
        for k, v in obj.attrs.items():
            if isinstance(v, (np.integer, np.floating)): v = v.item()
            out[str(k)] = v
    except Exception:
        pass
    return out

with h5py.File(WIN_PATH, "r") as f:
    starts = f["/windows/start_idx"][()]
    start = int(starts[len(starts)//2])  # pegue uma janela do meio
    meta = read_attrs(f["/meta_in"]) if "/meta_in" in f else {}
    src = meta.get("source_path")
    assert src, "meta_in.source_path ausente"
    src_abs = os.path.join(os.path.dirname(WIN_PATH), src)
    if not os.path.exists(src_abs):
        src_abs = os.path.join("data/interim", os.path.basename(src))
    print("whitened:", src_abs)

with h5py.File(src_abs, "r") as g:
    cand = None
    for c in ["/strain/StrainWhitened","/strain/whitened","/data/whitened","/whitened","/strain/Strain"]:
        if c in g and isinstance(g[c], h5py.Dataset):
            cand = c; break
    assert cand, "dataset whitened não encontrado no HDF5"
    dset = g[cand]
    attrs = read_attrs(dset) | read_attrs(g)
    fs_in = attrs.get("fs")
    if not fs_in:
        xs = attrs.get("xspacing") or attrs.get("Xspacing")
        assert xs, "impossível inferir fs (xspacing ausente)"
        fs_in = 1.0/float(xs)
    fs_in = int(round(float(fs_in)))
    wlen_in = int(round(WINDOW_SEC * fs_in))
    sl = dset[start:start+wlen_in].astype(np.float32)
    print("len_in:", sl.size, "min:", sl.min(), "max:", sl.max(), "std:", sl.std())

# reamostra se necessário
if fs_in != FS_TARGET:
    import scipy.signal as sig
    frac = Fraction(FS_TARGET, fs_in).limit_denominator(64)
    sl = sig.resample_poly(sl, frac.numerator, frac.denominator).astype(np.float32)
    # pad/crop
    wlen = int(round(WINDOW_SEC * FS_TARGET))
    if sl.size > wlen: sl = sl[-wlen:]
    elif sl.size < wlen:
        pad = np.zeros(wlen, np.float32); pad[-sl.size:] = sl; sl = pad

sl = sl - sl.mean()
norm = float(np.linalg.norm(sl))
print("len_target:", sl.size, "| norm:", norm, "| std:", sl.std(), "| any!=0:", bool(np.any(sl!=0)))
