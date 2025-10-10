import os, re, math, glob
from collections import defaultdict

ROOT = "data/raw"
pat = re.compile(r".*([HLV]1).*?(\d+KHZ).*?-(\d+)\.hdf5$", re.I)

groups = defaultdict(list)
total = 0
for p in glob.glob(os.path.join(ROOT, "*.hdf5")):
    sz = os.path.getsize(p)
    total += sz
    m = pat.match(os.path.basename(p))
    if m:
        det, rate, dur = m.group(1).upper(), m.group(2).upper(), m.group(3)
        key = (det, rate, dur)
    else:
        key = ("UNKNOWN", "UNKNOWN", "UNKNOWN")
    groups[key].append((p, sz))

print(f"Arquivos: {sum(len(v) for v in groups.values())} | Tamanho total: {total/1e9:.2f} GB\n")
for key, files in sorted(groups.items()):
    det, rate, dur = key
    subtotal = sum(sz for _, sz in files)
    print(f"{det:>3} | {rate:>5} | {dur:>6}s  => {len(files)} arquivos, {subtotal/1e9:.2f} GB")
