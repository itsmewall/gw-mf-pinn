all: setup 
setup: 
tpython -m venv .venv 
tpip install -r requirements.txt 

.PHONY: build_accel test_accel

build_accel:
	. venv/bin/activate && python - <<'PY'
from src.accel import ncc_fft
import numpy as np
a = np.random.randn(4096)
b = np.roll(a, 7) + 0.01*np.random.randn(4096)
res = ncc_fft.correlate_all_lags(a, b, 16)
print("OK accel:", float(res.max()))
PY

test_accel:
	. venv/bin/activate && pytest -q tests/test_accel.py -q
