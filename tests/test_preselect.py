# tests/test_preselect.py
import numpy as np
from eval.mf_baseline import _preselect_templates, CFG
from physics import gwphys_cuda as m  # garante import

def test_preselect():
    CFG.PRESELECT_ENABLE = True
    CFG.PRESELECT_TOPK_TEMPL = 8
    X = np.random.randn(32, 4096).astype(np.float32)
    bank = {f"k{i}": np.random.randn(4096).astype(np.float32) for i in range(128)}
    idx = _preselect_templates(X, bank, CFG.PRESELECT_TOPK_TEMPL)
    assert len(idx) == CFG.PRESELECT_TOPK_TEMPL
    print("ok. topk =", len(idx))

if __name__ == "__main__":
    test_preselect()