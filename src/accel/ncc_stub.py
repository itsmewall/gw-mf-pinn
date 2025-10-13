import numpy as np

def correlate_all_lags(a: np.ndarray, b: np.ndarray, max_shift: int) -> np.ndarray:
    """
    Fallback puro-Python (O(N * max_shift)). a e b com mesmo tamanho.
    Retorna vetor de correlação normalizada para lags [-max_shift..+max_shift].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.ndim == 1 and b.ndim == 1 and a.size == b.size
    n = a.size
    am = a.mean(); asd = a.std() + 1e-12
    bm = b.mean(); bsd = b.std() + 1e-12
    a0 = (a - am) / asd
    b0 = (b - bm) / bsd

    lags = np.arange(-max_shift, max_shift + 1, dtype=int)
    out = np.zeros_like(lags, dtype=np.float64)
    for i, k in enumerate(lags):
        if k >= 0:
            # a alinhado com b deslocado +k
            lo = 0; hi = n - k
            out[i] = np.dot(a0[lo:hi], b0[lo + k:hi + k]) / (hi - lo)
        else:
            k2 = -k
            lo = 0; hi = n - k2
            out[i] = np.dot(a0[lo + k2:hi + k2], b0[lo:hi]) / (hi - lo)
    return out