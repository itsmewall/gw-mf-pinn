import numpy as np
from src.accel import ncc_fft

def test_peak_shift():
    fs = 2048
    n = 8192
    lag = 13
    a = np.random.randn(n)
    b = np.roll(a, lag) + 0.001*np.random.randn(n)
    corr = ncc_fft.correlate_all_lags(a, b, 64)
    idx = np.argmax(corr)
    est_lag = idx - 64
    assert abs(est_lag - lag) <= 1
