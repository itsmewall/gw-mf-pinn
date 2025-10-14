# src/viz/spectro_quick.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

def save_quick_spectrogram(hdf5_path: str, out_png: str,
                           gps0: float, dur: float = 4.0,
                           fmax: float = 512.0):
    ts = TimeSeries.read(hdf5_path, format='hdf5', path='strain/Strain')
    seg = ts.crop(gps0, gps0 + dur)
    q = seg.q_transform(outseg=(gps0, gps0 + dur))
    plot = Plot(q, yscale='log', ylim=(20, fmax))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plot.save(out_png)
