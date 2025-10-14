
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = ROOT
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from viz.visualization_3d import Wave3DVisualizer

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

OUTDIR   = os.path.join(os.path.dirname(ROOT), "results", "viz3d")
FIGSIZE  = (960, 720)
COLOR    = (0.5, 0.8, 1.0)
TUBE_R   = 0.012
FPS      = 30
FRAMES   = 180
SPIN_DEG = 360
CAMERA   = {"azimuth": 45.0, "elevation": 70.0, "distance": None}

def add_noise_by_snr(signal, snr, seed=123):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=signal.size)
    sig_rms = np.sqrt(np.mean(signal**2) + 1e-12)
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-12)
    return signal + noise * (sig_rms / snr)

def chirp(fs=4096.0, dur=1.2, f0=30.0, f1=220.0, amp=1.0, snr=10):
    t = np.arange(0, dur, 1.0/fs)
    k = (f1 - f0) / max(dur, 1e-9)
    w = amp * np.sin(2*np.pi*(f0*t + 0.5*k*t**2)) * np.hanning(t.size)
    return add_noise_by_snr(w, snr), fs

def ringdown(fs=4096.0, dur=1.5, f=200.0, tau=0.12, amp=1.0, snr=8):
    t = np.arange(0, dur, 1.0/fs)
    env = np.exp(-t / max(tau, 1e-9))
    w = amp * env * np.sin(2*np.pi*f*t) * np.hanning(t.size)
    return add_noise_by_snr(w, snr), fs

def sinegauss(fs=4096.0, dur=1.0, f0=150.0, q=9.0, amp=1.0, snr=12):
    t = np.arange(0, dur, 1.0/fs)
    sigma = q/(2*np.pi*f0)
    env = np.exp(-0.5*((t - dur/2)/max(sigma, 1e-9))**2)
    w = amp * env * np.sin(2*np.pi*f0*t)
    return add_noise_by_snr(w, snr), fs

def movie_pyvista(w, fs, outfile):
    import pyvista as pv
    pv.global_theme.background = "black"
    pv.global_theme.window_size = FIGSIZE

    t = np.arange(w.size)/fs
    pts = np.c_[t, w, np.zeros_like(w)]
    poly = pv.Spline(pts, n_points=len(pts))
    tube = poly.tube(radius=max(TUBE_R, 1e-4))

    p = pv.Plotter(off_screen=True)
    p.add_mesh(tube, color=COLOR, smooth_shading=True)

    # plano
    xmin, xmax = float(t.min()), float(t.max())
    ymin, ymax = float(w.min()), float(w.max())
    dx = 0.02*(xmax - xmin + 1e-9)
    dy = 0.10*max(abs(ymin), abs(ymax), 1e-6)
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy
    plane = pv.Plane(center=((xmin+xmax)/2, (ymin+ymax)/2, 0), direction=(0,0,1),
                     i_size=(xmax-xmin), j_size=(ymax-ymin), i_resolution=20, j_resolution=10)
    p.add_mesh(plane, color=(0.9,0.9,0.9), opacity=0.25, show_edges=True, edge_color=(0.2,0.2,0.2))
    p.add_text("GW 3D", color="white", font_size=12)

    if CAMERA.get("azimuth"):   p.camera.azimuth(float(CAMERA["azimuth"]))
    if CAMERA.get("elevation"): p.camera.elevation(float(CAMERA["elevation"]))

    os.makedirs(OUTDIR, exist_ok=True)
    p.open_movie(outfile, framerate=FPS)
    p.show(auto_close=False)
    step = float(SPIN_DEG)/max(1, FRAMES)
    for _ in range(FRAMES):
        try: p.camera.azimuth(step)
        except Exception:
            try: p.camera.Azimuth(step)
            except Exception: pass
        p.render()
        p.write_frame()
    p.close()
    return outfile

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = [
        ("chirp", *chirp()),
        ("ringdown", *ringdown()),
        ("sinegauss", *sinegauss()),
    ]
    for name, w, fs in jobs:
        out = os.path.join(OUTDIR, f"{name}.mp4")
        try:
            res = movie_pyvista(w, fs, out)
            print(f"[{name}] OK {res}")
        except Exception as e:
            print(f"[{name}] erro {e}")

if __name__ == "__main__":
    main()
