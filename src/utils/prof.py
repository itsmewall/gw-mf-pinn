# src/utils/prof.py
import os
import time
import json

try:
    import psutil
except ImportError:
    psutil = None

from contextlib import contextmanager


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@contextmanager
def log_step(step: str, tag: str = "mf2_weekend", out_dir: str = "logs"):
    """
    Uso:
        with log_step("mf2_long_train"):
            run_long_train()
    """
    _ensure_dir(out_dir)
    t0 = time.time()

    proc = None
    if psutil is not None:
        proc = psutil.Process()
        cpu0 = proc.cpu_times()
        mem0 = proc.memory_info().rss
    else:
        cpu0 = mem0 = None

    try:
        yield
    finally:
        t1 = time.time()
        if proc is not None:
            cpu1 = proc.cpu_times()
            mem1 = proc.memory_info().rss
            cpu_user = round(cpu1.user - cpu0.user, 4)
            cpu_sys = round(cpu1.system - cpu0.system, 4)
            rss_start = mem0
            rss_end = mem1
            rss_mb_start = round(rss_start / (1024 * 1024), 2)
            rss_mb_end = round(rss_end / (1024 * 1024), 2)
        else:
            cpu_user = cpu_sys = None
            rss_start = rss_end = None
            rss_mb_start = rss_mb_end = None

        rec = {
            "tag": tag,
            "step": step,
            "wall_s": round(t1 - t0, 4),
            "cpu_user_s": cpu_user,
            "cpu_sys_s": cpu_sys,
            "rss_start": rss_start,
            "rss_end": rss_end,
            "rss_mb_start": rss_mb_start,
            "rss_mb_end": rss_mb_end,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out_path = os.path.join(out_dir, "mf2_weekend.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
