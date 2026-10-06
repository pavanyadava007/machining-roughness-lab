"""Window features from 10 kHz force and vibration signals.

Pure numpy/scipy functions so the tests can run on synthetic signals.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from mrl.config import BANDS_HZ, FS_HZ


def time_features(x: np.ndarray) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x**2)))
    ac = x - x.mean()
    lo, hi = np.percentile(x, [0.5, 99.5])
    return {
        "mean": float(x.mean()),
        "std": float(ac.std()),
        "rms": rms,
        "p2p": float(hi - lo),
        "skew": float(stats.skew(ac)),
        "kurt": float(stats.kurtosis(ac)),
        "crest": float(np.max(np.abs(ac)) / (ac.std() + 1e-12)),
    }


def spectral_features(x: np.ndarray, spindle_hz: float, fs: int = FS_HZ) -> dict[str, float]:
    """Band energy shares, spectral centroid and spindle-harmonic amplitudes (Welch PSD)."""
    f, pxx = signal.welch(x - x.mean(), fs=fs, nperseg=min(len(x), fs // 2), detrend="linear")
    total = float(np.trapezoid(pxx, f)) + 1e-12
    out: dict[str, float] = {}
    for lo, hi in BANDS_HZ:
        m = (f >= lo) & (f < hi)
        out[f"band_{lo}_{hi}"] = float(np.trapezoid(pxx[m], f[m]) / total) if m.sum() > 1 else 0.0
    out["centroid_hz"] = float(np.sum(f * pxx) / (np.sum(pxx) + 1e-12))
    out["log_power"] = float(np.log10(total))
    df = f[1] - f[0]
    for k in (1, 2):
        m = np.abs(f - k * spindle_hz) <= max(df, 1.0)
        out[f"spindle_h{k}"] = float(np.sqrt(pxx[m].max() * df)) if m.any() else 0.0
    return out


def window_features(
    forces: np.ndarray, acc: np.ndarray, speed_rpm: float, feed: float, depth: float, fs: int = FS_HZ
) -> dict[str, float]:
    """Features of one window. forces: (n, 3) Fx Fy Fz in N; acc: (n, 3) in g."""
    spindle_hz = speed_rpm / 60.0
    feats: dict[str, float] = {}
    fres = np.linalg.norm(forces, axis=1)
    chans = {"Fx": forces[:, 0], "Fy": forces[:, 1], "Fz": forces[:, 2], "Fres": fres,
             "Ax": acc[:, 0], "Ay": acc[:, 1], "Az": acc[:, 2]}
    for name, x in chans.items():
        for k, v in time_features(x).items():
            feats[f"{name}_{k}"] = v
        for k, v in spectral_features(x, spindle_hz, fs).items():
            feats[f"{name}_{k}"] = v
    # resultant force over the nominal chip cross-section f*d (N/mm^2); a specific-cutting-pressure proxy
    feats["kc_res"] = float(fres.mean() / (feed * depth))
    feats["force_ratio_xz"] = float(abs(forces[:, 0].mean()) / (abs(forces[:, 2].mean()) + 1e-9))
    feats["force_ratio_yz"] = float(abs(forces[:, 1].mean()) / (abs(forces[:, 2].mean()) + 1e-9))
    return feats


def iter_windows(n: int, win: int):
    for i in range(n // win):
        yield i, slice(i * win, (i + 1) * win)


def kinematic_ra_um(feed_mm_rev: float, nose_radius_mm: float) -> float:
    """Ideal geometric roughness of turning, Ra ~ f^2 / (32 r) (mm -> um)."""
    return feed_mm_rev**2 / (32.0 * nose_radius_mm) * 1000.0
