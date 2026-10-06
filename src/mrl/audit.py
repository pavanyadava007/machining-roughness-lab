"""Plausibility checks on the trial data before any model is trusted.

Each check states what physics or test practice would expect, what the data shows, and a verdict.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from mrl.config import FS_HZ, PROCESSED, RAW, RESULTS


def implied_cut_length_mm(win: pd.DataFrame) -> pd.Series:
    """Duration x feed x spindle rev/s: should be about the same for every cut on the same bar."""
    g = win.groupby("exp").agg(n=("win", "size"), speed=("speed", "first"), feed=("feed", "first"))
    return g.n * g.feed * g.speed / 60.0


def within_cut_drift(win: pd.DataFrame, col: str = "Fres_mean", frac: float = 0.1) -> pd.Series:
    def drift(x: pd.DataFrame) -> float:
        k = max(2, int(len(x) * frac))
        a, b = x[col].iloc[:k].mean(), x[col].iloc[-k:].mean()
        return float((b - a) / abs(a) * 100.0)

    return win.sort_values(["exp", "win"]).groupby("exp")[["win", col]].apply(drift)


def spindle_rpm_from_force(f_res: np.ndarray, nominal_rpm: float, fs: int = FS_HZ, tol: float = 0.3) -> float:
    """Dominant force line within +-30 % of the logged spindle frequency, returned in rpm."""
    from scipy import signal

    fr, pxx = signal.welch(f_res - f_res.mean(), fs=fs, nperseg=min(len(f_res), 8 * fs), detrend="linear")
    f0 = nominal_rpm / 60.0
    m = (fr >= f0 * (1 - tol)) & (fr <= f0 * (1 + tol))
    return float(fr[m][np.argmax(pxx[m])] * 60.0)


def spindle_table(cuts: pd.Index) -> pd.DataFrame:
    rows = []
    for e in cuts:
        path = RAW / "dyn" / f"f_{e}.csv"
        if not path.exists():
            continue
        f = pd.read_csv(path, usecols=["Fx", "Fy", "Fz"]).to_numpy()
        nominal = int(e.split("_")[0])
        rows.append({"exp": e, "nominal_rpm": nominal,
                     "spectrum_rpm": spindle_rpm_from_force(np.linalg.norm(f, axis=1), nominal)})
    return pd.DataFrame(rows).set_index("exp") if rows else pd.DataFrame()


def run(win: pd.DataFrame, lab: pd.DataFrame, spindle: pd.DataFrame | None = None) -> dict:
    cut = win.groupby("exp").agg(speed=("speed", "first"), feed=("feed", "first"), depth=("depth", "first"),
                                 Fres=("Fres_mean", "median"), Arms=("Ax_rms", "median"))
    checks = []

    length = implied_cut_length_mm(win)
    cv = float(length.std() / length.mean())
    checks.append({"check": "implied cut length (duration x f x n) is constant",
                   "expect": "same bar length on every pass, CV < 10 %",
                   "observed": f"{length.min():.0f}-{length.max():.0f} mm, CV {cv * 100:.1f} %",
                   "verdict": "pass" if cv < 0.10 else "warn"})

    odd = length[(length - length.median()).abs() / length.median() > 0.3]
    if len(odd):
        ratio = odd.median() / length.median()
        checks[-1]["observed"] += f"; {len(odd)} cuts at about {ratio:.1f}x: " + ", ".join(odd.index)

    if spindle is not None and len(spindle):
        dev = (spindle.spectrum_rpm / spindle.nominal_rpm - 1) * 100
        checks.append({"check": "logged spindle speed matches the force spectrum",
                       "expect": "spindle line within +-8 % of nominal",
                       "observed": f"deviation {dev.min():+.0f} to {dev.max():+.0f} %"
                                   + (f"; long cuts {', '.join(odd.index)} also match, so feed or pass length "
                                      "differs from the log (ask the lab)" if len(odd) else ""),
                       "verdict": "pass" if dev.abs().max() <= 8 else "warn"})

    rho = stats.spearmanr(cut.feed * cut.depth, cut.Fres)
    checks.append({"check": "mean resultant force rises with chip cross-section f x d",
                   "expect": "strong positive rank correlation (rho > 0.7)",
                   "observed": f"rho {rho.statistic:+.2f} (p {rho.pvalue:.3f}), median force "
                               f"{cut.Fres.min():.0f}-{cut.Fres.max():.0f} N",
                   "verdict": "pass" if rho.statistic > 0.7 else "fail"})

    drift = within_cut_drift(win)
    big = int((drift.abs() > 20).sum())
    checks.append({"check": "force is stationary within one cut",
                   "expect": "first vs last 10 % of windows within +-20 %",
                   "observed": f"median drift {drift.median():+.0f} %, {big}/{len(drift)} cuts beyond +-20 %",
                   "verdict": "pass" if big <= 2 else "fail"})

    by_speed = cut.groupby("speed").Arms.median()
    mono = bool(np.all(np.diff(by_speed.to_numpy()) > 0))
    checks.append({"check": "vibration level ordered by spindle speed",
                   "expect": "RMS acceleration rises with speed (same mount, same setup)",
                   "observed": ", ".join(f"{s} rpm {v:.3f} g" for s, v in by_speed.items()),
                   "verdict": "pass" if mono else "warn: speeds may be separate sessions (batch effect)"})

    g = lab.groupby("exp")
    cvs = (g.Ra.std(ddof=1) / g.Ra.mean()).sort_values(ascending=False)
    worst = cvs.head(3)
    checks.append({"check": "roughness repeatability (3 readings per cut)",
                   "expect": "CV of Ra readings below about 20 %",
                   "observed": f"median CV {cvs.median() * 100:.0f} %, worst "
                               + ", ".join(f"{e} {v * 100:.0f} %" for e, v in worst.items()),
                   "verdict": "pass" if cvs.median() < 0.2 else "warn"})

    rz = lab.Rz
    z = (rz - rz.median()) / (1.4826 * stats.median_abs_deviation(rz))
    out = lab.loc[z.abs() > 5, ["exp", "rep", "Rz"]]
    checks.append({"check": "single roughness readings without gross outliers",
                   "expect": "robust z-score of every Rz reading < 5",
                   "observed": "none" if out.empty else "; ".join(f"{r.exp} rep {r.rep}: Rz {r.Rz} um"
                                                                  for r in out.itertuples()),
                   "verdict": "pass" if out.empty else "warn"})

    return {"checks": checks, "cut_length_mm": length.round(1).to_dict(),
            "spindle_rpm_from_spectrum": ({} if spindle is None or not len(spindle)
                                          else spindle.spectrum_rpm.round(0).to_dict()),
            "force_drift_pct": drift.round(1).to_dict()}


def main() -> None:
    win = pd.read_parquet(PROCESSED / "windows.parquet")
    lab = pd.read_csv(PROCESSED / "labels.csv")
    res = run(win, lab, spindle_table(pd.Index(sorted(win.exp.unique()))))
    (RESULTS / "audit.json").write_text(json.dumps(res, indent=2))
    for c in res["checks"]:
        print(f"[{c['verdict']}] {c['check']}: {c['observed']}")


if __name__ == "__main__":
    main()
