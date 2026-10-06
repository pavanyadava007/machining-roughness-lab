"""Raw CSVs -> data/processed/windows.parquet (one row per 1 s window) and labels.csv (one row per cut)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from mrl.config import ACC_CH, DEPTHS_MM, FEEDS_MM_REV, FORCE_CH, FS_HZ, PROCESSED, RAW, SPEEDS_RPM, WINDOW_S, exp_id
from mrl.features import iter_windows, window_features


def load_labels() -> pd.DataFrame:
    rows = []
    for s, f, d in itertools.product(SPEEDS_RPM, FEEDS_MM_REV, DEPTHS_MM):
        eid = exp_id(s, f, d)
        r = pd.read_csv(RAW / "rough" / f"r_{eid}.csv")
        for rep, row in enumerate(r.itertuples(index=False)):
            rows.append({"exp": eid, "speed": s, "feed": f, "depth": d, "rep": rep,
                         "Ra": row.Ra, "Rq": row.Rq, "Rz": row.Rz})
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    labels.to_csv(PROCESSED / "labels.csv", index=False)
    win = int(WINDOW_S * FS_HZ)
    rows = []
    for s, f, d in itertools.product(SPEEDS_RPM, FEEDS_MM_REV, DEPTHS_MM):
        eid = exp_id(s, f, d)
        dyn = pd.read_csv(RAW / "dyn" / f"f_{eid}.csv", usecols=list(FORCE_CH)).to_numpy(np.float64)
        acc = pd.read_csv(RAW / "acc" / f"a_{eid}.csv", usecols=list(ACC_CH)).to_numpy(np.float64)
        n = min(len(dyn), len(acc))
        for i, sl in iter_windows(n, win):
            feats = window_features(dyn[sl], acc[sl], s, f, d)
            rows.append({"exp": eid, "speed": s, "feed": f, "depth": d, "win": i, **feats})
        print(f"{eid}: {n / FS_HZ:6.1f} s -> {n // win} windows")
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / "windows.parquet", index=False)
    print(f"{len(df)} windows, {df.shape[1] - 5} features, {df.exp.nunique()} cuts")


if __name__ == "__main__":
    main()
