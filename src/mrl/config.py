"""Paths and constants shared by the pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
META = DATA / "meta"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

MANIFEST = META / "manifest_v3.json"
DATASET_DOI = "10.17632/ybyhxv4czd.3"
DATASET_URL = "https://data.mendeley.com/datasets/ybyhxv4czd/3"

FS_HZ = 10_000  # sampling rate of dynamometer and accelerometer (dataset paper)
WINDOW_S = 1.0  # feature window length
NOSE_RADIUS_MM = 0.8  # CNMG120408 insert: "08" = 0.8 mm corner radius

SPEEDS_RPM = (400, 630, 1000)
FEEDS_MM_REV = (0.05, 0.10, 0.16)
DEPTHS_MM = (0.20, 0.40, 0.60)

FORCE_CH = ("Fx", "Fy", "Fz")
ACC_CH = ("Acc_X", "Acc_Y", "Acc_Z")

# frequency bands for spectral energy shares, Hz (Nyquist is 5 kHz)
BANDS_HZ = ((0, 20), (20, 100), (100, 500), (500, 1000), (1000, 2500), (2500, 5000))

SEED = 7


def exp_id(speed: int, feed: float, depth: float) -> str:
    """Experiment id in the dataset's file-name convention, e.g. 0400_0.05_0.20."""
    return f"{speed:04d}_{feed:.2f}_{depth:.2f}"


def parse_exp_id(eid: str) -> tuple[int, float, float]:
    s, f, d = eid.split("_")
    return int(s), float(f), float(d)
