import numpy as np
import pytest

from mrl.features import kinematic_ra_um, spectral_features, time_features, window_features

FS = 10_000


def _sine(f_hz, amp=1.0, n=FS, offset=0.0):
    t = np.arange(n) / FS
    return offset + amp * np.sin(2 * np.pi * f_hz * t)


def test_time_features_of_a_sine():
    f = time_features(_sine(50, amp=2.0))
    assert f["rms"] == pytest.approx(2 / np.sqrt(2), rel=1e-3)
    assert f["crest"] == pytest.approx(np.sqrt(2), rel=1e-2)


def test_band_shares_put_energy_in_the_right_band():
    f = spectral_features(_sine(700) + 0.01 * np.random.default_rng(0).normal(size=FS), spindle_hz=10)
    assert f["band_500_1000"] > 0.95
    assert sum(v for k, v in f.items() if k.startswith("band_")) <= 1.0 + 1e-6


def test_spindle_harmonic_is_found():
    x = _sine(400 / 60, amp=5.0) + _sine(1500, amp=1.0)
    f = spectral_features(x, spindle_hz=400 / 60)
    assert f["spindle_h1"] > 1.0
    assert f["spindle_h2"] < 0.5 * f["spindle_h1"]


def test_window_features_kc_uses_chip_area():
    n = FS
    forces = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 100.0)])
    acc = np.zeros((n, 3))
    f = window_features(forces, acc, 400, feed=0.1, depth=0.5)
    assert f["kc_res"] == pytest.approx(100 / 0.05)


def test_kinematic_roughness():
    # f = 0.16 mm/rev, r = 0.8 mm -> 0.0256 / 25.6 mm = 1.0 um
    assert kinematic_ra_um(0.16, 0.8) == pytest.approx(1.0)
