import numpy as np
import pandas as pd
import pytest

from mrl.audit import implied_cut_length_mm, spindle_rpm_from_force, within_cut_drift
from mrl.evaluate import cut_table, metrics, noise_floor, paired_bootstrap, run_cv
from mrl.plan import fit_gp, greedy_max_variance, replicate_table


def _synthetic(n_cuts=9, n_win=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, labs = [], []
    for i in range(n_cuts):
        s, f, d = [400, 630, 1000][i % 3], [0.05, 0.10, 0.16][(i // 3) % 3], 0.2 + 0.2 * (i % 3)
        e = f"{s:04d}_{f:.2f}_{d:.2f}"
        ra = 0.4 + 0.3 * d
        for w in range(n_win):
            rows.append({"exp": e, "speed": s, "feed": f, "depth": d, "win": w,
                         "x1": ra + rng.normal(0, 0.01), "x2": rng.normal()})
        for r in range(3):
            labs.append({"exp": e, "speed": s, "feed": f, "depth": d, "rep": r, "Ra": ra + rng.normal(0, 0.02),
                         "Rz": 5 * ra})
    return pd.DataFrame(rows), pd.DataFrame(labs)


def test_noise_floor_matches_within_sd():
    _, lab = _synthetic()
    nf = noise_floor(lab, "Ra")
    assert nf["rmse_floor_cut_mean"] == pytest.approx(nf["within_cut_sd"] / np.sqrt(3))
    assert 0 < nf["r2_ceiling"] <= 1


def test_leave_one_cut_out_never_trains_on_the_held_out_cut():
    win, lab = _synthetic()
    cuts = cut_table(win, lab, ["x1", "x2"], "Ra")
    seen = []

    def spy(tr, te, _w, _f):
        assert not set(tr.index) & set(te.index)
        seen.append(te.index[0])
        return np.zeros(len(te))

    run_cv(cuts, win, ["x1", "x2"], [[e] for e in cuts.index], {"spy": spy})
    assert sorted(seen) == sorted(cuts.index)


def test_informative_feature_beats_mean_on_synthetic_data():
    from mrl.evaluate import m_mean, m_sensor_ridge

    win, lab = _synthetic()
    cuts = cut_table(win, lab, ["x1", "x2"], "Ra")
    p = run_cv(cuts, win, ["x1", "x2"], [[e] for e in cuts.index], {"mean": m_mean, "ridge": m_sensor_ridge})
    assert metrics(cuts.y, p.ridge)["rmse"] < metrics(cuts.y, p["mean"])["rmse"]
    b = paired_bootstrap(cuts.y, p.ridge, p["mean"], n_boot=500)
    assert b["ci95"][1] < 0


def test_implied_cut_length_and_drift():
    win = pd.DataFrame({"exp": ["a"] * 30 + ["b"] * 15, "win": list(range(30)) + list(range(15)),
                        "speed": [600] * 45, "feed": [0.1] * 30 + [0.2] * 15,
                        "Fres_mean": list(np.linspace(100, 150, 30)) + [100.0] * 15})
    length = implied_cut_length_mm(win)
    assert length["a"] == pytest.approx(30.0) and length["b"] == pytest.approx(30.0)
    drift = within_cut_drift(win)
    assert drift["a"] > 40 and drift["b"] == pytest.approx(0.0)


def test_spindle_speed_from_force_spectrum():
    t = np.arange(20 * 10_000) / 10_000
    f = 200 + 10 * np.sin(2 * np.pi * (650 / 60) * t)
    assert spindle_rpm_from_force(f, nominal_rpm=630) == pytest.approx(650, abs=8)


def test_greedy_planner_picks_distinct_far_points():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.2]])
    gp = fit_gp(X, np.array([1.0, 1.1, 0.9]))
    cand = np.array([[0.05, 0.05], [3.0, 3.0], [-3.0, 3.0]])
    picks = greedy_max_variance(gp, X, cand, 2)
    assert len(set(picks)) == 2 and 0 not in picks


def test_replicates_shrink_as_root_n():
    r = replicate_table(0.09, ns=(1, 4))
    assert r[1]["sd_of_cut_mean_um"] == pytest.approx(0.045)
