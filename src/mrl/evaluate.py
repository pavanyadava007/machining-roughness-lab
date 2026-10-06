"""Model comparison with cut-level validation.

Unit of evaluation is the cut (27 parameter combinations, one label = mean of 3 roughness readings).
Every window of a held-out cut is unseen during training, so neighbouring windows of the same cut
cannot leak. The random window split is run only to show how large that leak would be.
"""

from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mrl.config import NOSE_RADIUS_MM, PROCESSED, RESULTS, SEED
from mrl.features import kinematic_ra_um

warnings.filterwarnings("ignore", category=UserWarning)
ALPHAS = np.logspace(-2, 4, 25)
PROC = ["speed", "feed", "depth"]
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)


def load():
    win = pd.read_parquet(PROCESSED / "windows.parquet")
    lab = pd.read_csv(PROCESSED / "labels.csv")
    feat_cols = [c for c in win.columns if c not in ("exp", "speed", "feed", "depth", "win")]
    return win, lab, feat_cols


def cut_table(win: pd.DataFrame, lab: pd.DataFrame, feat_cols: list[str], target: str) -> pd.DataFrame:
    agg = win.groupby("exp")[feat_cols].median()
    y = lab.groupby("exp").agg(speed=("speed", "first"), feed=("feed", "first"), depth=("depth", "first"),
                               y=(target, "mean"), y_sd=(target, "std"))
    return y.join(agg)


def noise_floor(lab: pd.DataFrame, target: str) -> dict[str, float]:
    """Repeatability of the roughness readings sets the best error any model can reach on cut means."""
    g = lab.groupby("exp")[target]
    s2_within = float(g.var(ddof=1).mean())
    n_rep = float(g.size().mean())
    means = g.mean()
    floor = float(np.sqrt(s2_within / n_rep))
    return {"within_cut_sd": float(np.sqrt(s2_within)), "rmse_floor_cut_mean": floor,
            "between_cut_sd": float(means.std(ddof=0)),
            "r2_ceiling": float(1 - floor**2 / means.var(ddof=0))}


def metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "spearman": float(stats.spearmanr(y, p).statistic)}


# ---- model zoo: each takes (train cut table, test cut table, windows) and returns test predictions ----

def m_mean(tr, te, _w, _f):
    return np.full(len(te), tr.y.mean())


def m_kinematic_raw(_tr, te, _w, _f):
    return np.array([kinematic_ra_um(f, NOSE_RADIUS_MM) for f in te.feed])


def m_kinematic_cal(tr, te, _w, _f):
    k = lambda t: np.array([[kinematic_ra_um(f, NOSE_RADIUS_MM)] for f in t.feed])  # noqa: E731
    return LinearRegression().fit(k(tr), tr.y).predict(k(te))


def m_process_ridge(tr, te, _w, _f):
    m = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), RidgeCV(alphas=ALPHAS))
    return m.fit(tr[PROC], tr.y).predict(te[PROC])


def m_process_gpr(tr, te, _w, _f):
    k = ConstantKernel(1.0) * RBF([1.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(0.1, (1e-4, 1e1))
    sc = StandardScaler().fit(tr[PROC])
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)
    return gp.fit(sc.transform(tr[PROC]), tr.y).predict(sc.transform(te[PROC]))


def m_sensor_ridge(tr, te, _w, feats):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return m.fit(tr[feats], tr.y).predict(te[feats])


def m_sensor_pls(tr, te, _w, feats):
    m = make_pipeline(StandardScaler(), PLSRegression(n_components=2, scale=False))
    return m.fit(tr[feats], tr.y).predict(te[feats]).ravel()


def m_depth_group_mean(tr, te, _w, _f):
    g = tr.groupby("depth").y.mean()
    return te.depth.map(g).fillna(tr.y.mean()).to_numpy()


def m_sensor_select5_ridge(tr, te, _w, feats, k=5):
    """Top-k features by |Spearman| with the label, chosen on the training cuts of each fold only."""
    rank = sorted(feats, key=lambda f: -abs(stats.spearmanr(tr[f], tr.y).statistic))[:k]
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return m.fit(tr[rank], tr.y).predict(te[rank])


def _lgb_windows(tr, te, w, cols):
    wtr = w[w.exp.isin(tr.index)]
    wte = w[w.exp.isin(te.index)]
    ytr = wtr.exp.map(tr.y)
    m = lgb.LGBMRegressor(**LGB_PARAMS).fit(wtr[cols], ytr)
    pw = pd.Series(m.predict(wte[cols]), index=wte.index)
    return pw.groupby(wte.exp).mean().reindex(te.index).to_numpy()


def m_sensor_lgb(tr, te, w, feats):
    return _lgb_windows(tr, te, w, feats)


def m_sensor_process_lgb(tr, te, w, feats):
    return _lgb_windows(tr, te, w, feats + PROC)


def m_sensor_process_ridge(tr, te, _w, feats):
    cols = feats + PROC
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return m.fit(tr[cols], tr.y).predict(te[cols])


MODELS = {
    "mean_baseline": m_mean,
    "kinematic_f2_32r_raw": m_kinematic_raw,
    "kinematic_f2_32r_calibrated": m_kinematic_cal,
    "process_ridge_quadratic": m_process_ridge,
    "process_gpr": m_process_gpr,
    "depth_group_mean": m_depth_group_mean,
    "sensor_ridge": m_sensor_ridge,
    "sensor_pls2": m_sensor_pls,
    "sensor_select5_ridge": m_sensor_select5_ridge,
    "sensor_lgbm_windows": m_sensor_lgb,
    "sensor_process_ridge": m_sensor_process_ridge,
    "sensor_process_lgbm_windows": m_sensor_process_lgb,
}


def run_cv(cuts, win, feats, folds, models=MODELS):
    preds = {name: pd.Series(np.nan, index=cuts.index) for name in models}
    for test_idx in folds:
        te = cuts.loc[test_idx]
        tr = cuts.drop(index=test_idx)
        for name, fn in models.items():
            preds[name].loc[test_idx] = fn(tr, te, win, feats)
    return pd.DataFrame(preds)


def paired_bootstrap(y, p_model, p_base, n_boot=5000):
    """95 % CI of RMSE(model) - RMSE(baseline) over cuts; negative = model better."""
    rng = np.random.default_rng(SEED)
    y, a, b = (np.asarray(v, float) for v in (y, p_model, p_base))
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    d = np.sqrt(((y[idx] - a[idx]) ** 2).mean(1)) - np.sqrt(((y[idx] - b[idx]) ** 2).mean(1))
    return {"delta_rmse": float(np.sqrt(((y - a) ** 2).mean()) - np.sqrt(((y - b) ** 2).mean())),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def permutation_test(cuts, feats, n_perm=300):
    """Null distribution of LOCO RMSE for sensor_ridge with Ra shuffled across cuts."""
    folds = [[e] for e in cuts.index]
    rng = np.random.default_rng(SEED)
    obs = metrics(cuts.y, run_cv(cuts, None, feats, folds, {"r": m_sensor_ridge})["r"])["rmse"]
    null = []
    for _ in range(n_perm):
        c = cuts.copy()
        c["y"] = rng.permutation(c.y.to_numpy())
        null.append(metrics(c.y, run_cv(c, None, feats, folds, {"r": m_sensor_ridge})["r"])["rmse"])
    null = np.array(null)
    return {"observed_rmse": obs, "null_median_rmse": float(np.median(null)),
            "p_value": float((np.sum(null <= obs) + 1) / (n_perm + 1)), "n_perm": n_perm}


def leakage_demo(win, lab, feats, target):
    """Window-level R2: random split (windows of one cut on both sides) vs cut-grouped split."""
    y_cut = lab.groupby("exp")[target].mean()
    y = win.exp.map(y_cut).to_numpy()
    X = win[feats].to_numpy()
    out = {}
    for name, splitter in (("random_window_kfold5", KFold(5, shuffle=True, random_state=SEED)),):
        p = np.zeros(len(y))
        for tr, te in splitter.split(X):
            p[te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[tr], y[tr]).predict(X[te])
        out[name] = metrics(y, p)
    exps = win.exp.unique()
    p = np.zeros(len(y))
    for e in exps:
        te = (win.exp == e).to_numpy()
        p[te] = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[~te], y[~te]).predict(X[te])
    out["leave_one_cut_out_windows"] = metrics(y, p)
    return out


def top_correlations(cuts, feats, k=8):
    rows = []
    for f in feats:
        r = stats.spearmanr(cuts[f], cuts.y)
        rows.append((f, float(r.statistic), float(r.pvalue)))
    rows.sort(key=lambda t: -abs(t[1]))
    return [{"feature": f, "spearman": r, "p_uncorrected": p} for f, r, p in rows[:k]]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    win, lab, feats = load()
    report: dict = {"n_cuts": int(win.exp.nunique()), "n_windows": len(win), "n_features": len(feats),
                    "n_roughness_readings": len(lab), "targets": {}}
    all_preds = []
    for target in ("Ra", "Rz"):
        cuts = cut_table(win, lab, feats, target)
        loco = [[e] for e in cuts.index]
        loso = [list(cuts.index[cuts.speed == s]) for s in sorted(cuts.speed.unique())]
        p_loco = run_cv(cuts, win, feats, loco)
        p_loso = run_cv(cuts, win, feats, loso)
        t = {"noise_floor": noise_floor(lab, target),
             "leave_one_cut_out": {m: metrics(cuts.y, p_loco[m]) for m in MODELS},
             "leave_one_speed_out": {m: metrics(cuts.y, p_loso[m]) for m in MODELS},
             "top_spearman_features": top_correlations(cuts, feats)}
        t["vs_mean_baseline_bootstrap"] = {m: paired_bootstrap(cuts.y, p_loco[m], p_loco["mean_baseline"])
                                           for m in MODELS if m != "mean_baseline"}
        if target == "Ra":
            t["permutation_sensor_ridge"] = permutation_test(cuts, feats)
            t["leakage_demo_lgbm"] = leakage_demo(win, lab, feats, target)
        report["targets"][target] = t
        pr = p_loco.add_prefix("loco_").join(p_loso.add_prefix("loso_"))
        pr.insert(0, "measured", cuts.y)
        pr.insert(1, "measured_sd", cuts.y_sd)
        pr.insert(0, "target", target)
        all_preds.append(pr.join(cuts[PROC]).reset_index())
        best = min(t["leave_one_cut_out"].items(), key=lambda kv: kv[1]["rmse"])
        print(f"[{target}] floor {t['noise_floor']['rmse_floor_cut_mean']:.3f}  best LOCO {best[0]} "
              f"rmse {best[1]['rmse']:.3f} r2 {best[1]['r2']:.2f}")
        for m, v in t["leave_one_cut_out"].items():
            s = t["leave_one_speed_out"][m]
            print(f"   {m:30s} LOCO rmse {v['rmse']:.3f} r2 {v['r2']:+.2f} rho {v['spearman']:+.2f} | "
                  f"LOSO rmse {s['rmse']:.3f} r2 {s['r2']:+.2f}")
    pd.concat(all_preds).to_csv(RESULTS / "predictions.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report["targets"]["Ra"][k] for k in ("noise_floor", "permutation_sensor_ridge",
                                                            "leakage_demo_lgbm")}, indent=1))


if __name__ == "__main__":
    main()
