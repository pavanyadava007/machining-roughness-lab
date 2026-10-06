"""Where to run the next validation trials, and how many roughness readings per cut.

A Gaussian process on (speed, feed, depth) is fitted to the 27 cut means. Its posterior variance
does not depend on the measured values, only on where trials were run, so greedy max-variance
selection with fixed kernel hyperparameters is exact: pick the most uncertain candidate, add it
as a pseudo-observation, repeat.
"""

from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from mrl.config import PROCESSED, RESULTS, SEED

warnings.filterwarnings("ignore", module="sklearn.gaussian_process")
PROC = ["speed", "feed", "depth"]
GRID = {"speed": np.linspace(400, 1000, 7), "feed": np.linspace(0.05, 0.16, 6), "depth": np.linspace(0.2, 0.6, 5)}


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    # smoothness prior: length scales of at least 0.5 standard deviations of each process parameter.
    # Without it the 27 noisy cut means pull the feed length scale to ~0.08 and the GP explains
    # almost everything as noise, so the planner would cluster all picks in one corner.
    k = ConstantKernel(1.0) * RBF([1.0] * X.shape[1], (0.5, 20.0)) + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=SEED).fit(X, y)


def greedy_max_variance(gp: GaussianProcessRegressor, X_train: np.ndarray, X_cand: np.ndarray, k: int) -> list[int]:
    kernel = gp.kernel_
    chosen: list[int] = []
    X = X_train.copy()
    for _ in range(k):
        g = GaussianProcessRegressor(kernel, optimizer=None, normalize_y=False).fit(X, np.zeros(len(X)))
        _, sd = g.predict(X_cand, return_std=True)
        sd[chosen] = -1
        i = int(np.argmax(sd))
        chosen.append(i)
        X = np.vstack([X, X_cand[i]])
    return chosen


def replicate_table(within_sd: float, ns=(1, 2, 3, 5, 8)) -> list[dict]:
    return [{"readings_per_cut": n, "sd_of_cut_mean_um": within_sd / np.sqrt(n)} for n in ns]


def main(k: int = 6) -> None:
    lab = pd.read_csv(PROCESSED / "labels.csv")
    cuts = lab.groupby(["speed", "feed", "depth"], as_index=False).Ra.mean()
    sc = StandardScaler().fit(cuts[PROC])
    X = sc.transform(cuts[PROC])
    gp = fit_gp(X, cuts.Ra.to_numpy())
    cand = pd.DataFrame(list(itertools.product(*GRID.values())), columns=PROC)
    Xc = sc.transform(cand)
    mean, sd = gp.predict(Xc, return_std=True)
    cand["pred_Ra_um"], cand["pred_sd_um"] = mean, sd
    pick = greedy_max_variance(gp, X, Xc, k)
    nxt = cand.iloc[pick].assign(order=range(1, k + 1))
    nxt.round(4).to_csv(RESULTS / "next_trials.csv", index=False)
    within_sd = float(np.sqrt(lab.groupby("exp").Ra.var(ddof=1).mean()))
    out = {"kernel": str(gp.kernel_), "next_trials": nxt.round(4).to_dict("records"),
           "replicates": replicate_table(within_sd), "within_cut_sd_um": within_sd,
           "rerun_first": "the 6 cuts at 630 rpm with f 0.10/0.16 (implied cut length 2x, see audit)"}
    (RESULTS / "plan.json").write_text(json.dumps(out, indent=2))
    print(f"kernel {gp.kernel_}")
    print(nxt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
