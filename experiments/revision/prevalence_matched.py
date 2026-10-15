#!/usr/bin/env python3
"""Reviewer comment 3: why is the cost reduction large for gait and small for speech?

Hypothesis: under the 4:1 loss the Bayes rule refers anyone with P(PD|x) > 0.2.
At speech prevalence (0.75) a symmetric SVM already sits near the refer-all bound,
so weighting has little room to help; at gait prevalence (0.55) it does not.

Test: swap prevalences by subsampling. Speech is subsampled to the gait prevalence
(all controls, PD cases removed) and gait to the speech prevalence (all PD, controls
removed). A fresh subsample is drawn for each split. If prevalence drives the gap,
the speech reduction should grow and the gait reduction should shrink.
"""
from __future__ import annotations

import json
import pickle
from multiprocessing import Pool

import numpy as np
from sklearn.preprocessing import StandardScaler

from rev_common import OUT, load_gait, load_speech, mean_ci, rc, refer_all_cost

ARMS = ("standard_symmetric", "cost_only", "joint_cal")
BUDGET = 0.5


def subsample(y, target_prev, rng):
    pd_idx, ct_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if target_prev < y.mean():  # remove PD cases
        k = int(round(target_prev * len(ct_idx) / (1 - target_prev)))
        pd_idx = rng.choice(pd_idx, k, replace=False)
    else:  # remove controls
        k = int(round((1 - target_prev) * len(pd_idx) / target_prev))
        ct_idx = rng.choice(ct_idx, k, replace=False)
    return np.sort(np.concatenate([pd_idx, ct_idx]))


def one_split(args):
    X, y, seed = args
    pool, test = rc.split_indices(y, seed)
    sc = StandardScaler().fit(X[pool])
    Xp, Xt, yp, yt = sc.transform(X[pool]), sc.transform(X[test]), y[pool], y[test]
    gamma = rc.gamma_scale(Xp)
    budget = max(rc.INITIAL_N, round(len(pool) * BUDGET))
    row = {"seed": seed, "n": len(y), "prevalence": float(y.mean()), "refer_all": refer_all_cost(yt)}
    for arm in ARMS:
        row[arm] = rc.run_al(Xp, yp, Xt, yt, seed, budget, *rc.METHODS[arm], gamma)["common_cost_r4"]
    for k, v in rc.full_pool_baselines(Xp, yp, Xt, yt, seed, gamma).items():
        row[k] = v["common_cost_r4"]
    return row


def run(X, y, target_prev=None):
    jobs = []
    for seed in range(rc.N_SPLITS):
        if target_prev is None:
            idx = np.arange(len(y))
        else:
            idx = subsample(y, target_prev, np.random.RandomState(5000 + seed))
        jobs.append((X[idx], y[idx], seed))
    with Pool(8) as p:
        rows = p.map(one_split, jobs)
    summ = {k: mean_ci([r[k] for r in rows]) for k in rows[0] if k not in ("seed",)}
    std = np.array([r["standard_symmetric"] for r in rows]); joint = np.array([r["joint_cal"] for r in rows])
    summ["joint_vs_standard_reduction_pct"] = float(100 * (std.mean() - joint.mean()) / std.mean())
    summ["standard_minus_refer_all"] = float(std.mean() - np.mean([r["refer_all"] for r in rows]))
    return rows, summ


def main():
    Xg, yg, _ = load_gait()
    Xs, ys = load_speech()
    pg, ps = float(yg.mean()), float(ys.mean())
    settings = {
        "gait_original": (Xg, yg, None),
        "gait_at_speech_prevalence": (Xg, yg, ps),
        "speech_original": (Xs, ys, None),
        "speech_at_gait_prevalence": (Xs, ys, pg),
    }
    result, allrows = {}, {}
    for name, (X, y, prev) in settings.items():
        rows, summ = run(X, y, prev)
        result[name] = summ; allrows[name] = rows
        print(name, {k: (round(v[0], 2) if isinstance(v, tuple) else round(v, 2)) for k, v in summ.items()}, flush=True)
    json.dump(result, open(OUT / "prevalence_matched.json", "w"), indent=2)
    pickle.dump(allrows, open(OUT / "prevalence_matched_records.pkl", "wb"))


if __name__ == "__main__":
    main()
