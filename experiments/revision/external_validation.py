#!/usr/bin/env python3
"""Reviewer comment 4: robustness across cohorts.

(a) Gait leave-one-study-out (LOSO). The PhysioNet gaitpdb pools three independent
    studies (Ga: Yogev et al.; Ju: Hausdorff et al.; Si: Frenkel-Toledo et al.)
    recorded with different protocols and sites. Each study is held out in turn as
    an external test cohort; the other two form the development pool. Twenty
    seeds vary only the label-blind initial sample and k-means restarts.
(b) Independent speech cohort: UCI Parkinsons (Little et al. 2007), 31 participants,
    22 dysphonia features. Its features are not compatible with the Sakar set, so
    this is a replication of the framework, not a transfer of a fitted model.
    Splits are grouped by participant; each recording is one observation.
Arms: component ablation (standard / cost-only / joint) at 50% and full pool,
full-pool references, refer-all, and the four adaptive policies at 50%.
"""
from __future__ import annotations

import json
import pickle
from multiprocessing import Pool

import numpy as np
from sklearn.preprocessing import StandardScaler

from rev_common import OUT, ca, load_gait, load_little, mean_ci, rc, refer_all_cost

ARMS = ("standard_symmetric", "cost_only", "joint_cal")
POLICIES = ("standard", "fixed_r4", "adaptive_rule", "clinrisk_llm")
LOSO_SEEDS = 20


def evaluate(Xp, yp, Xt, yt, seed):
    sc = StandardScaler().fit(Xp)
    Xp, Xt = sc.transform(Xp), sc.transform(Xt)
    gamma = rc.gamma_scale(Xp)
    budget = max(rc.INITIAL_N, round(len(yp) * 0.5))
    row = {"seed": seed, "n_pool": len(yp), "n_test": len(yt),
           "test_prevalence": float(yt.mean()), "refer_all": refer_all_cost(yt)}
    for arm in ARMS:
        row[f"{arm}@0.5"] = rc.run_al(Xp, yp, Xt, yt, seed, budget, *rc.METHODS[arm], gamma)
    for k, v in rc.full_pool_baselines(Xp, yp, Xt, yt, seed, gamma).items():
        row[k] = v
    for mode in POLICIES:
        rows = ca.run_one(Xp, yp, Xt, yt, seed, mode)
        row[f"policy_{mode}@0.5"] = next(r for r in rows if r["budget"] == 50)
    return row


def loso_job(args):
    X, y, study, held, seed = args
    tr, te = study != held, study == held
    return {"held_out": held, **evaluate(X[tr], y[tr], X[te], y[te], seed)}


def grouped_split(subject, y, seed):
    subs = np.unique(subject)
    lab = np.array([y[subject == s][0] for s in subs])
    rng = np.random.RandomState(seed + 1000)
    test_s = []
    for c in (0, 1):
        z = subs[lab == c].copy(); rng.shuffle(z)
        test_s.extend(z[:max(1, round(rc.TEST_FRAC * len(z)))])
    te = np.isin(subject, test_s)
    return np.flatnonzero(~te), np.flatnonzero(te)


def little_job(args):
    X, y, subject, seed = args
    pool, test = grouped_split(subject, y, seed)
    return evaluate(X[pool], y[pool], X[test], y[test], seed)


def cost(v):
    return v["common_cost_r4"] if "common_cost_r4" in v else v["cost4"]


def summarise(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    out = {"refer_all": mean_ci([r["refer_all"] for r in rows]),
           "test_prevalence": float(np.mean([r["test_prevalence"] for r in rows]))}
    for k in keys:
        out[k] = {"c4": mean_ci([cost(r[k]) for r in rows]),
                  "fnr": float(np.mean([r[k]["fnr"] for r in rows])),
                  "specificity": float(np.mean([r[k]["specificity"] for r in rows])),
                  "balanced_accuracy": float(np.mean([r[k]["balanced_accuracy"] for r in rows])),
                  "auc": float(np.mean([r[k]["auc"] for r in rows]))}
    return out


def main():
    X, y, study = load_gait()
    jobs = [(X, y, study, h, s) for h in ("Ga", "Ju", "Si") for s in range(LOSO_SEEDS)]
    with Pool(8) as p:
        loso = p.map(loso_job, jobs)
    result = {"gait_loso": {h: summarise([r for r in loso if r["held_out"] == h]) for h in ("Ga", "Ju", "Si")}}
    result["gait_loso_counts"] = {h: {"n": int((study == h).sum()), "pd": int(y[study == h].sum())}
                                  for h in ("Ga", "Ju", "Si")}

    Xl, yl, subj = load_little()
    with Pool(8) as p:
        little = p.map(little_job, [(Xl, yl, subj, s) for s in range(rc.N_SPLITS)])
    result["little_speech"] = summarise(little)
    result["little_counts"] = {"recordings": len(yl), "participants": int(len(np.unique(subj))),
                               "pd_participants": int(len(np.unique(subj[yl == 1])))}
    json.dump(result, open(OUT / "external_validation.json", "w"), indent=2)
    pickle.dump({"loso": loso, "little": little}, open(OUT / "external_validation_records.pkl", "wb"))
    for cohort in ("Ga", "Ju", "Si"):
        r = result["gait_loso"][cohort]
        print(cohort, "refer-all %.1f" % r["refer_all"][0],
              {k: round(v["c4"][0], 1) for k, v in r.items() if isinstance(v, dict)})
    r = result["little_speech"]
    print("Little", "refer-all %.1f" % r["refer_all"][0],
          {k: round(v["c4"][0], 1) for k, v in r.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
