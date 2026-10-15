#!/usr/bin/env python3
"""Reviewer comment 2: downstream testing burden of the sensitivity-weighted models.

1. Refer-all reference: C4 of labelling every test participant PD, per split.
2. Prevalence scenarios per 1,000 people for the 50%-budget and full-pool operating
   points (sensitivity/specificity transported to target prevalences).
3. Net benefit at threshold probability 0.2 (the 4:1 FN:FP exchange rate). Note
   C4/100 = 4*pi - 4*NB(0.2), so C4 is a linear transform of net benefit.
4. Decision curves for full-pool symmetric SVMs with Platt-scaled probabilities.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rev_common import OUT, ROOT, load_gait, load_speech, mean_ci, rc, refer_all_cost

PREVALENCES = (0.01, 0.05, 0.10, 0.30)
THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.01), 2)
OPERATING = {
    "standard_symmetric@0.5": ("standard_symmetric", 0.5),
    "cost_only@0.5": ("cost_only", 0.5),
    "joint_cal@0.5": ("joint_cal", 0.5),
    "fullpool_symmetric": ("fullpool_symmetric", 1.0),
    "fullpool_cost": ("fullpool_cost", 1.0),
    "fullpool_cost_threshold": ("fullpool_cost_threshold", 1.0),
}


def operating_points(records):
    out = {}
    for label, (method, budget) in OPERATING.items():
        rows = [r for r in records if r["method"] == method and abs(r["budget"] - budget) < 1e-9]
        sens = np.array([1 - r["fnr"] for r in rows])
        spec = np.array([r["specificity"] for r in rows])
        out[label] = {"sensitivity": float(sens.mean()), "specificity": float(spec.mean()),
                      "c4": mean_ci([r["common_cost_r4"] for r in rows])}
    return out


def scenario(sens, spec, prev, n=1000):
    pd_n, ctrl = prev * n, (1 - prev) * n
    tp, fn = sens * pd_n, (1 - sens) * pd_n
    fp, tn = (1 - spec) * ctrl, spec * ctrl
    referred = tp + fp
    return {"prevalence": prev, "referred": referred, "tp": tp, "fn": fn, "fp": fp,
            "ppv": tp / referred if referred else float("nan"),
            "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "tests_per_detected_case": referred / tp if tp else float("nan"),
            "unnecessary_tests_avoided_vs_refer_all": ctrl - fp,
            "net_benefit_pt0.2": (tp - 0.25 * fp) / n}


def decision_curves(X, y):
    nb_model, nb_all = [], []
    for seed in range(rc.N_SPLITS):
        pool, test = rc.split_indices(y, seed)
        sc = StandardScaler().fit(X[pool])
        Xp, Xt = sc.transform(X[pool]), sc.transform(X[test])
        clf = SVC(C=1, kernel="rbf", gamma=rc.gamma_scale(Xp), probability=True,
                  random_state=seed).fit(Xp, y[pool])
        p, yt = clf.predict_proba(Xt)[:, 1], y[test]
        n, prev = len(yt), yt.mean()
        row_m, row_a = [], []
        for pt in THRESHOLDS:
            pred = p >= pt
            tp = np.sum(pred & (yt == 1)); fp = np.sum(pred & (yt == 0))
            w = pt / (1 - pt)
            row_m.append(tp / n - fp / n * w)
            row_a.append(prev - (1 - prev) * w)
        nb_model.append(row_m); nb_all.append(row_a)
    return np.array(nb_model), np.array(nb_all)


def main():
    data = {"gait": (load_gait()[:2], "gait_classical12"), "speech": (load_speech(), "speech_raw754")}
    result = {}
    for name, ((X, y), tag) in data.items():
        records = pickle.load(open(ROOT / "results" / "revised_cal" / f"{tag}_records.pkl", "rb"))
        refer_all = [refer_all_cost(y[rc.split_indices(y, s)[1]]) for s in range(rc.N_SPLITS)]
        ops = operating_points(records)
        study_prev = float(y.mean())
        scen = {lab: [scenario(o["sensitivity"], o["specificity"], p) for p in PREVALENCES + (study_prev,)]
                for lab, o in ops.items()}
        nb_m, nb_a = decision_curves(X, y)
        result[name] = {
            "study_prevalence": study_prev,
            "refer_all_c4": mean_ci(refer_all),
            "operating_points": ops,
            "scenarios_per_1000": scen,
            "decision_curve": {"thresholds": THRESHOLDS.tolist(),
                               "nb_symmetric_svm": nb_m.mean(0).tolist(),
                               "nb_symmetric_svm_sd": nb_m.std(0).tolist(),
                               "nb_refer_all": nb_a.mean(0).tolist()},
        }
        print(name, "refer-all C4", result[name]["refer_all_c4"], flush=True)
    json.dump(result, open(OUT / "burden_dca.json", "w"), indent=2)
    print(OUT / "burden_dca.json")


if __name__ == "__main__":
    main()
