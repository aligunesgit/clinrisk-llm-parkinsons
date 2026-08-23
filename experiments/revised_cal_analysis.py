#!/usr/bin/env python3
"""Leakage-resistant ablation study for the Parkinson cAL manuscript.

The analysis deliberately uses only pre-specified representations (gait-12 and
speech-754), a random label-blind initial sample, a common evaluation loss, and
separate query-only, cost-only, and joint cost-sensitive AL arms.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_score, roc_auc_score, roc_curve)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revised_cal"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 60
TEST_FRAC = 0.30
INITIAL_N = 10
BATCH = 5
CANDIDATES = 25
RISK = 4.0
BUDGETS = tuple(np.round(np.arange(0.1, 1.01, 0.1), 1))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_indices(y, seed):
    rng = np.random.RandomState(seed + 1000)
    test, pool = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = max(1, round(TEST_FRAC * len(idx)))
        test.extend(idx[:n_test]); pool.extend(idx[n_test:])
    return np.asarray(pool), np.asarray(test)


def gamma_scale(X):
    return 1.0 / (X.shape[1] * X.var()) if X.var() else 1.0


def kernel_kmeans(X, k, gamma, seed, restarts=10, max_iter=100):
    if len(X) <= k:
        return np.arange(len(X))
    K = rbf_kernel(X, X, gamma=gamma)
    diag = np.diag(K)
    rng = np.random.RandomState(seed)
    best_assign, best_obj = None, np.inf
    for _ in range(restarts):
        assign = rng.randint(0, k, len(X))
        for c in range(k):
            assign[c] = c
        for _ in range(max_iter):
            dist = np.full((len(X), k), np.inf)
            for c in range(k):
                mask = assign == c
                if mask.any():
                    dist[:, c] = diag - 2 * K[:, mask].mean(1) + K[np.ix_(mask, mask)].mean()
            new = dist.argmin(1)
            if np.array_equal(new, assign):
                break
            assign = new
        obj = dist[np.arange(len(X)), assign].sum()
        if obj < best_obj:
            best_assign, best_obj = assign.copy(), obj
    return best_assign


def query_batch(clf, X, risk_query, gamma, seed):
    d = clf.decision_function(X)
    # Lowest score is queried. Dividing the predicted-negative side by risk
    # gives priority to possible false negatives near the current boundary.
    score = np.where(d <= 0, np.abs(d) / RISK if risk_query else np.abs(d), np.abs(d))
    cand = np.argsort(score)[:min(CANDIDATES, len(X))]
    if len(cand) <= BATCH:
        return cand
    groups = kernel_kmeans(X[cand], BATCH, gamma, seed)
    chosen = []
    for c in range(BATCH):
        local = np.flatnonzero(groups == c)
        if len(local):
            chosen.append(cand[local[np.argmin(score[cand[local]])]])
    return np.asarray(chosen, dtype=int)


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "fnr": fn / (fn + tp), "specificity": tn / (tn + fp),
        "precision": precision_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, score), "common_cost_r4": (RISK * fn + fp) / len(y) * 100,
        "fn": int(fn), "fp": int(fp), "n": len(y),
    }


METHODS = {
    "random_symmetric": (False, False, True),
    "standard_symmetric": (False, False, False),
    "query_only": (True, False, False),
    "cost_only": (False, True, False),
    "joint_cal": (True, True, False),
}


def run_al(Xp, yp, Xt, yt, seed, budget, risk_query, cost_train, random_query, gamma):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(yp))
    labeled = list(order[:INITIAL_N])
    unlabeled = list(order[INITIAL_N:])
    # Label-blind initialization; if it contains one class, every additional
    # random label is charged to the budget until both classes are observed.
    while len(set(yp[labeled])) < 2 and unlabeled:
        labeled.append(unlabeled.pop())
    budget = max(budget, len(labeled))
    while len(labeled) < budget and unlabeled:
        cw = {0: 1.0, 1: RISK if cost_train else 1.0}
        clf = SVC(C=1, kernel="rbf", gamma=gamma, class_weight=cw).fit(Xp[labeled], yp[labeled])
        take = min(BATCH, budget - len(labeled), len(unlabeled))
        if random_query:
            local = rng.choice(len(unlabeled), take, replace=False)
        else:
            local = query_batch(clf, Xp[unlabeled], risk_query, gamma, seed + len(labeled))[:take]
        selected = [unlabeled[i] for i in local]
        labeled.extend(selected)
        selected_set = set(selected)
        unlabeled = [i for i in unlabeled if i not in selected_set]
    cw = {0: 1.0, 1: RISK if cost_train else 1.0}
    clf = SVC(C=1, kernel="rbf", gamma=gamma, class_weight=cw).fit(Xp[labeled], yp[labeled])
    score = clf.decision_function(Xt)
    return metrics(yt, (score > 0).astype(int), score)


def full_pool_baselines(Xp, yp, Xt, yt, seed, gamma):
    out = {}
    for name, weight in (("fullpool_symmetric", 1.0), ("fullpool_cost", RISK)):
        clf = SVC(C=1, kernel="rbf", gamma=gamma, class_weight={0: 1, 1: weight}).fit(Xp, yp)
        score = clf.decision_function(Xt)
        out[name] = metrics(yt, (score > 0).astype(int), score)
    proto = SVC(C=1, kernel="rbf", gamma=gamma, class_weight={0: 1, 1: 1})
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(proto, Xp, yp, cv=cv, method="decision_function")
    fpr, tpr, thresholds = roc_curve(yp, oof)
    prevalence = yp.mean()
    expected = RISK * prevalence * (1 - tpr) + (1 - prevalence) * fpr
    threshold = thresholds[np.argmin(expected)]
    clf = proto.fit(Xp, yp)
    score = clf.decision_function(Xt)
    out["fullpool_cost_threshold"] = metrics(yt, (score > threshold).astype(int), score)
    return out


def ci(values):
    a = np.asarray(values)
    # Descriptive split-level interval; not used as an independent-population p-value.
    res = bootstrap((a,), np.mean, n_resamples=5000, random_state=7, method="percentile")
    return [float(a.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)]


def analyse_dataset(X, y):
    records = []
    for seed in range(N_SPLITS):
        pool, test = split_indices(y, seed)
        scaler = StandardScaler().fit(X[pool])
        Xp, Xt = scaler.transform(X[pool]), scaler.transform(X[test])
        yp, yt = y[pool], y[test]
        gamma = gamma_scale(Xp)
        for fraction in BUDGETS:
            budget = max(INITIAL_N, round(len(pool) * fraction))
            for name, args in METHODS.items():
                m = run_al(Xp, yp, Xt, yt, seed, budget, *args, gamma)
                records.append({"seed": seed, "budget": fraction, "method": name, **m})
        for name, m in full_pool_baselines(Xp, yp, Xt, yt, seed, gamma).items():
            records.append({"seed": seed, "budget": 1.0, "method": name, **m})
    summary = {}
    for budget in BUDGETS:
        summary[str(budget)] = {}
        names = sorted({r["method"] for r in records if r["budget"] == budget})
        for name in names:
            rows = [r for r in records if r["budget"] == budget and r["method"] == name]
            summary[str(budget)][name] = {k: ci([r[k] for r in rows]) for k in
                                          ("fnr", "specificity", "precision", "balanced_accuracy", "auc", "common_cost_r4")}
    return records, summary


def main():
    gait = load_module("gait", ROOT / "experiments" / "gait_al_simulation.py")
    speech = load_module("speech", ROOT / "experiments" / "speech_al_simulation.py")
    # Override legacy absolute paths before invoking their read-only loaders.
    gait.DATA_DIR = ROOT / "datasets" / "gait" / "physionet-vgrf"
    gait.DEMO_PATH = gait.DATA_DIR / "demographics.txt"
    speech.DATA_PATH = ROOT / "datasets" / "pd_speech_features.csv"
    Xg, yg, _, _ = gait.load_gait_features()
    Xs, ys, _ = speech.load_patient_features(speech.DATA_PATH)
    final = {}
    for name, X, y in (("gait_classical12", Xg, yg), ("speech_raw754", Xs, ys)):
        print(f"Running {name}: n={len(y)}, p={X.shape[1]}", flush=True)
        records, summary = analyse_dataset(X, y)
        final[name] = {"n": len(y), "p": X.shape[1], "summary": summary}
        with open(OUT / f"{name}_records.pkl", "wb") as f:
            pickle.dump(records, f)
    with open(OUT / "summary.json", "w") as f:
        json.dump(final, f, indent=2)
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
