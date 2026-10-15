#!/usr/bin/env python3
"""Reviewer 2 analyses.

threshold : symmetric AL (standard and random acquisition) with a post-hoc 4:1
            cost threshold tuned by out-of-fold scores on the *labelled set only*,
            at every budget (does simple threshold tuning match cost-sensitive AL?).
kmeans    : the four adaptive policies rerun with the kernel k-means diversity
            stage of the component ablation instead of rank spreading.
oof       : stability and bias of the three-fold OOF sensitivity/specificity used
            as calibration signals, against the test-set sensitivity/specificity of
            the same symmetric SVM, across the annotation budget.
pca       : speech component arms after unsupervised PCA (90% variance, fitted on
            the development pool) to check for high-dimensional instability.

Usage: python3 reviewer2_analyses.py {threshold|kmeans|oof|pca|all}
"""
from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rev_common import OUT, ca, load_gait, load_speech, mean_ci, rc

DATA = {}


def data(name):
    if not DATA:
        DATA["gait"] = load_gait()[:2]
        DATA["speech"] = load_speech()
    return DATA[name]


def prepared(name, seed):
    X, y = data(name)
    pool, test = rc.split_indices(y, seed)
    sc = StandardScaler().fit(X[pool])
    return sc.transform(X[pool]), y[pool], sc.transform(X[test]), y[test]


# ── threshold-tuned symmetric AL ───────────────────────────────────────────────
def labelled_set(Xp, yp, seed, budget, random_query, gamma):
    """Replicates rc.run_al for symmetric training and returns the labelled indices."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(yp))
    labeled, unlabeled = list(order[:rc.INITIAL_N]), list(order[rc.INITIAL_N:])
    while len(set(yp[labeled])) < 2 and unlabeled:
        labeled.append(unlabeled.pop())
    budget = max(budget, len(labeled))
    while len(labeled) < budget and unlabeled:
        clf = SVC(C=1, kernel="rbf", gamma=gamma, class_weight={0: 1.0, 1: 1.0}).fit(Xp[labeled], yp[labeled])
        take = min(rc.BATCH, budget - len(labeled), len(unlabeled))
        if random_query:
            local = rng.choice(len(unlabeled), take, replace=False)
        else:
            local = rc.query_batch(clf, Xp[unlabeled], False, gamma, seed + len(labeled))[:take]
        selected = [unlabeled[i] for i in local]
        labeled.extend(selected)
        unlabeled = [i for i in unlabeled if i not in set(selected)]
    return np.asarray(labeled)


def tuned_threshold(XL, yL, seed, gamma):
    k = int(min(5, np.bincount(yL, minlength=2).min()))
    proto = SVC(C=1, kernel="rbf", gamma=gamma, class_weight={0: 1, 1: 1})
    if k < 2:
        return 0.0, proto.fit(XL, yL)
    oof = cross_val_predict(proto, XL, yL, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                            method="decision_function")
    fpr, tpr, thr = roc_curve(yL, oof)
    prev = yL.mean()
    thr = np.clip(thr, -1e6, 1e6)
    return float(thr[np.argmin(rc.RISK * prev * (1 - tpr) + (1 - prev) * fpr)]), proto.fit(XL, yL)


def threshold_job(args):
    name, seed = args
    Xp, yp, Xt, yt = prepared(name, seed)
    gamma = rc.gamma_scale(Xp)
    rows = []
    for frac in rc.BUDGETS:
        budget = max(rc.INITIAL_N, round(len(yp) * frac))
        for label, rq in (("standard_threshold", False), ("random_threshold", True)):
            L = labelled_set(Xp, yp, seed, budget, rq, gamma)
            t, clf = tuned_threshold(Xp[L], yp[L], seed, gamma)
            s = clf.decision_function(Xt)
            rows.append({"dataset": name, "seed": seed, "budget": float(frac), "method": label,
                         "threshold": t, **rc.metrics(yt, (s > t).astype(int), s)})
    return rows


# ── adaptive policies with kernel k-means diversity ───────────────────────────
GAMMA = {}


def kmeans_query(clf, X, a, rng):
    d = clf.decision_function(X)
    score = np.where(d <= 0, np.abs(d) / a, np.abs(d))
    cand = np.argsort(score)[:min(rc.CANDIDATES, len(X))]
    if len(cand) <= rc.BATCH:
        return cand
    groups = rc.kernel_kmeans(X[cand], rc.BATCH, GAMMA["g"], int(rng.randint(1 << 30)))
    chosen = []
    for c in range(rc.BATCH):
        local = np.flatnonzero(groups == c)
        if len(local):
            chosen.append(cand[local[np.argmin(score[cand[local]])]])
    return np.asarray(chosen, dtype=int)


def kmeans_job(args):
    name, seed = args
    Xp, yp, Xt, yt = prepared(name, seed)
    GAMMA["g"] = rc.gamma_scale(Xp)
    ca.query = kmeans_query
    out = []
    for mode in ("standard", "fixed_r4", "adaptive_rule", "clinrisk_llm"):
        r = next(z for z in ca.run_one(Xp, yp, Xt, yt, seed, mode) if z["budget"] == 50)
        out.append({"dataset": name, "seed": seed, "mode": mode, **r})
    return out


# ── OOF signal stability ──────────────────────────────────────────────────────
TRACE = []
_ORIG_OOF = ca.oof_context


def traced_oof(X, y):
    sens, spec = _ORIG_OOF(X, y)
    Xt, yt = GAMMA["test"]
    clf = SVC(C=1, kernel="rbf", gamma=ca.gamma(X), class_weight={0: 1, 1: 1}).fit(X, y)
    p = clf.predict(Xt)
    TRACE.append({"n_lab": len(y), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                  "oof_sens": sens, "oof_spec": spec,
                  "test_sens": float(p[yt == 1].mean()), "test_spec": float(1 - p[yt == 0].mean())})
    return sens, spec


def oof_job(args):
    name, seed = args
    Xp, yp, Xt, yt = prepared(name, seed)
    GAMMA["test"] = (Xt, yt)
    ca.oof_context = traced_oof
    out = []
    for mode in ("standard", "clinrisk_llm"):
        TRACE.clear()
        ca.run_one(Xp, yp, Xt, yt, seed, mode)
        out.extend({"dataset": name, "seed": seed, "mode": mode, "frac": t["n_lab"] / len(yp), **t} for t in TRACE)
    return out


def summarise_oof(rows):
    res = {}
    for name in ("gait", "speech"):
        res[name] = {}
        for mode in ("standard", "clinrisk_llm"):
            res[name][mode] = []
            for lo in np.arange(0.0, 1.0, 0.1):
                z = [r for r in rows if r["dataset"] == name and r["mode"] == mode and lo <= r["frac"] < lo + 0.1]
                if not z:
                    continue
                ok = [r for r in z if r["oof_sens"] is not None and r["oof_spec"] is not None]
                ent = {"budget_bin": f"{lo:.1f}-{lo + 0.1:.1f}", "n_steps": len(z),
                       "unavailable_frac": 1 - len(ok) / len(z),
                       "min_class_median": float(np.median([min(r["n_pos"], r["n_neg"]) for r in z]))}
                for m in ("sens", "spec"):
                    if ok:
                        o = np.array([r[f"oof_{m}"] for r in ok]); t = np.array([r[f"test_{m}"] for r in ok])
                        ent[m] = {"oof_mean": float(o.mean()), "oof_sd": float(o.std()), "test_mean": float(t.mean()),
                                  "bias": float((o - t).mean()), "mae": float(np.abs(o - t).mean()),
                                  "corr": float(np.corrcoef(o, t)[0, 1]) if o.std() and t.std() else None}
                if ok:
                    for cut in (0.50, 0.60, 0.70):
                        agree = np.mean([(r["oof_spec"] < cut) == (r["test_spec"] < cut) for r in ok])
                        ent[f"spec_lt_{cut:.2f}_agreement"] = float(agree)
                res[name][mode].append(ent)
    return res


# ── speech with PCA ───────────────────────────────────────────────────────────
def pca_job(args):
    name, seed = args
    Xp, yp, Xt, yt = prepared(name, seed)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=seed).fit(Xp)
    sc = StandardScaler().fit(pca.transform(Xp))
    Xp, Xt = sc.transform(pca.transform(Xp)), sc.transform(pca.transform(Xt))
    gamma = rc.gamma_scale(Xp)
    budget = max(rc.INITIAL_N, round(len(yp) * 0.5))
    row = {"dataset": name, "seed": seed, "n_components": int(pca.n_components_)}
    for arm in ("standard_symmetric", "query_only", "cost_only", "joint_cal"):
        row[arm] = rc.run_al(Xp, yp, Xt, yt, seed, budget, *rc.METHODS[arm], gamma)
    for k, v in rc.full_pool_baselines(Xp, yp, Xt, yt, seed, gamma).items():
        row[k] = v
    return row


def run(kind):
    jobs = [(n, s) for n in ("gait", "speech") for s in range(rc.N_SPLITS)]
    if kind == "pca":
        jobs = [("speech", s) for s in range(rc.N_SPLITS)]
    fn = {"threshold": threshold_job, "kmeans": kmeans_job, "oof": oof_job, "pca": pca_job}[kind]
    with Pool(8) as p:
        res = p.map(fn, jobs)
    rows = [r for x in res for r in (x if isinstance(x, list) else [x])]
    pickle.dump(rows, open(OUT / f"r2_{kind}_records.pkl", "wb"))
    if kind == "threshold":
        summ = {n: {m: {str(b): {k: mean_ci([r[k] for r in rows if r["dataset"] == n and r["method"] == m
                                             and r["budget"] == b])
                                 for k in ("common_cost_r4", "fnr", "specificity", "balanced_accuracy")}
                        for b in map(float, rc.BUDGETS)}
                    for m in ("standard_threshold", "random_threshold")} for n in ("gait", "speech")}
    elif kind == "kmeans":
        summ = {n: {m: {k: mean_ci([r[k] for r in rows if r["dataset"] == n and r["mode"] == m])
                        for k in ("cost4", "fnr", "specificity", "balanced_accuracy", "risk_weight")}
                    for m in ("standard", "fixed_r4", "adaptive_rule", "clinrisk_llm")} for n in ("gait", "speech")}
    elif kind == "oof":
        summ = summarise_oof(rows)
    else:
        summ = {"n_components": mean_ci([r["n_components"] for r in rows]),
                **{k: {m: mean_ci([r[k][m] for r in rows]) for m in
                       ("common_cost_r4", "fnr", "specificity", "balanced_accuracy", "auc")}
                   for k in rows[0] if isinstance(rows[0][k], dict)}}
    json.dump(summ, open(OUT / f"r2_{kind}.json", "w"), indent=2)
    print(kind, "done", flush=True)


if __name__ == "__main__":
    for k in (["threshold", "kmeans", "oof", "pca"] if sys.argv[1] == "all" else [sys.argv[1]]):
        run(k)
