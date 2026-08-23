#!/usr/bin/env python3
"""
Active Learning Simulation — Parkinson's Disease Detection
UCI "Parkinson's Disease Classification" Dataset (Sakar et al. 2018)
Patient-level pool-based AL

Fourth leg of the voice(n=32) -> gait(n=160) -> telemonitoring(n=5875 rec/42 pts)
scaling study. This dataset sits in a useful gap: 252 patients (188 PD / 64
control), each with exactly 3 sustained-phonation recordings -> patient-level
mean aggregation gives 252 well-balanced clusters of size 3, vs. telemonitoring's
42 highly imbalanced clusters of ~140. Repeated-measures leakage risk is much
smaller here, so this is the best remaining candidate (of the four datasets
tried) for a clean AL-strategy comparison.

Known risk going in: 754 raw features (MFCC, wavelet/TQWT, vocal-fold, time-
frequency) vs. 252 patients is a severe p >> n regime even before any AL
subsetting starts round 0 with ~15 labels against 754 features. RF's
max_features="sqrt" tempers this per-tree, but predict_proba on a forest
trained on a handful of patients can still be a noisy signal for margin/BALD
early in the curve. Flag this as a first-class hypothesis when reading
results, not just "does AL help" -- if all strategies converge unusually
slowly even after 60-80 labels, high dimensionality (not the AL strategy) is
the likely bottleneck.

4 query strategies (identical to al_simulation.py / gait_al_simulation.py):
  1. Random baseline
  2. Margin Sampling  (min P_max - P_second)
  3. BALD via RF      (mutual information between y and theta)
  4. Cost-Sensitive Margin  (margin x expected misclassification cost)

Metrics tracked at each query step:
  - AUC-ROC on held-out test patients
  - Sensitivity @ 90% specificity
  - Expected misclassification cost (C_FN=5, C_FP=1)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.spatial.distance import cdist
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

# -- CONFIG -----------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "pd_speech_features.csv"
OUT_DIR   = ROOT / "results" / "speech_al_simulation"

N_INITIAL   = 15      # labeled patients at round 0 (~6% of pool)
N_TRIALS    = 15      # Monte-Carlo repetitions (n=252, 754 features -> cut vs voice/gait for runtime)
TEST_FRAC   = 0.25    # held-out fraction (~63 patients)
C_FN        = 5.0     # cost of missing a PD patient
C_FP        = 1.0     # cost of a false alarm
TARGET_SPEC = 0.90    # specificity level for sensitivity@spec metric

RF_PARAMS = dict(n_estimators=80, max_features="sqrt",
                 class_weight="balanced", n_jobs=1)
# -----------------------------------------------------------------------------


# -- DATA ---------------------------------------------------------------------

def load_patient_features(path: Path):
    """
    756 recordings (252 patients x 3 reps) -> 252 patient-level vectors via
    per-patient mean. Header row 0 is a coarse feature-group label (e.g.
    "Baseline Features") that only labels the first column of each group and
    is otherwise blank -- the real column names are on row 1, hence header=1.
    """
    df = pd.read_csv(path, header=1)
    feat_cols = [c for c in df.columns if c not in ("id", "class")]

    agg = df.groupby("id")[feat_cols].mean()
    labels = df.groupby("id")["class"].first()   # verified consistent per id

    X = agg.values.astype(float)
    y = labels.loc[agg.index].values.astype(int)
    return X, y, agg.columns.tolist()


# -- QUERY STRATEGIES (identical to al_simulation.py / gait_al_simulation.py) --

def _entropy(p, eps=1e-12):
    return -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))


def strategy_random(clf, X_pool, rng, X_lab=None, **kwargs):
    return rng.randint(0, len(X_pool))


def strategy_margin(clf, X_pool, rng, X_lab=None, **kwargs):
    prob = clf.predict_proba(X_pool)
    top2 = np.sort(prob, axis=1)[:, ::-1][:, :2]
    margin = top2[:, 0] - top2[:, 1]
    return int(np.argmin(margin))


def strategy_bald(clf, X_pool, rng, X_lab=None, **kwargs):
    tree_p1 = np.array([t.predict_proba(X_pool)[:, 1] for t in clf.estimators_])
    mean_p1 = tree_p1.mean(axis=0)
    h_mean  = _entropy(mean_p1)
    mean_h  = _entropy(tree_p1).mean(axis=0)
    mi      = h_mean - mean_h
    return int(np.argmax(mi))


def strategy_cost_sensitive(clf, X_pool, rng, X_lab=None, c_fn=C_FN, c_fp=C_FP, **kwargs):
    prob = clf.predict_proba(X_pool)
    top2 = np.sort(prob, axis=1)[:, ::-1][:, :2]
    margin   = top2[:, 0] - top2[:, 1]
    p_pd     = prob[:, 1]
    exp_cost = p_pd * c_fn + (1 - p_pd) * c_fp
    score    = (1 - margin) * exp_cost
    return int(np.argmax(score))


def strategy_margin_diversity(clf, X_pool, rng, X_lab=None, top_frac=0.25, **kwargs):
    """
    Margin sampling restricted to the most-uncertain top_frac of the pool,
    then within that candidate set pick the point furthest (max min-distance)
    from the already-labeled set -- a cheap uncertainty+diversity hybrid that
    avoids the classic margin-sampling failure mode of repeatedly querying a
    tight cluster of near-duplicate uncertain points.
    """
    prob = clf.predict_proba(X_pool)
    top2 = np.sort(prob, axis=1)[:, ::-1][:, :2]
    margin = top2[:, 0] - top2[:, 1]

    k = max(1, int(np.ceil(len(margin) * top_frac)))
    cand_idx = np.argsort(margin)[:k]

    if X_lab is None or len(X_lab) == 0:
        return int(cand_idx[0])

    dists = cdist(X_pool[cand_idx], X_lab)
    min_dist = dists.min(axis=1)
    return int(cand_idx[np.argmax(min_dist)])


STRATEGIES = {
    "Random":                    strategy_random,
    "Margin Sampling":           strategy_margin,
    "BALD (RF Ensemble)":        strategy_bald,
    f"Cost-Sensitive\n(C_FN={int(C_FN)}xC_FP)": strategy_cost_sensitive,
    "Margin + Diversity":        strategy_margin_diversity,
}

STYLE = {
    "Random":                    dict(color="#888888", ls="--", lw=1.8),
    "Margin Sampling":           dict(color="#2196F3", ls="-",  lw=2.2),
    "BALD (RF Ensemble)":        dict(color="#FF9800", ls="-",  lw=2.2),
    f"Cost-Sensitive\n(C_FN={int(C_FN)}xC_FP)": dict(color="#E53935", ls="-",  lw=2.5),
    "Margin + Diversity":        dict(color="#8E24AA", ls="-",  lw=2.2),
}


# -- METRICS --------------------------------------------------------------------

def sensitivity_at_spec(y_true, y_score, target=TARGET_SPEC):
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    mask = spec >= target
    return float(tpr[mask].max()) if mask.any() else 0.0


def exp_cost(y_true, y_score, thr=0.5, c_fn=C_FN, c_fp=C_FP):
    pred = (y_score >= thr).astype(int)
    fn = ((pred == 0) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    return (fn * c_fn + fp * c_fp) / len(y_true)


# -- AL TRIAL --------------------------------------------------------------------

def run_trial(X, y, query_fn, seed):
    rng = np.random.RandomState(seed)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    pool_idx, test_idx = next(sss.split(X, y))
    X_test, y_test = X[test_idx], y[test_idx]

    pool_pd = pool_idx[y[pool_idx] == 1]
    pool_he = pool_idx[y[pool_idx] == 0]
    n_pd0   = max(1, round(N_INITIAL * (len(pool_pd) / len(pool_idx))))
    n_he0   = N_INITIAL - n_pd0

    init = list(rng.choice(pool_pd, size=min(n_pd0, len(pool_pd)), replace=False)) + \
           list(rng.choice(pool_he, size=min(n_he0, len(pool_he)), replace=False))

    labeled   = list(init)
    unlabeled = [i for i in pool_idx if i not in labeled]

    scaler = StandardScaler()
    aucs, sens_list, costs = [], [], []

    while True:
        X_lab_s  = scaler.fit_transform(X[labeled])
        X_test_s = scaler.transform(X_test)
        y_lab    = y[labeled]

        if len(np.unique(y_lab)) < 2:
            aucs.append(0.5); sens_list.append(0.0); costs.append(C_FN)
        else:
            clf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
            clf.fit(X_lab_s, y_lab)

            y_sc = clf.predict_proba(X_test_s)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_sc)
            except Exception:
                auc = 0.5

            aucs.append(auc)
            sens_list.append(sensitivity_at_spec(y_test, y_sc))
            costs.append(exp_cost(y_test, y_sc))

            if not unlabeled:
                break

            X_pool_s = scaler.transform(X[unlabeled])
            q_local  = query_fn(clf, X_pool_s, rng, X_lab=X_lab_s)
            q_global = unlabeled[q_local]
            labeled.append(q_global)
            unlabeled.remove(q_global)

        if not unlabeled:
            break

    return np.array(aucs), np.array(sens_list), np.array(costs)


# -- EXPERIMENT --------------------------------------------------------------------

def run_all(X, y):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = OUT_DIR / "partial_results.pkl"

    results = {}
    for name, fn in STRATEGIES.items():
        tag = name.replace("\n", " ")
        print(f"  [{tag}]", end="", flush=True)
        trial_a, trial_s, trial_c = [], [], []
        for seed in range(N_TRIALS):
            a, s, c = run_trial(X, y, fn, seed)
            trial_a.append(a); trial_s.append(s); trial_c.append(c)
            if (seed + 1) % 5 == 0:
                print(f" {seed+1}", end="", flush=True)
        print()
        results[name] = dict(auc=trial_a, sens=trial_s, cost=trial_c)
        with open(partial_path, "wb") as f:
            pickle.dump(results, f)   # incremental checkpoint after each strategy
        print(f"  checkpoint saved: {len(results)}/{len(STRATEGIES)} strategies done")
    return results


# -- HELPERS ---------------------------------------------------------------------

def pad_aggregate(arrays):
    max_l  = max(len(a) for a in arrays)
    padded = np.array([np.pad(a, (0, max_l - len(a)), constant_values=a[-1]) for a in arrays])
    return padded.mean(0), padded.std(0)


def labels_to_reach(arrays, threshold, n_pool):
    needed = []
    for arr in arrays:
        idx = np.where(arr >= threshold)[0]
        needed.append(int(idx[0]) + N_INITIAL if len(idx) else n_pool + N_INITIAL)
    return needed


# -- PLOTS -----------------------------------------------------------------------

def plot_learning_curves(results, out_dir, n_pool, n_total, n_pd, n_he, n_features=754):
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("auc",  "AUC-ROC",                             [0.42, 1.02], 0.90,  "AUC = 0.90",         "lower right"),
        ("sens", f"Sensitivity @ {TARGET_SPEC:.0%} Spec", [-0.05, 1.05], 0.85, "Sensitivity = 0.85", "lower right"),
        ("cost", f"Expected Cost  (C_FN={int(C_FN)}, C_FP={int(C_FP)})", None, None, None,           "upper right"),
    ]

    fig = plt.figure(figsize=(17, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    fig.suptitle(
        "Active Learning Simulation - Parkinson's Disease Detection\n"
        f"UCI Speech Features Dataset (Sakar 2018)  |  {n_total} patients ({n_pd} PD / {n_he} Healthy)  |  "
        f"{N_TRIALS} Monte-Carlo trials  |  Pool-based AL  |  {n_features} features",
        fontsize=10.5, fontweight="bold", y=1.03,
    )

    for ax, (key, ylabel, ylim, hline, hlabel, loc) in zip(axes, panels):
        for name in STRATEGIES:
            mean_, std_ = pad_aggregate(results[name][key])
            x = np.arange(N_INITIAL, N_INITIAL + len(mean_))
            st = STYLE[name]
            ax.plot(x, mean_, label=name.replace("\n", " "), **st)
            ax.fill_between(x, mean_ - std_, mean_ + std_, color=st["color"], alpha=0.10)

        ax.set_xlabel("Labeled Patients", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10, pad=5)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)
        if hline is not None:
            ax.axhline(hline, color="gray", lw=1.2, ls=":", alpha=0.65, label=hlabel)
        ax.legend(fontsize=7.5, loc=loc)

    plt.tight_layout()
    path = out_dir / "speech_al_learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_label_efficiency(results, out_dir, n_pool):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Label Efficiency - Parkinson's AL Simulation (Speech Features)",
                 fontsize=11, fontweight="bold")

    targets = [
        ("auc",  0.90, "Labels to reach AUC >= 0.90"),
        ("sens", 0.85, f"Labels to reach Sensitivity@{TARGET_SPEC:.0%} >= 0.85"),
    ]
    colors_bar = [STYLE[n]["color"] for n in STRATEGIES]

    for ax, (key, threshold, title) in zip(axes, targets):
        names  = list(STRATEGIES.keys())
        medians, q25s, q75s = [], [], []
        for name in names:
            needed = labels_to_reach(results[name][key], threshold, n_pool)
            medians.append(np.median(needed))
            q25s.append(np.percentile(needed, 25))
            q75s.append(np.percentile(needed, 75))

        x   = np.arange(len(names))
        err = np.array([np.array(medians) - q25s, np.array(q75s) - np.array(medians)])
        bars = ax.bar(x, medians, color=colors_bar, width=0.55, alpha=0.85,
                      yerr=err, capsize=5, error_kw=dict(lw=1.5, capthick=1.5))

        for bar, med in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width() / 2, med + 0.3,
                    f"{med:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylabel("Labeled Patients (median)", fontsize=10)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_ylim([0, n_pool + N_INITIAL + 2])
        ax.axhline(N_INITIAL, color="black", lw=1, ls="--", alpha=0.4, label=f"Initial pool ({N_INITIAL})")
        ax.axhline(n_pool + N_INITIAL, color="red", lw=1, ls=":", alpha=0.3, label="Max labels")
        ax.legend(fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "speech_al_label_efficiency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# -- SUMMARY TABLE -----------------------------------------------------------------

def print_summary(results, n_pool):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"RESULTS: Labels needed to reach target  (median across {N_TRIALS} trials)")
    print(sep)

    checks = [
        ("AUC >= 0.90",              "auc",  0.90),
        (f"Sensitivity@{TARGET_SPEC:.0%} >= 0.85", "sens", 0.85),
        ("Expected Cost < 1.0",     "cost", None),
    ]

    for label, key, thr in checks:
        print(f"\n  {label}:")
        rows = []
        for name in STRATEGIES:
            data = results[name][key]
            if key == "cost":
                needed = []
                for arr in data:
                    idx = np.where(arr < 1.0)[0]
                    needed.append(int(idx[0]) + N_INITIAL if len(idx) else n_pool + N_INITIAL)
            else:
                needed = labels_to_reach(data, thr, n_pool)
            med = np.median(needed)
            q25, q75 = np.percentile(needed, 25), np.percentile(needed, 75)
            pct = np.mean([v < n_pool + N_INITIAL for v in needed]) * 100
            rows.append((name.replace("\n", " "), med, q25, q75, pct))
        rows.sort(key=lambda r: r[1])
        for name, med, q25, q75, pct in rows:
            star = " *" if pct == 100 else ""
            print(f"    {name:<42}  {med:4.0f} patients  [{q25:.0f}-{q75:.0f}]  {pct:.0f}% trials{star}")

    print(f"\n  Final Expected Cost (all labels, median):")
    rows = []
    for name in STRATEGIES:
        final_costs = [arr[-1] for arr in results[name]["cost"]]
        rows.append((name.replace("\n", " "), np.median(final_costs)))
    rows.sort(key=lambda r: r[1])
    for name, c in rows:
        print(f"    {name:<42}  {c:.3f}")

    print(f"\n{sep}")
    print("INTERPRETATION:")
    print("  - Lower = fewer labels needed = more label-efficient strategy")
    print("  - IQR [25th-75th percentile] shows variance across random seeds")
    print(f"  - Cost-sensitive strategy penalizes FN (missed PD) {int(C_FN)}x more than FP")
    print("  - 754 features vs. ~189-patient pool is a p>>n regime -- if ALL")
    print("    strategies plateau slowly/similarly, dimensionality is the likely")
    print("    bottleneck, not the query strategy choice")
    print(sep)


# -- MAIN --------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Parkinson Active Learning Simulation - Speech Features (Sakar 2018)")
    print("=" * 60)

    print("\nLoading + aggregating to patient level...")
    X, y, feat_names = load_patient_features(DATA_PATH)
    n_pool_approx = round(len(X) * (1 - TEST_FRAC)) - N_INITIAL

    print(f"  {len(X)} patients  |  PD: {y.sum()}  healthy: {(y==0).sum()}")
    print(f"  {X.shape[1]} features (patient-level mean of 3 recordings)")
    print(f"  Test set: ~{len(X) - round(len(X)*(1-TEST_FRAC))} patients  |  Pool: ~{n_pool_approx} queryable")

    print(f"\nRunning {N_TRIALS} trials x {len(STRATEGIES)} strategies...")
    print(f"  Dots = 5-trial checkpoint\n")
    results = run_all(X, y)

    print_summary(results, n_pool_approx + N_INITIAL)

    p1 = plot_learning_curves(results, OUT_DIR, n_pool_approx + N_INITIAL, len(X), int(y.sum()), int((y==0).sum()))
    p2 = plot_label_efficiency(results, OUT_DIR, n_pool_approx + N_INITIAL)

    print(f"\nOutputs:")
    print(f"  {p1}")
    print(f"  {p2}")
    print("\nDone.")


if __name__ == "__main__":
    main()
