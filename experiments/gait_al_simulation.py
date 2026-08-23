#!/usr/bin/env python3
"""
Active Learning Simulation — Parkinson's Disease Detection
Physionet Gait VGRF Dataset (patient-level pool-based AL)

Continuation of the UCI Voice AL simulation (see al_simulation.py). Open
thread from 2026-06-24: n=32 (voice) is too small for AL strategies to
diverge; this dataset has n≈150-165 patients, where a real gap is expected.

Feature extraction: threshold-crossing gait-event detection on the two
Total Force channels (L/R), producing classic Hausdorff/Plotnik-style
gait variability + asymmetry features — this exact dataset is the source
of that literature, so these are the features known to separate PD from
controls (not generic time-series stats).

4 query strategies (identical to the voice simulation):
  1. Random baseline
  2. Margin Sampling  (min P_max - P_second)
  3. BALD via RF      (mutual information between y and θ)
  4. Cost-Sensitive Margin  (margin × expected misclassification cost)

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
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "datasets" / "gait" / "physionet-vgrf"
DEMO_PATH   = DATA_DIR / "demographics.txt"
OUT_DIR     = ROOT / "results" / "gait_al_simulation"

FORCE_THRESHOLD = 20.0   # N — contact vs. swing threshold on Total Force channel
MIN_STRIDE, MAX_STRIDE = 0.6, 2.5   # s — plausible stride-time bounds (artifact filter)
MIN_SWING,  MAX_SWING  = 0.2, 1.2   # s
MIN_STANCE, MAX_STANCE = 0.3, 1.8   # s
MIN_VALID_STRIDES = 5    # per foot, else subject dropped

N_INITIAL   = 8        # labeled patients at round 0 (larger pool than voice study)
N_TRIALS    = 20       # Monte-Carlo repetitions (n=160 pool means ~112 RF fits/trial — cut from 50 for runtime)
TEST_FRAC   = 0.25     # held-out fraction
C_FN        = 5.0      # cost of missing a PD patient
C_FP        = 1.0      # cost of a false alarm
TARGET_SPEC = 0.90     # specificity level for sensitivity@spec metric

RF_PARAMS = dict(n_estimators=100, max_features="sqrt",
                 class_weight="balanced", n_jobs=1)
# ──────────────────────────────────────────────────────────────────────────────


# ── DEMOGRAPHICS ────────────────────────────────────────────────────────────────

def load_demographics(path: Path):
    """
    Manual tab-split parse: the file has inconsistent trailing-tab counts
    per row (159 well-formed cols vs. 30-wide rows further down), which
    breaks pandas' C parser. Only the first 3 columns (ID, Study, Group)
    are needed here.
    """
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            rows.append((parts[0].strip(), parts[2].strip()))
    df = pd.DataFrame(rows, columns=["ID", "Group"])
    df["Group"] = pd.to_numeric(df["Group"], errors="coerce")
    df = df.dropna(subset=["Group"])
    df["label"] = (df["Group"] == 1).astype(int)   # 1=PD, 2=Control
    return df.set_index("ID")["label"]


# ── GAIT EVENT DETECTION ─────────────────────────────────────────────────────────

def _detect_events(force, time):
    """Threshold-crossing heel-strike / toe-off detection on a force channel."""
    contact = (force > FORCE_THRESHOLD).astype(int)
    d = np.diff(contact)
    strike_t = time[np.where(d == 1)[0] + 1]
    toeoff_t = time[np.where(d == -1)[0] + 1]

    # Align: drop a leading toe-off with no preceding strike, and a trailing
    # strike with no following toe-off (incomplete stance at trial start/end).
    if len(toeoff_t) and len(strike_t) and toeoff_t[0] < strike_t[0]:
        toeoff_t = toeoff_t[1:]
    if len(strike_t) and len(toeoff_t) and strike_t[-1] > toeoff_t[-1]:
        strike_t = strike_t[:-1]
    n = min(len(strike_t), len(toeoff_t))
    strike_t, toeoff_t = strike_t[:n], toeoff_t[:n]
    return strike_t, toeoff_t


def _foot_features(force, time):
    strike_t, toeoff_t = _detect_events(force, time)
    if len(strike_t) < MIN_VALID_STRIDES + 1:
        return None

    stance = toeoff_t - strike_t
    stride = np.diff(strike_t)
    swing  = strike_t[1:] - toeoff_t[:-1]

    stance = stance[(stance >= MIN_STANCE) & (stance <= MAX_STANCE)]
    stride_mask = (stride >= MIN_STRIDE) & (stride <= MAX_STRIDE)
    swing_mask  = (swing  >= MIN_SWING)  & (swing  <= MAX_SWING)
    stride = stride[stride_mask]
    swing  = swing[swing_mask]

    if len(stride) < MIN_VALID_STRIDES or len(swing) < MIN_VALID_STRIDES:
        return None

    return dict(
        n_strikes=len(strike_t),
        stride_mean=stride.mean(), stride_cv=stride.std() / stride.mean(),
        swing_mean=swing.mean(),   swing_cv=swing.std() / swing.mean(),
        stance_mean=stance.mean() if len(stance) else np.nan,
        duration=time[-1] - time[0],
    )


def extract_subject_features(path: Path):
    if path.stat().st_size == 0:
        return None   # truncated download — empty file
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 19:
        return None
    time    = raw[:, 0]
    force_l = raw[:, 17]
    force_r = raw[:, 18]

    fl = _foot_features(force_l, time)
    fr = _foot_features(force_r, time)
    if fl is None or fr is None:
        return None

    cadence = (fl["n_strikes"] + fr["n_strikes"]) / ((time[-1] - time[0]) / 60.0)
    gait_asymmetry = 100.0 * abs(np.log(fl["swing_mean"] / fr["swing_mean"]))

    return {
        "cadence":            cadence,
        "gait_asymmetry":     gait_asymmetry,
        "stride_time_L":      fl["stride_mean"],
        "stride_time_R":      fr["stride_mean"],
        "stride_cv_L":        fl["stride_cv"],
        "stride_cv_R":        fr["stride_cv"],
        "swing_time_L":       fl["swing_mean"],
        "swing_time_R":       fr["swing_mean"],
        "swing_cv_L":         fl["swing_cv"],
        "swing_cv_R":         fr["swing_cv"],
        "stance_time_L":      fl["stance_mean"],
        "stance_time_R":      fr["stance_mean"],
    }


def load_gait_features():
    labels = load_demographics(DEMO_PATH)
    files = sorted(DATA_DIR.glob("*_01.txt"))

    rows, ids, dropped = [], [], []
    for path in files:
        subj_id = path.stem.rsplit("_", 1)[0]
        if subj_id not in labels.index:
            continue
        feats = extract_subject_features(path)
        if feats is None:
            dropped.append(subj_id)
            continue
        rows.append(feats)
        ids.append(subj_id)

    df = pd.DataFrame(rows, index=ids)
    y = labels.loc[ids].values.astype(int)
    X = df.values.astype(float)
    return X, y, df.columns.tolist(), dropped


# ── QUERY STRATEGIES (identical to al_simulation.py) ────────────────────────────

def _entropy(p, eps=1e-12):
    return -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))


def strategy_random(clf, X_pool, rng):
    return rng.randint(0, len(X_pool))


def strategy_margin(clf, X_pool, rng):
    prob = clf.predict_proba(X_pool)
    top2 = np.sort(prob, axis=1)[:, ::-1][:, :2]
    margin = top2[:, 0] - top2[:, 1]
    return int(np.argmin(margin))


def strategy_bald(clf, X_pool, rng):
    tree_p1 = np.array([t.predict_proba(X_pool)[:, 1] for t in clf.estimators_])
    mean_p1 = tree_p1.mean(axis=0)
    h_mean  = _entropy(mean_p1)
    mean_h  = _entropy(tree_p1).mean(axis=0)
    mi      = h_mean - mean_h
    return int(np.argmax(mi))


def strategy_cost_sensitive(clf, X_pool, rng, c_fn=C_FN, c_fp=C_FP):
    prob = clf.predict_proba(X_pool)
    top2 = np.sort(prob, axis=1)[:, ::-1][:, :2]
    margin   = top2[:, 0] - top2[:, 1]
    p_pd     = prob[:, 1]
    exp_cost = p_pd * c_fn + (1 - p_pd) * c_fp
    score    = (1 - margin) * exp_cost
    return int(np.argmax(score))


STRATEGIES = {
    "Random":                    strategy_random,
    "Margin Sampling":           strategy_margin,
    "BALD (RF Ensemble)":        strategy_bald,
    f"Cost-Sensitive\n(C_FN={int(C_FN)}×C_FP)": strategy_cost_sensitive,
}

STYLE = {
    "Random":                    dict(color="#888888", ls="--", lw=1.8),
    "Margin Sampling":           dict(color="#2196F3", ls="-",  lw=2.2),
    "BALD (RF Ensemble)":        dict(color="#FF9800", ls="-",  lw=2.2),
    f"Cost-Sensitive\n(C_FN={int(C_FN)}×C_FP)": dict(color="#E53935", ls="-",  lw=2.5),
}


# ── METRICS ────────────────────────────────────────────────────────────────────

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


# ── AL TRIAL ──────────────────────────────────────────────────────────────────

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
            q_local  = query_fn(clf, X_pool_s, rng)
            q_global = unlabeled[q_local]
            labeled.append(q_global)
            unlabeled.remove(q_global)

        if not unlabeled:
            break

    return np.array(aucs), np.array(sens_list), np.array(costs)


# ── EXPERIMENT ────────────────────────────────────────────────────────────────

def run_all(X, y):
    import pickle
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


# ── HELPERS ───────────────────────────────────────────────────────────────────

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


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_learning_curves(results, out_dir, n_pool, n_total, n_pd, n_he):
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
        "Active Learning Simulation — Parkinson's Disease Detection\n"
        f"Gait VGRF Dataset (Physionet)  |  {n_total} patients ({n_pd} PD / {n_he} Healthy)  |  "
        f"{N_TRIALS} Monte-Carlo trials  |  Pool-based AL",
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
    path = out_dir / "gait_al_learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_label_efficiency(results, out_dir, n_pool):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Label Efficiency — Parkinson's AL Simulation (Gait VGRF)",
                 fontsize=11, fontweight="bold")

    targets = [
        ("auc",  0.90, "Labels to reach AUC ≥ 0.90"),
        ("sens", 0.85, f"Labels to reach Sensitivity@{TARGET_SPEC:.0%} ≥ 0.85"),
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
    path = out_dir / "gait_al_label_efficiency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def print_summary(results, n_pool):
    sep = "═" * 72
    print(f"\n{sep}")
    print("RESULTS: Labels needed to reach target  (median across 50 trials)")
    print(sep)

    checks = [
        ("AUC ≥ 0.90",              "auc",  0.90),
        (f"Sensitivity@{TARGET_SPEC:.0%} ≥ 0.85", "sens", 0.85),
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
            star = " ★" if pct == 100 else ""
            print(f"    {name:<42}  {med:4.0f} patients  [{q25:.0f}–{q75:.0f}]  {pct:.0f}% trials{star}")

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
    print("  • Lower = fewer labels needed = more label-efficient strategy")
    print("  • IQR [25th–75th percentile] shows variance across random seeds")
    print(f"  • Cost-sensitive strategy penalizes FN (missed PD) {int(C_FN)}× more than FP")
    print(sep)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Parkinson Active Learning Simulation — Gait VGRF")
    print("=" * 60)

    print("\nExtracting gait features from VGRF signals...")
    X, y, feat_names, dropped = load_gait_features()
    n_pool_approx = round(len(X) * (1 - TEST_FRAC)) - N_INITIAL

    print(f"  {len(X)} patients  |  PD: {y.sum()}  healthy: {(y==0).sum()}")
    print(f"  {len(dropped)} subjects dropped (insufficient valid strides): {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
    print(f"  {X.shape[1]} features: {feat_names}")
    print(f"  Test set: ~{len(X) - round(len(X)*(1-TEST_FRAC))} patients  |  Pool: ~{n_pool_approx} queryable")

    print(f"\nRunning {N_TRIALS} trials × {len(STRATEGIES)} strategies...")
    print(f"  Dots = 10-trial checkpoint\n")
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
