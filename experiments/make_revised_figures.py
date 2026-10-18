#!/usr/bin/env python3
"""Publication figures for the leakage-resistant cAL ablation."""
from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revised_cal"
FIGS = ROOT / "manuscripts" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
ORDER = ["standard_symmetric", "query_only", "cost_only", "joint_cal"]
LABEL = {"standard_symmetric": "Standard", "query_only": "Query-only",
         "cost_only": "Cost-only", "joint_cal": "Joint cAL",
         "fullpool_symmetric": "Symmetric", "fullpool_cost": "Class-weighted",
         "fullpool_cost_threshold": "Cost threshold"}
COLOR = {"standard_symmetric": "#4C72B0", "query_only": "#DD8452",
         "cost_only": "#55A868", "joint_cal": "#8172B2",
         "fullpool_symmetric": "#4C72B0", "fullpool_cost": "#8172B2",
         "fullpool_cost_threshold": "#C44E52"}


def load(name):
    with open(RESULTS / f"{name}_records.pkl", "rb") as f:
        return pickle.load(f)


def vals(records, method, budget, key):
    return np.array([float(r[key]) for r in records
                     if r["method"] == method and r["budget"] == budget])


def save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(FIGS / name)


def learning_summary(gait, speech):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), sharex=True)
    metrics = [("common_cost_r4", r"Common cost $C_4$ (points per 100)"),
               ("fnr", "False-negative rate"), ("specificity", "Specificity")]
    for row, (records, row_name) in enumerate(((gait, "Gait: classical-12"),
                                                (speech, "Speech: raw-753"))):
        for col, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            # Main-text curves retain only the three arms needed to separate
            # class weighting from its incremental query contribution.
            for method in ("standard_symmetric", "cost_only", "joint_cal"):
                means, lo, hi = [], [], []
                budgets = np.array(sorted({float(r["budget"]) for r in records
                                           if r["method"] == method}))
                for budget in budgets:
                    x = vals(records, method, budget, key)
                    sem = x.std(ddof=1) / np.sqrt(len(x))
                    means.append(x.mean()); lo.append(x.mean()-sem); hi.append(x.mean()+sem)
                means, lo, hi = map(np.asarray, (means, lo, hi))
                ax.plot(budgets*100, means, marker="o", lw=1.5, ms=3,
                        color=COLOR[method], label=LABEL[method])
                ax.fill_between(budgets*100, lo, hi, color=COLOR[method], alpha=.12)
            # Symmetric and class-weighted full-pool values equal the 100%
            # endpoints; only the non-redundant cost-threshold reference remains.
            ref = "fullpool_cost_threshold"
            ref_mean = vals(records, ref, 1.0, key).mean()
            ax.axhline(ref_mean, color=COLOR[ref], ls=":", lw=1.15,
                       label="Full pool: Cost threshold")
            ax.set_title(title); ax.grid(alpha=.25, lw=.5); ax.set_xticks([10,25,50,75,100])
            if row == 1: ax.set_xlabel("Pool labeled (\%)")
        axes[row, 0].set_ylabel(row_name)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=[0, .06, 1, 1])
    save(fig, "fig_revised_budget_curves.pdf")


def tradeoff(gait, speech):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25))
    for ax, records, title in zip(axes, (gait, speech),
                                  ("(a) Gait: classical-12", "(b) Speech: raw-753")):
        for method in ORDER:
            for budget, marker, alpha in ((0.5, "o", 1), (1.0, "s", .72)):
                sp = vals(records, method, budget, "specificity")
                se = 1 - vals(records, method, budget, "fnr")
                ax.errorbar(sp.mean(), se.mean(), xerr=sp.std(), yerr=se.std(),
                            marker=marker, ms=5, capsize=2, lw=.8, color=COLOR[method], alpha=alpha,
                            label=f"{LABEL[method]} ({int(100*budget)}%)")
        ax.plot([0, 1], [1, 0], ls=":", color=".65", lw=.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.grid(alpha=.2, lw=.5)
        ax.set_xlabel("Specificity"); ax.set_ylabel("Sensitivity (1 - FNR)"); ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.16))
    fig.tight_layout(rect=[0, .13, 1, 1])
    save(fig, "fig_revised_tradeoff.pdf")


def fullpool(gait, speech):
    methods = ["fullpool_symmetric", "fullpool_cost", "fullpool_cost_threshold"]
    display_colors = {"fullpool_symmetric": "#777777",
                      "fullpool_cost": "#E68632",
                      "fullpool_cost_threshold": "#3B75AF"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    for ax, records, title in zip(axes, (gait, speech),
                                  ("(a) Gait: classical-12", "(b) Speech: raw-753")):
        means, lows, highs = [], [], []
        rng = np.random.default_rng(20260823)
        for method in methods:
            x = vals(records, method, 1.0, "common_cost_r4")
            boot = rng.choice(x, size=(5000, len(x)), replace=True).mean(axis=1)
            means.append(x.mean())
            lows.append(np.percentile(boot, 2.5))
            highs.append(np.percentile(boot, 97.5))
        y = np.arange(len(methods))[::-1]
        for yi, method, mean, low, high in zip(y, methods, means, lows, highs):
            ax.errorbar(mean, yi, xerr=np.array([[mean-low], [high-mean]]),
                        fmt="o", ms=6.5, capsize=3, elinewidth=1.5,
                        color=display_colors[method], zorder=3)
            ax.annotate(f"{mean:.1f}", (mean, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=display_colors[method])
        ax.axvline(means[0], color=display_colors["fullpool_symmetric"],
                   ls="--", lw=.9, alpha=.65, zorder=1)
        ax.set_yticks(y, [LABEL[m] for m in methods])
        ax.set_xlabel(r"$C_4$ (points per 100)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.22, lw=.5)
        ax.set_ylim(-.65, 2.65)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_revised_fullpool_cost.pdf")


def confusion(gait, speech):
    fig, axes = plt.subplots(2, 2, figsize=(5.7, 5.0))
    for row, (records, name) in enumerate(((gait, "Gait"), (speech, "Speech"))):
        for col, method in enumerate(("standard_symmetric", "joint_cal")):
            rows = [r for r in records if r["method"] == method and r["budget"] == 0.5]
            fn, fp = np.mean([r["fn"] for r in rows]), np.mean([r["fp"] for r in rows])
            # Stratification fixes test-class counts within each dataset.
            positives = rows[0]["fn"] / rows[0]["fnr"] if rows[0]["fnr"] else (26 if name == "Gait" else 56)
            positives = 26 if name == "Gait" else 56
            negatives = rows[0]["n"] - positives
            cm = np.array([[negatives-fp, fp], [fn, positives-fn]])
            ax = axes[row, col]; ax.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=.75)
            for i, labrow in enumerate((("TN", "FP"), ("FN", "TP"))):
                for j, lab in enumerate(labrow):
                    ax.text(j, i, f"{lab}\n{cm[i,j]:.1f}", ha="center", va="center", fontsize=9)
            ax.set_xticks([0,1], ["Control", "PD"]); ax.set_yticks([0,1], ["Control", "PD"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{name}: {LABEL[method]} (50%)")
    fig.tight_layout()
    save(fig, "fig_revised_confusion_matrices.pdf")


if __name__ == "__main__":
    g, s = load("gait_classical12"), load("speech_raw754")
    learning_summary(g, s); tradeoff(g, s); fullpool(g, s); confusion(g, s)
