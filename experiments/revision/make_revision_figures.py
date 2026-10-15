#!/usr/bin/env python3
"""Figures for the IJMS revision: prevalence swap (comment 3) and decision curves (comment 2)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rev_common import OUT, ROOT

# Manuscript figure folder when run inside the authors' workspace; results/revision otherwise.
FIG = ROOT / "manuscripts" / "IJMS" / "figures"
FIG = FIG if FIG.exists() else OUT
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"standard_symmetric": "#7f7f7f", "cost_only": "#4c78a8", "joint_cal": "#e45756",
       "fullpool_cost_threshold": "#54a24b"}
LAB = {"standard_symmetric": "Standard AL", "cost_only": "Cost-only", "joint_cal": "Joint cAL",
       "fullpool_cost_threshold": "Full-pool cost threshold"}


def prevalence_figure():
    d = json.load(open(OUT / "prevalence_matched.json"))
    panels = [("gait_original", "Gait, original\n(prevalence 0.55)"),
              ("gait_at_speech_prevalence", "Gait, subsampled\n(prevalence 0.75)"),
              ("speech_original", "Speech, original\n(prevalence 0.75)"),
              ("speech_at_gait_prevalence", "Speech, subsampled\n(prevalence 0.55)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width, x = 0.2, np.arange(len(panels))
    for j, arm in enumerate(COL):
        means = [d[p][arm][0] for p, _ in panels]
        lo = [d[p][arm][0] - d[p][arm][1] for p, _ in panels]
        hi = [d[p][arm][2] - d[p][arm][0] for p, _ in panels]
        ax.bar(x + (j - 1.5) * width, means, width, yerr=[lo, hi], color=COL[arm], label=LAB[arm],
               capsize=2, error_kw={"lw": 0.8})
    for i, (p, _) in enumerate(panels):
        ra = d[p]["refer_all"][0]
        ax.hlines(ra, i - 0.45, i + 0.45, colors="k", linestyles="--", lw=1)
        ax.text(i, d[p]["standard_symmetric"][2] + 3, f"−{d[p]['joint_vs_standard_reduction_pct']:.1f}%",
                ha="center", fontsize=8)
    ax.plot([], [], "k--", lw=1, label="Refer all (trivial)")
    ax.set_xticks(x); ax.set_xticklabels([t for _, t in panels])
    ax.set_ylabel("$C_4$ (points per 100 participants)")
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_ylim(0, 90)
    fig.tight_layout()
    fig.savefig(FIG / "fig_revision_prevalence_swap.pdf"); fig.savefig(OUT / "fig_revision_prevalence_swap.png", dpi=200)


def dca_figure():
    d = json.load(open(OUT / "burden_dca.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
    for ax, name in zip(axes, ("gait", "speech")):
        dc = d[name]["decision_curve"]; t = np.array(dc["thresholds"])
        m, sd = np.array(dc["nb_symmetric_svm"]), np.array(dc["nb_symmetric_svm_sd"])
        ax.plot(t, m, color="#4c78a8", label="Full-pool symmetric SVM (Platt)")
        ax.fill_between(t, m - sd, m + sd, color="#4c78a8", alpha=0.15, lw=0)
        ax.plot(t, dc["nb_refer_all"], "k--", lw=1, label="Refer all")
        ax.axhline(0, color="0.6", lw=0.8, label="Refer none")
        ax.axvline(0.2, color="#e45756", lw=0.8, ls=":")
        ops = d[name]["operating_points"]; pi = d[name]["study_prevalence"]
        for lab, mk, dx in (("standard_symmetric@0.5", "o", -0.015), ("joint_cal@0.5", "s", 0.0),
                            ("fullpool_cost_threshold", "^", 0.015)):
            s, sp = ops[lab]["sensitivity"], ops[lab]["specificity"]
            ax.plot(0.2 + dx, s * pi - (1 - sp) * (1 - pi) * 0.25, mk, color="#e45756", ms=5,
                    label={"standard_symmetric@0.5": "Standard AL (50%)", "joint_cal@0.5": "Joint cAL (50%)",
                           "fullpool_cost_threshold": "Cost threshold (full)"}[lab])
        ax.set_title(f"{name.capitalize()} (prevalence {pi:.2f})")
        ax.set_xlabel("Threshold probability $p_t$")
        ax.set_ylim(-0.05, pi + 0.05)
    axes[0].set_ylabel("Net benefit")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "fig_revision_decision_curves.pdf"); fig.savefig(OUT / "fig_revision_decision_curves.png", dpi=200)


if __name__ == "__main__":
    prevalence_figure()
    dca_figure()
    print("figures written")


def oof_figure():
    d = json.load(open(OUT / "r2_oof.json"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for row, name in enumerate(("gait", "speech")):
        for col, m in enumerate(("sens", "spec")):
            ax = axes[row, col]
            for mode, color, lab in (("standard", "#7f7f7f", "Standard AL"), ("clinrisk_llm", "#e45756", "ClinRisk-LLM")):
                ents = [e for e in d[name][mode] if m in e]
                x = np.array([float(e["budget_bin"].split("-")[0]) + 0.05 for e in ents])
                o = np.array([e[m]["oof_mean"] for e in ents]); sd = np.array([e[m]["oof_sd"] for e in ents])
                t = np.array([e[m]["test_mean"] for e in ents])
                ax.plot(x, o, color=color, label=f"{lab}: OOF estimate")
                ax.fill_between(x, o - sd, o + sd, color=color, alpha=0.15, lw=0)
                ax.plot(x, t, color=color, ls="--", label=f"{lab}: test value")
            ax.set_title(f"{name.capitalize()}: {'sensitivity' if m == 'sens' else 'specificity'}")
            ax.set_ylim(0, 1.05)
            if row == 1:
                ax.set_xlabel("Fraction of pool labeled")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_revision_oof_stability.pdf"); fig.savefig(OUT / "fig_revision_oof_stability.png", dpi=200)


if __name__ == "__main__":
    oof_figure()
