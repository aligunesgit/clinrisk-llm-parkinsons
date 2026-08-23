#!/usr/bin/env python3
"""Create the ClinRisk-LLM framework flowchart used in the manuscript."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscripts" / "figures" / "fig_clinrisk_framework.pdf"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.25, 4.65))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

COL = {
    "blue": "#DCEAF7", "blue_edge": "#3B75AF",
    "orange": "#FCE7D4", "orange_edge": "#D87827",
    "green": "#DDEEDC", "green_edge": "#4F8A4B",
    "purple": "#E9E0F2", "purple_edge": "#7B5AA6",
    "gray": "#EEEEEE", "gray_edge": "#777777",
    "ink": "#263238",
}


def box(x, y, w, h, text, fc, ec, fontsize=8.1, lw=1.15, dashed=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.08,rounding_size=0.10",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fontsize, color=COL["ink"], linespacing=1.18)
    return p


def arrow(x1, y1, x2, y2, color="#55636B", rad=0, label=None,
          label_xy=None, dashed=False):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=10, linewidth=1.1, color=color,
                        linestyle="--" if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        lx, ly = label_xy if label_xy else ((x1+x2)/2, (y1+y2)/2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7,
                color=color, backgroundcolor="white")


# Three-stage architecture inspired by a modular systems diagram.
def stage(x, y, w, h, title, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.06,rounding_size=0.08",
                               facecolor=fc, edgecolor=ec, linewidth=1.05,
                               linestyle=(0, (6, 3)), alpha=.42))
    ax.plot([x, x+w], [y+h-.46, y+h-.46], color=ec, lw=1.0,
            linestyle=(0, (6, 3)))
    ax.text(x+w/2, y+h-.23, title, ha="center", va="center",
            fontsize=8.1, weight="bold", color=COL["ink"])


stage(.25, 1.12, 3.05, 5.95, "Stage 1: Data Preparation",
      "#EAF3FA", COL["blue_edge"])
stage(3.58, 1.12, 5.40, 5.95, "Stage 2: LLM-Guided Active Learning",
      "#FFF3E8", COL["orange_edge"])
stage(9.27, 1.12, 2.48, 5.95, "Stage 3: Evaluation",
      "#EDF5EC", COL["green_edge"])

# Stage 1: two modalities and a participant-level split.
box(.60, 5.35, 2.35, .78, "Gait + speech biosignals",
    COL["blue"], COL["blue_edge"], 8.3)
ax.plot([.82, 1.04, 1.25, 1.48, 1.70],
        [5.57, 5.88, 5.48, 5.82, 5.60], color=COL["blue_edge"], lw=1.1)
ax.plot([1.87, 2.04, 2.19, 2.34, 2.52, 2.72],
        [5.62, 5.75, 5.49, 5.78, 5.50, 5.67], color=COL["blue_edge"], lw=1.0)
box(.60, 3.85, 2.35, .82, "Participant-level\npreprocessing", "white", COL["blue_edge"], 8.2)
box(.60, 2.18, 2.35, .82, "70/30 split", "white", COL["blue_edge"], 8.4)
arrow(1.78, 5.35, 1.78, 4.67)
arrow(1.78, 3.85, 1.78, 3.00)

# Stage 2: policy above a concise AL loop.
box(3.95, 5.48, 2.00, .82, "Clinical safety\nspecification", COL["purple"], COL["purple_edge"], 7.9)
box(6.55, 5.48, 1.98, .82, "Frozen LLM\npolicy", COL["purple"], COL["purple_edge"], 8.0)
arrow(5.95, 5.89, 6.55, 5.89, COL["purple_edge"])

box(3.95, 3.72, 1.95, .88, "OOF pool\nmonitoring", COL["green"], COL["green_edge"], 8.1)
box(6.15, 3.72, 2.38, .88, "Select cost + query\nweights  (rₜ, aₜ)", COL["purple"], COL["purple_edge"], 8.0)
arrow(5.90, 4.16, 6.15, 4.16)
arrow(7.54, 5.48, 7.40, 4.60, COL["purple_edge"])

box(3.95, 2.05, 2.20, .88, "Cost-sensitive SVM", COL["orange"], COL["orange_edge"], 8.2)
box(6.50, 2.05, 2.03, .88, "Risk-aware query", COL["orange"], COL["orange_edge"], 8.2)
arrow(7.34, 3.72, 5.55, 2.93, COL["purple_edge"], rad=.10)
arrow(6.15, 2.49, 6.50, 2.49)
arrow(2.95, 2.59, 3.95, 4.16, COL["blue_edge"], rad=-.12,
      label="development pool", label_xy=(3.63, 3.15))
arrow(7.52, 2.05, 4.88, 3.72, COL["green_edge"], rad=-.28,
      label="oracle labels / update", label_xy=(6.15, 1.63))

# Stage 3: model and test remain separate until the terminal comparison.
box(9.62, 5.20, 1.78, .82, "Final model", COL["orange"], COL["orange_edge"], 8.3)
box(9.62, 3.62, 1.78, .82, "Held-out\ntest set", COL["gray"], COL["gray_edge"], 8.1)
box(9.52, 1.87, 1.98, .92, "C₄, FNR,\nspecificity + BA",
    COL["green"], COL["green_edge"], 8.0)
arrow(8.53, 2.49, 9.62, 5.61, COL["orange_edge"], rad=-.16)
arrow(10.51, 5.20, 10.51, 2.79, COL["green_edge"])
arrow(10.51, 3.62, 10.51, 2.79, COL["gray_edge"], dashed=True)
ax.plot([2.95, 3.18, 9.35], [2.41, .93, .93], color=COL["gray_edge"],
        lw=1.05, linestyle="--")
ax.plot([9.35, 9.35], [.93, 4.03], color=COL["gray_edge"],
        lw=1.05, linestyle="--")
arrow(9.35, 4.03, 9.62, 4.03, COL["gray_edge"], dashed=True)
ax.text(6.30, .99, "test partition (isolated)", ha="center", va="center",
        fontsize=7.0, color=COL["gray_edge"], backgroundcolor="white")

ax.text(6.0, .48, "Test outcomes are excluded from policy generation, training, and query selection.",
        ha="center", fontsize=7.7, color=COL["gray_edge"], style="italic")

ax.set_ylim(.25, 7.35)
fig.savefig(OUT, bbox_inches="tight", pad_inches=.05)
plt.close(fig)
print(OUT)
