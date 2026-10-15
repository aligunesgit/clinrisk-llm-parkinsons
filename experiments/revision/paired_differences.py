#!/usr/bin/env python3
"""Reviewer 2: paired split-level differences instead of means alone.

For each comparison A vs B at 50% labelling, the per-split difference
C4(A) - C4(B) is summarised by its mean, median, IQR and the proportion of the 60
splits in which A has strictly lower cost (ties reported separately).
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from rev_common import OUT, ROOT

COMPONENT = [("joint_cal", "standard_symmetric"), ("cost_only", "standard_symmetric"),
             ("query_only", "standard_symmetric"), ("joint_cal", "cost_only"),
             ("standard_symmetric", "random_symmetric"), ("standard_threshold", "joint_cal"),
             ("joint_cal", "refer_all")]
POLICY = [("clinrisk_llm", "standard"), ("clinrisk_llm", "fixed_r4"), ("clinrisk_llm", "adaptive_rule"),
          ("adaptive_rule", "fixed_r4")]


def describe(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "q1": float(np.percentile(d, 25)), "q3": float(np.percentile(d, 75)),
            "a_better": float(np.mean(d < 0)), "tie": float(np.mean(d == 0)), "b_better": float(np.mean(d > 0))}


def main():
    out = {}
    thr = pickle.load(open(OUT / "r2_threshold_records.pkl", "rb"))
    for name, tag in (("gait", "gait_classical12"), ("speech", "speech_raw754")):
        rec = pickle.load(open(ROOT / "results" / "revised_cal" / f"{tag}_records.pkl", "rb"))
        cost = {}
        for r in rec:
            if abs(r["budget"] - 0.5) < 1e-9:
                cost.setdefault(r["method"], {})[r["seed"]] = r["common_cost_r4"]
                # refer-all: FN = 0, FP = number of test controls = FP + TN
                ctrl = r["fp"] / (1 - r["specificity"]) if r["specificity"] < 1 else None
                if ctrl is not None:
                    cost.setdefault("refer_all", {})[r["seed"]] = 100 * round(ctrl) / r["n"]
        for r in thr:
            if r["dataset"] == name and r["budget"] == 0.5 and r["method"] == "standard_threshold":
                cost.setdefault("standard_threshold", {})[r["seed"]] = r["common_cost_r4"]
        seeds = sorted(cost["joint_cal"])
        out[name] = {f"{a} vs {b}": describe([cost[a][s] for s in seeds], [cost[b][s] for s in seeds])
                     for a, b in COMPONENT}
        pol = pickle.load(open(ROOT / "results" / "clinrisk_agent" / f"{name}_records.pkl", "rb"))
        pc = {}
        for r in pol:
            if r["budget"] == 50:
                pc.setdefault(r["mode"], {})[r["seed"]] = r["cost4"]
        out[name].update({f"{a} vs {b}": describe([pc[a][s] for s in seeds], [pc[b][s] for s in seeds])
                          for a, b in POLICY})
    json.dump(out, open(OUT / "paired_differences.json", "w"), indent=2)
    for n, v in out.items():
        for k, d in v.items():
            print(f"{n:6s} {k:42s} mean {d['mean']:7.2f} median {d['median']:7.2f} "
                  f"IQR [{d['q1']:6.2f},{d['q3']:6.2f}] A<B {d['a_better']:.2f} tie {d['tie']:.2f}")


if __name__ == "__main__":
    main()
