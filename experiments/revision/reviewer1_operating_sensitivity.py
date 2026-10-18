"""Post-hoc utility and burden of existing operating points; no model retuning.

Uses saved split-level error rates. Transported scenarios assume unchanged
sensitivity/specificity. Percentiles describe split variability, not population CIs.
"""
import csv
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/revision'
METHODS = ('standard_symmetric', 'cost_only', 'joint_cal')


def main():
    utility, burden, summary = [], [], {}
    for modality, tag in [('gait', 'gait_classical12'), ('speech', 'speech_raw754')]:
        with (ROOT / f'results/revised_cal/{tag}_records.pkl').open('rb') as f:
            records = pickle.load(f)
        arms = {m: sorted([r for r in records if r['method'] == m and
                          abs(r['budget'] - .5) < 1e-9], key=lambda r: r['seed'])
                for m in METHODS}
        assert all(len(v) == 60 for v in arms.values())
        assert len({tuple(r['seed'] for r in v) for v in arms.values()}) == 1
        for ratio in (1, 2, 4, 8, 16):
            for method, rows in arms.items():
                costs = np.array([100 * (ratio*r['fn'] + r['fp'])/r['n'] for r in rows])
                utility.append(dict(modality=modality, method=method, fn_fp_ratio=ratio,
                                    mean_cost=costs.mean(), split_p025=np.percentile(costs, 2.5),
                                    split_p975=np.percentile(costs, 97.5)))
        for prevalence in (.01, .05, .10, .30):
            scenarios = {}
            for method, rows in arms.items():
                tp = np.array([(1-r['fnr'])*1000*prevalence for r in rows])
                fp = np.array([(1-r['specificity'])*1000*(1-prevalence) for r in rows])
                scenarios[method] = (tp, fp)
                for metric, values in [('detected', tp), ('unnecessary_tests', fp),
                                       ('total_tests', tp+fp), ('net_benefit_0.2', (tp-.25*fp)/1000)]:
                    burden.append(dict(modality=modality, prevalence=prevalence, method=method,
                                       metric=metric, mean=values.mean(),
                                       split_p025=np.percentile(values, 2.5),
                                       split_p975=np.percentile(values, 97.5)))
            a, b = scenarios['joint_cal'], scenarios['standard_symmetric']
            dtp, dfp = (a[0]-b[0]).mean(), (a[1]-b[1]).mean()
            summary[f'{modality}@{prevalence}'] = dict(
                additional_cases=dtp, additional_unnecessary_tests=dfp,
                additional_total_tests=dtp+dfp,
                total_tests_per_additional_case=(dtp+dfp)/dtp,
                unnecessary_tests_per_additional_case=dfp/dtp,
                break_even_fn_fp_ratio_vs_standard=dfp/dtp)
    for name, rows in [('reviewer1_utility_sensitivity.csv', utility),
                       ('reviewer1_burden_sensitivity.csv', burden)]:
        with (OUT/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (OUT/'reviewer1_incremental_burden.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
