# ClinRisk-LLM for Parkinson's Disease Biosignals

This repository contains the analysis code and frozen policy used in the study
"LLM-Guided Adaptive Cost Calibration in Active Learning for Parkinson's
Disease Biosignals."

## Contents

- `experiments/revised_cal_analysis.py`: component-wise active-learning ablation
- `experiments/clinrisk_agent_analysis.py`: adaptive-policy comparison
- `experiments/clinrisk_llm_policy.json`: frozen LLM-derived policy provenance
- `experiments/make_*`: scripts used to generate the manuscript figures
- `results/`: aggregate summary outputs reported in the manuscript

## Data

The datasets are not redistributed in this repository. Download the Gait in
Parkinson's Disease data from PhysioNet and place it in
`datasets/gait/physionet-vgrf/`. Download the Parkinson's Disease
Classification data from the UCI Machine Learning Repository and save it as
`datasets/pd_speech_features.csv`.

- PhysioNet: https://physionet.org/content/gaitpdb/1.0.0/
- UCI: https://doi.org/10.24432/C5MS4X

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/revised_cal_analysis.py
python experiments/clinrisk_agent_analysis.py
python experiments/make_revised_figures.py
python experiments/make_clinrisk_figures.py
python experiments/make_clinrisk_framework.py
```

The analyses use participant-level splitting and a common 4:1
false-negative:false-positive evaluation loss. The LLM is not called during
evaluation. The frozen deterministic policy is loaded from
`experiments/clinrisk_llm_policy.json`.
