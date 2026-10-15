"""Shared loaders and helpers for the IJMS revision analyses.

Reuses the frozen manuscript code (revised_cal_analysis.py and
clinrisk_agent_analysis.py) so every new analysis runs the identical
classifier, query rules and evaluation loss.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = load_module("revised_cal_analysis", EXP / "revised_cal_analysis.py")
ca = load_module("clinrisk_agent_analysis", EXP / "clinrisk_agent_analysis.py")


def load_gait():
    """Gait-12 features, labels and PhysioNet sub-study (Ga/Ju/Si) per participant."""
    gait = load_module("gait", EXP / "gait_al_simulation.py")
    gait.DATA_DIR = ROOT / "datasets" / "gait" / "physionet-vgrf"
    gait.DEMO_PATH = gait.DATA_DIR / "demographics.txt"
    X, y, _, dropped = gait.load_gait_features()
    labels = gait.load_demographics(gait.DEMO_PATH)
    ids = [p.stem.rsplit("_", 1)[0] for p in sorted(gait.DATA_DIR.glob("*_01.txt"))]
    ids = [i for i in ids if i in labels.index and i not in dropped]
    assert len(ids) == len(y)
    study = np.array([i[:2] for i in ids])
    return X, y, study


def load_speech():
    speech = load_module("speech", EXP / "speech_al_simulation.py")
    speech.DATA_PATH = ROOT / "datasets" / "pd_speech_features.csv"
    X, y, _ = speech.load_patient_features(speech.DATA_PATH)
    return X, y


def load_little():
    """UCI Parkinsons (Little et al. 2007): recording-level features + subject ids."""
    df = pd.read_csv(ROOT / "datasets" / "voice" / "uci-parkinson-voice" / "parkinsons.data")
    subject = df["name"].str.extract(r"(S\d+)")[0].values
    y = df["status"].values.astype(int)
    X = df.drop(columns=["name", "status"]).values.astype(float)
    return X, y, subject


def refer_all_cost(y_test) -> float:
    """C4 of the trivial policy that labels every participant PD (FN=0, FP=#controls)."""
    return 100.0 * float((np.asarray(y_test) == 0).sum()) / len(y_test)


def mean_ci(values, n_boot=5000, seed=7):
    a = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    boots = a[rng.randint(0, len(a), (n_boot, len(a)))].mean(1)
    return float(a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
