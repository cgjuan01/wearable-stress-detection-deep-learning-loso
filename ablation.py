"""Feature-family ablation, within-subject normalisation, and the stress-vs-amusement task.

Reads results/features.csv written by train_loso.py; writes results/ablation.csv,
results/loso_per_subject_subjnorm.csv, results/stress_vs_amusement.csv.

Usage:
    python src/ablation.py --features results/features.csv --out results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train_loso
from train_loso import FEATURE_COLS, loso_gbm

FAMILIES = {
    "hrv": ["hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "n_beats"],
    "eda": ["scl_mean", "scl_slope", "scr_count", "scr_amp_mean", "phasic_std"],
    "motion": ["enmo_mean", "enmo_std", "motion_frac"],
    "temp": ["temp_mean", "temp_slope"],
}


def subject_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free within-subject normalisation: z-score each feature against the
    subject's own session distribution."""
    d = df.copy()
    d[FEATURE_COLS] = df.groupby("sid")[FEATURE_COLS].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return d


def summ(r: pd.DataFrame, name: str) -> dict:
    return {"setting": name,
            "auroc_median": r.auroc.median(), "auroc_min": r.auroc.min(), "auroc_max": r.auroc.max(),
            "f1_median": r.f1.median(), "f1_min": r.f1.min(), "f1_max": r.f1.max()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    df = pd.read_csv(a.features)

    # 1. ablation by feature family (raw features)
    rows = []
    settings = [("all", FEATURE_COLS)]
    settings += [(f"only_{k}", v) for k, v in FAMILIES.items()]
    settings += [(f"drop_{k}", [c for c in FEATURE_COLS if c not in v]) for k, v in FAMILIES.items()]
    for name, cols in settings:
        train_loso.FEATURE_COLS = cols
        rows.append(summ(loso_gbm(df), name))
    train_loso.FEATURE_COLS = FEATURE_COLS
    abl = pd.DataFrame(rows); abl.to_csv(out / "ablation.csv", index=False)
    print("Ablation:\n", abl.round(3).to_string(index=False))

    # 2. within-subject normalisation
    dfn = subject_zscore(df)
    r_norm = loso_gbm(dfn); r_norm.to_csv(out / "loso_per_subject_subjnorm.csv", index=False)
    print("\nWithin-subject z-scored:\n", pd.DataFrame([summ(r_norm, "subjnorm")]).round(3).to_string(index=False))

    # 3. stress vs amusement only (both arousal conditions)
    rows = []
    for name, d in [("raw", df), ("subjnorm", dfn)]:
        r = loso_gbm(d[d.condition.isin([2, 3])].reset_index(drop=True))
        rows.append(summ(r, f"stress_vs_amusement_{name}"))
    sva = pd.DataFrame(rows); sva.to_csv(out / "stress_vs_amusement.csv", index=False)
    print("\nStress vs amusement:\n", sva.round(3).to_string(index=False))
