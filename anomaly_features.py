"""Unsupervised anomaly detection in feature space.

Fit a Gaussian to BASELINE windows of the training subjects (no stress labels used),
score the held-out subject's windows by Mahalanobis distance from that baseline
physiology, and report AUROC against the stress label. Features are z-scored
within subject first (label-free).

Usage:
    python src/anomaly_features.py --features results/features.csv --out results
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score

from ablation import subject_zscore
from train_loso import FEATURE_COLS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = subject_zscore(pd.read_csv(a.features)).fillna(0)
    rows = []
    for sid in sorted(df.sid.unique()):
        tr = df[(df.sid != sid) & (df.condition == 1)]
        te = df[df.sid == sid]
        cov = EmpiricalCovariance().fit(tr[FEATURE_COLS])
        rows.append({"sid": sid, "model": "mahalanobis_features_baseline_only", "n": len(te),
                     "auroc": roc_auc_score(te.y, cov.mahalanobis(te[FEATURE_COLS]))})
    r = pd.DataFrame(rows)
    r.to_csv(Path(a.out) / "anomaly_features_mahalanobis.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(f"median AUROC {r.auroc.median():.3f} [{r.auroc.min():.3f}-{r.auroc.max():.3f}]")
