"""Leave-one-subject-out evaluation of two predictive models.

Model A: HistGradientBoosting on the hand-built physiological features.
Model B: 1D CNN on the raw 4-channel 32 Hz window (needs torch).

The held-out unit is a whole person. A random window split would leak subject
identity through resting HR and skin conductance level and inflate every metric;
that split is deliberately not offered here.

Usage:
    python train_loso.py --root /path/to/WESAD --out results/ [--no-cnn] [--no-amusement]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from load_wesad import SUBJECTS, load_subject
from windows import build_dataset

FEATURE_COLS = ["hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "n_beats",
                "scl_mean", "scl_slope", "scr_count", "scr_amp_mean", "phasic_std",
                "enmo_mean", "enmo_std", "motion_frac", "temp_mean", "temp_slope"]


def metrics(y, p, thr=0.5):
    out = {"n": int(len(y)), "pos_frac": float(np.mean(y))}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auroc"] = np.nan
    out["f1"] = float(f1_score(y, p >= thr))
    out["bal_acc"] = float(balanced_accuracy_score(y, p >= thr))
    return out


# ---------------- Model A: features + GBM ----------------

def loso_gbm(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold, cross_val_predict
    res = []
    for sid in sorted(df.sid.unique()):
        tr, te = df[df.sid != sid], df[df.sid == sid]
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                             max_iter=300, random_state=seed)
        # threshold chosen on training subjects only, via grouped inner CV
        p_in = cross_val_predict(clf, tr[FEATURE_COLS], tr.y, groups=tr.sid,
                                 cv=GroupKFold(5), method="predict_proba")[:, 1]
        grid = np.linspace(0.05, 0.95, 91)
        thr = grid[np.argmax([f1_score(tr.y, p_in >= t) for t in grid])]
        clf.fit(tr[FEATURE_COLS], tr.y)
        p = clf.predict_proba(te[FEATURE_COLS])[:, 1]
        res.append({"sid": sid, "model": "gbm_features", "thr": float(thr), **metrics(te.y.values, p, thr)})
    return pd.DataFrame(res)


# ---------------- Model B: raw signal + 1D CNN ----------------

def loso_cnn(X: np.ndarray, y: np.ndarray, sids: np.ndarray, seed: int = 0,
             epochs: int = 15, lr: float = 1e-3, device: str | None = None) -> pd.DataFrame:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self, c_in=4):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(c_in, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))
        def forward(self, x):
            return self.f(x).squeeze(1)

    res = []
    for sid in sorted(np.unique(sids)):
        tr, te = sids != sid, sids == sid
        Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr], dtype=torch.float32)
        Xte = torch.tensor(X[te])
        pos_w = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-3)])
        net = Net(X.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
        n = len(Xtr)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
                opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xte.to(device))).cpu().numpy()
        res.append({"sid": sid, "model": "cnn_raw", **metrics(y[te], p)})
    return pd.DataFrame(res)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("model")[["auroc", "f1", "bal_acc"]]
    return pd.concat([g.median().add_suffix("_median"), g.min().add_suffix("_min"),
                      g.max().add_suffix("_max")], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--no-amusement", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    subs = [load_subject(a.root, s) for s in SUBJECTS]
    df, X, y = build_dataset(subs, include_amusement=not a.no_amusement)
    df.to_csv(Path(a.out) / "features.csv", index=False)
    print(f"{len(df)} windows from {df.sid.nunique()} subjects; stress fraction {y.mean():.2f}")

    res = [loso_gbm(df, a.seed)]
    if not a.no_cnn:
        res.append(loso_cnn(X, y, df.sid.values, a.seed, epochs=a.epochs))
    res = pd.concat(res)
    res.to_csv(Path(a.out) / f"loso_per_subject_seed{a.seed}.csv", index=False)
    summ = summarise(res)
    summ.to_csv(Path(a.out) / f"loso_summary_seed{a.seed}.csv")
    print(summ.round(3))
