"""Unsupervised anomaly detection on wrist physiology.

An autoencoder is trained on BASELINE windows from the training subjects only.
On the held-out subject, reconstruction error is the anomaly score; stress
windows should reconstruct worse. AUROC of that score against the stress label
is reported per subject. No label enters training.

Usage:
    python anomaly.py --root /path/to/WESAD --out results/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load_wesad import SUBJECTS, load_subject
from windows import build_dataset


def loso_autoencoder(X, y, cond, sids, seed=0, epochs=30, lr=1e-3, device=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    C, T = X.shape[1], X.shape[2]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(C, 16, 15, stride=4, padding=7), nn.ReLU(),
                nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.ReLU(),
                nn.Conv1d(32, 8, 5, stride=2, padding=2))
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(8, 32, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 9, stride=4, padding=4, output_padding=3), nn.ReLU(),
                nn.ConvTranspose1d(16, C, 15, stride=4, padding=7, output_padding=3))
        def forward(self, x):
            r = self.dec(self.enc(x))
            return r[..., :T]

    res = []
    for sid in sorted(np.unique(sids)):
        tr = (sids != sid) & (cond == 1)        # baseline windows of other subjects only
        te = sids == sid
        Xtr = torch.tensor(X[tr]); Xte = torch.tensor(X[te])
        net = AE().to(device); opt = torch.optim.Adam(net.parameters(), lr=lr)
        n = len(Xtr)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, 64):
                xb = Xtr[perm[i:i + 64]].to(device)
                opt.zero_grad(); loss = ((net(xb) - xb) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            err = ((net(Xte.to(device)) - Xte.to(device)) ** 2).mean(dim=(1, 2)).cpu().numpy()
        yt = y[te]
        auc = roc_auc_score(yt, err) if len(np.unique(yt)) == 2 else np.nan
        res.append({"sid": sid, "model": "autoencoder_baseline_only", "n": int(te.sum()),
                    "auroc": float(auc), "err_baseline": float(err[yt == 0].mean()),
                    "err_stress": float(err[yt == 1].mean()) if (yt == 1).any() else np.nan})
    return pd.DataFrame(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    subs = [load_subject(a.root, s) for s in SUBJECTS]
    df, X, y = build_dataset(subs, include_amusement=True)
    res = loso_autoencoder(X, y, df.condition.values, df.sid.values, a.seed, a.epochs)
    res.to_csv(Path(a.out) / f"anomaly_per_subject_seed{a.seed}.csv", index=False)
    print(res.round(3)); print("median AUROC:", round(res.auroc.median(), 3))
