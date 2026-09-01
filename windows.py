"""Cut each subject into fixed windows; emit (a) a feature table for the GBM and
(b) raw multichannel tensors for the CNN, with subject id kept on every row so
splits are always subject-wise.

Task: stress (label 2) vs non-stress (baseline 1; amusement 3 optional).
Windows are labelled by majority vote over the 700 Hz label stream and dropped if
the majority class is under 90% of the window (transition windows).
"""
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from load_wesad import FS, FS_LABEL, duration_s, slice_stream
from signals import acc_features, bvp_ibi, eda_features, hrv_features, temp_features

WIN_S = 60
STRIDE_S = 30
RAW_FS = 32          # all channels resampled to this for the CNN
RAW_CH = ["bvp", "eda", "temp", "acc_mag"]


def window_label(lab: np.ndarray, purity: float = 0.9):
    vals, counts = np.unique(lab, return_counts=True)
    k = int(np.argmax(counts))
    return int(vals[k]) if counts[k] / len(lab) >= purity else -1


def _resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    from math import gcd
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


def _zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-8 else 1.0)


def subject_windows(sub: dict, include_amusement: bool = True):
    """Yield (features: dict, raw: (C, T) float32, y: int, t0: float)."""
    T = duration_s(sub)
    t0 = 0.0
    while t0 + WIN_S <= T:
        t1 = t0 + WIN_S
        y_raw = window_label(slice_stream(sub["label"], FS_LABEL, t0, t1))
        keep = y_raw in (1, 2, 3) if include_amusement else y_raw in (1, 2)
        if keep:
            bvp = slice_stream(sub["BVP"], FS["BVP"], t0, t1)
            eda = slice_stream(sub["EDA"], FS["EDA"], t0, t1)
            tmp = slice_stream(sub["TEMP"], FS["TEMP"], t0, t1)
            acc = slice_stream(sub["ACC"], FS["ACC"], t0, t1)

            feats = {"sid": sub["sid"], "t0": t0, "condition": y_raw}
            feats.update(hrv_features(bvp_ibi(bvp, FS["BVP"])))
            feats.update(eda_features(eda, FS["EDA"]))
            feats.update(acc_features(acc, FS["ACC"]))
            feats.update(temp_features(tmp, FS["TEMP"]))

            n = WIN_S * RAW_FS
            raw = np.stack([
                _zscore(_resample(bvp, FS["BVP"], RAW_FS)[:n]),
                _zscore(_resample(eda, FS["EDA"], RAW_FS)[:n]),
                _zscore(_resample(tmp, FS["TEMP"], RAW_FS)[:n]),
                _zscore(_resample(np.linalg.norm(acc, axis=1), FS["ACC"], RAW_FS)[:n]),
            ]).astype(np.float32)

            yield feats, raw, int(y_raw == 2), t0
        t0 += STRIDE_S


def build_dataset(subjects: list[dict], include_amusement: bool = True):
    rows, raws, ys = [], [], []
    for sub in subjects:
        for feats, raw, y, _ in subject_windows(sub, include_amusement):
            rows.append(feats); raws.append(raw); ys.append(y)
    df = pd.DataFrame(rows)
    df["y"] = ys
    return df, np.stack(raws), np.asarray(ys, dtype=np.int64)
