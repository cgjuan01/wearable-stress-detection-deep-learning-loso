"""Load WESAD wrist (Empatica E4) signals and study-protocol labels.

WESAD layout (Schmidt et al. 2018): one pickle per subject, WESAD/S<n>/S<n>.pkl,
subjects S2..S17 (S1 and S12 do not exist -> 15 subjects).

data['signal']['wrist'] : ACC (32 Hz, 3 axes, in 1/64 g), BVP (64 Hz), EDA (4 Hz, uS), TEMP (4 Hz, C)
data['label']           : 700 Hz, aligned with the chest device; wrist streams start at the same t=0
labels: 0 undefined, 1 baseline, 2 stress, 3 amusement, 4 meditation, 5-7 ignore
"""
from pathlib import Path
import pickle
import numpy as np

SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]
FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
FS_LABEL = 700
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load_subject(root: str | Path, sid: str) -> dict:
    p = Path(root) / sid / f"{sid}.pkl"
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    w = d["signal"]["wrist"]
    out = {
        "sid": sid,
        "ACC": np.asarray(w["ACC"], dtype=np.float32) / 64.0,  # -> g
        "BVP": np.asarray(w["BVP"], dtype=np.float32).ravel(),
        "EDA": np.asarray(w["EDA"], dtype=np.float32).ravel(),
        "TEMP": np.asarray(w["TEMP"], dtype=np.float32).ravel(),
        "label": np.asarray(d["label"], dtype=np.int16).ravel(),
    }
    return out


def duration_s(sub: dict) -> float:
    """Usable duration = shortest stream, in seconds."""
    return min(
        len(sub["BVP"]) / FS["BVP"],
        len(sub["ACC"]) / FS["ACC"],
        len(sub["EDA"]) / FS["EDA"],
        len(sub["TEMP"]) / FS["TEMP"],
        len(sub["label"]) / FS_LABEL,
    )


def slice_stream(x: np.ndarray, fs: int, t0: float, t1: float) -> np.ndarray:
    return x[int(round(t0 * fs)) : int(round(t1 * fs))]
