"""Signal processing for wrist physiological streams.

Everything here is classical DSP with scipy; no learning. Each function takes one
window of one stream and returns a dict of named features, so the feature table is
auditable column by column.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass(x: np.ndarray, fs: int, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def lowpass(x: np.ndarray, fs: int, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, hi / nyq, btype="low")
    return filtfilt(b, a, x)


# ---------- BVP -> inter-beat intervals -> HR / HRV ----------

def bvp_ibi(bvp: np.ndarray, fs: int = 64) -> np.ndarray:
    """Return inter-beat intervals in seconds from a BVP window.

    Band-pass 0.5-4 Hz (30-240 bpm), then peak detection with a refractory
    period of 0.33 s (max 180 bpm) and a prominence gate relative to the window's
    own amplitude. IBIs outside 0.33-2.0 s are dropped as artefacts.
    """
    if len(bvp) < fs * 2:
        return np.array([])
    x = bandpass(bvp - np.mean(bvp), fs, 0.5, 4.0)
    prom = 0.3 * np.std(x)
    peaks, _ = find_peaks(x, distance=int(0.33 * fs), prominence=prom)
    ibi = np.diff(peaks) / fs
    return ibi[(ibi > 0.33) & (ibi < 2.0)]


def hrv_features(ibi: np.ndarray) -> dict:
    if len(ibi) < 4:
        return {"hr_mean": np.nan, "hr_std": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                "pnn50": np.nan, "n_beats": len(ibi)}
    hr = 60.0 / ibi
    d = np.diff(ibi)
    return {
        "hr_mean": float(hr.mean()),
        "hr_std": float(hr.std()),
        "sdnn": float(ibi.std() * 1000),
        "rmssd": float(np.sqrt(np.mean(d ** 2)) * 1000),
        "pnn50": float(np.mean(np.abs(d) > 0.05)),
        "n_beats": int(len(ibi)),
    }


# ---------- EDA tonic / phasic ----------

def eda_features(eda: np.ndarray, fs: int = 4) -> dict:
    """Simple tonic/phasic split: tonic = 0.05 Hz low-pass, phasic = residual.
    Skin-conductance responses counted as phasic peaks > 0.01 uS with >= 1 s spacing.
    (cvxEDA is the better decomposition; swap it in if time allows.)
    """
    if len(eda) < fs * 5:
        return {k: np.nan for k in ["scl_mean", "scl_slope", "scr_count", "scr_amp_mean", "phasic_std"]}
    tonic = lowpass(eda, fs, 0.05)
    phasic = eda - tonic
    peaks, props = find_peaks(phasic, height=0.01, distance=fs)
    t = np.arange(len(eda)) / fs
    slope = np.polyfit(t, tonic, 1)[0] if len(t) > 1 else np.nan
    return {
        "scl_mean": float(tonic.mean()),
        "scl_slope": float(slope),
        "scr_count": int(len(peaks)),
        "scr_amp_mean": float(props["peak_heights"].mean()) if len(peaks) else 0.0,
        "phasic_std": float(phasic.std()),
    }


# ---------- Accelerometer -> motion / artefact ----------

def acc_features(acc: np.ndarray, fs: int = 32) -> dict:
    """acc: (n, 3) in g. Motion magnitude and a crude artefact flag: fraction of
    1-second epochs whose ENMO exceeds 0.1 g (heavy wrist movement corrupts BVP)."""
    mag = np.linalg.norm(acc, axis=1)
    enmo = np.clip(mag - 1.0, 0, None)
    n_ep = len(enmo) // fs
    ep = enmo[: n_ep * fs].reshape(n_ep, fs).mean(axis=1) if n_ep else np.array([0.0])
    return {
        "enmo_mean": float(enmo.mean()),
        "enmo_std": float(enmo.std()),
        "motion_frac": float(np.mean(ep > 0.1)),
    }


def temp_features(temp: np.ndarray, fs: int = 4) -> dict:
    t = np.arange(len(temp)) / fs
    slope = np.polyfit(t, temp, 1)[0] if len(temp) > 1 else np.nan
    return {"temp_mean": float(temp.mean()), "temp_slope": float(slope)}
