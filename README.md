# Stress detection from wrist-worn physiological signals, evaluated on held-out people

Predictive models for stress vs non-stress from an Empatica E4 wrist device (blood volume pulse, electrodermal activity, skin temperature, accelerometer), trained and evaluated leave-one-subject-out on WESAD (Schmidt et al. 2018; 15 subjects, 1,032 sixty-second windows, 30% stress).

The held-out unit is always a whole person. Random window splits leak subject identity through resting heart rate and skin conductance level and inflate every metric; that split is not offered.

## Models

| Model | Input | Supervision | Question it answers |
|---|---|---|---|
| `gbm_features` | 16 hand-built physiological features per window | stress label | do classical HRV / EDA / motion features carry the signal? |
| `cnn_raw` | raw 4-channel signal, 32 Hz, 60 s | stress label | does a 1D CNN on the raw stream beat the features? |
| `mahalanobis_features` | features, z-scored within subject | none (fit on baseline windows only) | is stress detectable as an anomaly with no stress labels? |
| `autoencoder_raw` | raw signal, baseline windows only | none | same question on the raw stream |

## Repository layout

```
README.md
requirements.txt          numpy, scipy, pandas, scikit-learn, torch
.gitignore                keeps the dataset, features.csv and pickles out of git
src/
  load_wesad.py           reads one subject's pickle; wrist streams (BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz) and 700 Hz labels
  signals.py              signal processing: BVP peak detection and HRV, EDA tonic/phasic split, ENMO and motion artefact flag, temperature slope
  windows.py              60 s windows, 30 s stride, majority label with purity filter; emits the feature table and the raw 4-channel tensors
  train_loso.py           leave-one-subject-out for gbm_features (with inner CV threshold) and cnn_raw; writes features.csv and per-subject metrics
  ablation.py             feature family ablation, within-subject z-scoring, stress vs amusement task (reads features.csv)
  anomaly_features.py     unsupervised detection in feature space: Mahalanobis distance from baseline windows, fit on training subjects only
  anomaly.py              unsupervised detection on raw signal: autoencoder trained on baseline windows only
  make_fake_wesad.py      synthetic WESAD-shaped data for a smoke test; not for results
results/
  loso_per_subject_seed{0,1}.csv        per-subject AUROC / F1 / balanced accuracy for gbm_features and cnn_raw
  loso_summary_seed{0,1}.csv            median / min / max across subjects
  loso_per_subject_subjnorm.csv         gbm_features after within-subject z-scoring
  ablation.csv                          feature family ablation
  stress_vs_amusement.csv               stress vs amusement, raw and z-scored
  anomaly_features_mahalanobis.csv      per-subject AUROC, feature-space anomaly detection
  anomaly_per_subject_seed0.csv         per-subject AUROC and reconstruction errors, autoencoder
```

## Results (leave-one-subject-out, 15 folds; median across subjects, range in brackets)

**Stress vs non-stress (baseline + amusement)**

| Model | AUROC | F1 |
|---|---|---|
| gbm_features, raw features | 0.998 [0.913 to 1.000] | 0.894 [0.091 to 1.000] |
| gbm_features, within-subject z-scored | 1.000 [0.912 to 1.000] | 0.955 [0.231 to 1.000] |
| cnn_raw, seed 0 | 1.000 [0.831 to 1.000] | 0.840 [0.308 to 0.976] |
| cnn_raw, seed 1 | 0.994 [0.760 to 1.000] | 0.857 [0.182 to 1.000] |
| mahalanobis_features (unsupervised) | 0.956 [0.865 to 0.998] | n/a |
| autoencoder_raw (unsupervised) | 0.619 [0.071 to 0.972] | n/a |

AUROC is the primary metric because it is threshold-free. F1 for the feature model uses a threshold chosen by grouped inner cross validation on the training subjects; those thresholds range from 0.11 to 0.87 across folds, which is why F1 has a wide floor and why it is secondary. The CNN uses a fixed 0.5.

**Stress vs amusement only.** Both conditions raise arousal, so this is the discriminating test.

| Features | AUROC | F1 |
|---|---|---|
| raw | 1.000 [0.505 to 1.000] | 0.884 [0.174 to 1.000] |
| within-subject z-scored | 1.000 [0.671 to 1.000] | 0.974 [0.562 to 1.000] |

Separable for most subjects; for at least one, not at all. Amusement is about 6.5 minutes per subject, so each fold scores roughly a dozen amusement windows and the per-subject range is correspondingly noisy.

## Ablation by feature family (gbm_features, raw features)

| Features | AUROC median | AUROC min | F1 median | F1 min |
|---|---|---|---|---|
| all | 0.998 | 0.913 | 0.894 | 0.091 |
| only HRV | 0.980 | 0.759 | 0.808 | 0.524 |
| only EDA | 0.883 | 0.481 | 0.564 | 0.000 |
| only motion | 0.718 | 0.351 | 0.509 | 0.000 |
| only temperature | 0.859 | 0.053 | 0.537 | 0.000 |
| drop HRV | 0.968 | 0.586 | 0.774 | 0.091 |
| drop EDA | 0.983 | 0.800 | 0.848 | 0.562 |
| drop motion | 0.999 | 0.924 | 0.884 | 0.000 |
| drop temperature | 0.999 | 0.849 | 0.830 | 0.500 |

Heart rate and its variability carry the signal. Dropping the accelerometer features changes nothing, so wrist motion is not what the model reads. Skin temperature alone reaches 0.859 median but 0.053 minimum: it drifts over the session and acts as a clock, pointing the wrong way for some subjects. It is retained in the full model but should not be trusted on its own.

## What the failure cases say

- **S14** has AUROC 0.980 but F1 0.091 on raw features. The model orders S14's windows correctly (stress windows score higher than baseline windows) but assigns every window a low probability, so almost none cross the threshold. The cause is a scale difference: S14's heart rate, both at rest and under stress, is lower than the other fourteen subjects'. S14's stressed heart rate falls in the range where the other subjects were at rest, so a model trained on absolute values reads it as "not stressed". Label-free within-subject normalisation removes the absolute level and lifts F1 to 0.894. A population model on absolute physiology fails a person whose baseline is unusual; per-person calibration fixes it. This is the failure mode to expect in any population whose resting physiology shifts, including pregnancy.
- **S17** becomes the worst subject after normalisation (F1 0.231, AUROC 0.912): the inner CV threshold for that fold is 0.87. Threshold selection on 14 people is unstable, which is why threshold-free metrics lead.
- **Unsupervised detection** works on features (0.956) and fails on raw signal (0.619). Each raw window is z-scored per channel before the autoencoder sees it, which removes the level of heart rate and skin conductance, and the ablation shows that level is where the signal lives. The autoencoder is left with waveform shape, which reconstructs equally well stressed or not. Negative result with a known cause; kept as is.
- **CNN vs features.** At n = 15 a 1D CNN on the raw stream matches the feature model on ranking and does not beat it, the two models fail on different subjects (CNN: S15, S9, S2; features: S14), and the CNN's worst case AUROC moves with initialisation (0.831 at seed 0, 0.760 at seed 1) where the feature model has no seed dependence. An ensemble would probably help; not built.

## What this does and does not establish

A model trained on other people's wrist signals separates stress from non-stress in a person it has never seen, under a lab protocol (Trier Social Stress Test, seated baseline, amusement video). The effect of a TSST on heart rate is large, which is why the numbers are high. The TSST also involves standing and speaking, and wrist accelerometry cannot see posture, so part of the heart rate rise may be physical rather than psychological; this dataset cannot separate the two. Nothing here is evidence about free-living data, a wider age range, or pregnancy. n = 15 limits every conclusion, and the leave-one-subject-out range is reported so that limit is visible.

## Signal processing (`src/signals.py`)

- **BVP:** band-pass 0.5 to 4 Hz, peak detection with a 0.33 s refractory period and a prominence gate relative to window amplitude; inter-beat intervals outside 0.33 to 2.0 s dropped. Features: mean and SD of HR, SDNN, RMSSD, pNN50, beat count.
- **EDA:** tonic = 0.05 Hz low-pass, phasic = residual; skin conductance responses counted as phasic peaks over 0.01 uS. Features: SCL mean and slope, SCR count and mean amplitude, phasic SD.
- **Accelerometer:** ENMO magnitude, and a motion artefact fraction (share of 1 s epochs above 0.1 g).
- **Temperature:** mean and slope.

## Windowing (`src/windows.py`)

60 s windows, 30 s stride. Label by majority vote over the 700 Hz protocol labels; windows under 90% purity (transitions) dropped. Stress = label 2; non-stress = baseline (1) and amusement (3). Amusement is included by default because a model that only separates stress from sitting still is not detecting stress.

## Reproduce

Download WESAD (link below), unzip so that `WESAD/S2/S2.pkl` exists, then run in this order. Steps 1 and 2 need a GPU for the CNN (about 20 minutes each on a T4; pass `--no-cnn` to skip it). Steps 3 and 4 read `results/features.csv` and run in seconds on CPU. Step 5 needs a GPU.

```
pip install -r requirements.txt

# 1, 2. features, GBM and CNN, two seeds
#       -> results/features.csv, loso_per_subject_seed{0,1}.csv, loso_summary_seed{0,1}.csv
python src/train_loso.py --root /path/to/WESAD --out results --seed 0
python src/train_loso.py --root /path/to/WESAD --out results --seed 1

# 3. ablation, within-subject normalisation, stress vs amusement
#       -> results/ablation.csv, loso_per_subject_subjnorm.csv, stress_vs_amusement.csv
python src/ablation.py --features results/features.csv --out results

# 4. unsupervised detection in feature space
#       -> results/anomaly_features_mahalanobis.csv
python src/anomaly_features.py --features results/features.csv --out results

# 5. unsupervised detection on raw signal
#       -> results/anomaly_per_subject_seed0.csv
python src/anomaly.py --root /path/to/WESAD --out results --seed 0

# smoke test without the dataset (synthetic data; expect perfect scores, they mean nothing)
python src/make_fake_wesad.py /tmp/fake && python src/train_loso.py --root /tmp/fake --out /tmp/res --no-cnn
```

Every number in the tables above is read from the files in `results/`. Seeds are fixed; the CNN and autoencoder can differ at the third decimal across hardware.

Data: WESAD, https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html (2.1 GB; not redistributed here). Per-subject metric files are committed under `results/`; `results/features.csv` is derived from the dataset and is not.

## Reference

Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI 2018*. https://doi.org/10.1145/3242969.3242985
