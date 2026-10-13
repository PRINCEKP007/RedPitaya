"""Train the real-signal ESN (separate PW and BW readouts) on Red Pitaya data.

NO-PHI VERSION: the digital reservoir is driven by every raw sample (no local
Phi measurement matrix / block / stride compression).

The Red Pitaya digitises the photodetector behind the atomic vapour cell, so
each window is REAL samples, not I/Q. One window = 3 PRI (PRI fixed at 512 us).

Data sources (DATA_SOURCE):
  "recorded" : folder written by RedPitaya_ESN_Dashboard.py "Record labelled
               windows": windows_f32.bin + labels.csv + meta.json. Labels come
               from the HackRF TX UDP reference (or the manual reference).
  "matrix"   : like TrainESN_1.py: one .npy of shape (samples_per_window, N)
               (column k = window k, Fortran order) plus PW.csv (us) and
               BW.csv (kHz), each value repeated REPEATS times.

Edit the settings below, then run:  python TrainESN_Real.py
"""
import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from rp_core import (MODEL_FORMAT, BW_FEATURE_NAMES, N_SHAPE, PW_FEATURE_NAMES, RealReservoir,
                     Readout, assemble, create_real_reservoir, estimate_polarity, extract_features,
                     ridge_solve, save_model)

np.random.seed(42)

# ============================================================
# DATA
# ============================================================
DATA_SOURCE = "recorded"                 # "recorded" or "matrix"
RECORD_DIR = "rp_dataset"                # for "recorded"
MATRIX_FILE = "X_3PRI_real.npy"          # for "matrix"
PW_CSV, BW_CSV = "PW.csv", "BW.csv"      # for "matrix": PW in us, BW in kHz
REPEATS = 10                             # for "matrix": windows per PW/BW setting
MATRIX_DECIMATION = 32                   # for "matrix": Red Pitaya decimation used

BASE_RATE = 125e6                        # STEMlab 125-14 ADC clock
PRI_us = 512.0                           # fixed TX PRI
PRI_PER_WINDOW = 3

TEST_FRACTION = 0.05                     # held-out PW/BW settings (not just windows)

# ============================================================
# FEATURES / RESERVOIR / READOUT
# ============================================================
POLARITY = "auto"            # "auto" (learned from labels), "raises" or "lowers" the detector level
EDGE_SMOOTH_US = 1.0         # smoothing for level crossings and response shape
EDGE_WINDOW_US = 10.0        # averaging on each side of a candidate pulse edge
PW_SEARCH_MARGIN = 0.10      # edge pairs searched within label PW range +/- 10%

COMPUTE_RESERVOIR = True     # False: physical features only (faster training and RX)
N_res = 300
reservoir_density = 0.03
desired_radius = 0.9
input_scale = 0.5
# No Phi compression: every one of the 6,000 samples per window is a reservoir step.

# Each parameter: every candidate feature set and alpha scored by grouped CV.
#   "pw"    PW features (edge pair, 50% crossings, threshold widths, SNR)
#   "bw"    BW features (width / position / spread of the in-pulse response)
#   "shape" in-pulse response resampled to N_SHAPE points
#   "res"   real ESN reservoir state mean and mean-square
CANDIDATE_FEATURES = {
    "PW": [["res"], ["pw"], ["pw", "res"]],
    "BW": [["res"], ["bw", "pw"], ["bw", "pw", "shape"], ["bw", "pw", "shape", "res"]],
}
ALPHA_GRID = np.logspace(-6, 3, 19)
CV_FOLDS = 5
FEATURE_CACHE = "rp_features_cache.npz"  # None to always recompute
MODEL_PATH = "rp_esn_model.npz"


# ============================================================
# LOAD DATA
# ============================================================
def load_recorded(folder):
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_win = int(meta["samples_per_window"])
    X = np.memmap(os.path.join(folder, "windows_f32.bin"), dtype="<f4", mode="r")
    if X.size % n_win:
        raise ValueError("windows_f32.bin does not hold whole windows.")
    X = X.reshape(-1, n_win)
    pw, bw = [], []
    with open(os.path.join(folder, "labels.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pw.append(float(row["pw_us"]))
            bw.append(float(row["bw_khz"]))
    pw, bw = np.array(pw), np.array(bw)
    n = min(len(pw), X.shape[0])           # recorder may stop between the two writes
    return X[:n], pw[:n], bw[:n], float(meta["fs_hz"]), int(meta["samples_per_pri"])


def load_matrix():
    fs = BASE_RATE / MATRIX_DECIMATION
    spp = int(round(fs * PRI_us * 1e-6))
    A = np.load(MATRIX_FILE, mmap_mode="r")
    A = A.reshape(spp * PRI_PER_WINDOW, -1, order="F")
    pw = np.repeat(np.genfromtxt(PW_CSV, delimiter=","), REPEATS)
    bw = np.repeat(np.genfromtxt(BW_CSV, delimiter=","), REPEATS)
    n = min(A.shape[1], len(pw))
    return A[:, :n].T, pw[:n], bw[:n], fs, spp


if DATA_SOURCE == "recorded":
    X_all, PW_all, BW_all, Fs, SAMPLES_PER_PRI = load_recorded(RECORD_DIR)
else:
    X_all, PW_all, BW_all, Fs, SAMPLES_PER_PRI = load_matrix()
N_total = len(PW_all)
N = SAMPLES_PER_PRI * PRI_PER_WINDOW
if X_all.shape[1] != N:
    raise ValueError(f"Windows have {X_all.shape[1]} samples; expected {N} (3 x {SAMPLES_PER_PRI}).")

print("\n============================================================")
print("DATASET")
print("============================================================")
print(f"Windows: {N_total}  ·  {N} real samples each  ·  Fs {Fs/1e6:.6g} MSPS  ·  {SAMPLES_PER_PRI} samples/PRI")
print(f"PW labels {PW_all.min():.3f} .. {PW_all.max():.3f} us   BW labels {BW_all.min():.3f} .. {BW_all.max():.3f} kHz")

# PW/BW setting id: repeats of one setting never straddle train/test or CV folds.
_, SETTING_ID = np.unique(np.round(np.column_stack([PW_all, BW_all]), 3), axis=0, return_inverse=True)
SETTING_ID = SETTING_ID.reshape(-1)
settings = np.random.RandomState(0).permutation(SETTING_ID.max() + 1)
n_test_settings = max(1, int(round(TEST_FRACTION * settings.size)))
test_settings = set(settings[:n_test_settings].tolist())
is_test = np.array([s in test_settings for s in SETTING_ID])
train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
print(f"PW/BW settings: {settings.size}  ·  train windows {train.size}  ·  test windows {test.size}")

EDGE_SMOOTH = max(1, int(round(EDGE_SMOOTH_US * 1e-6 * Fs)))
EDGE_WINDOW = max(2, int(round(EDGE_WINDOW_US * 1e-6 * Fs)))
PW_RANGE_US = (max(0.0, PW_all[train].min() * (1 - PW_SEARCH_MARGIN)),
               min(PRI_us, PW_all[train].max() * (1 + PW_SEARCH_MARGIN)))

if POLARITY == "auto":
    probe = np.random.RandomState(1).choice(train, size=min(300, train.size), replace=False)
    polarity = estimate_polarity((np.asarray(X_all[k], dtype=np.float64) for k in probe), PW_all[probe],
                                 SAMPLES_PER_PRI, Fs, EDGE_SMOOTH, EDGE_WINDOW, PW_RANGE_US)
else:
    polarity = 1 if POLARITY == "raises" else -1
print(f"Pulse polarity: {'raises' if polarity > 0 else 'lowers'} the detector level"
      f"{' (learned from labels)' if POLARITY == 'auto' else ''}")
print(f"Edge smoothing {EDGE_SMOOTH} samples, edge window {EDGE_WINDOW} samples, "
      f"PW search {PW_RANGE_US[0]:.1f}..{PW_RANGE_US[1]:.1f} us")

# ============================================================
# FEATURES (cached)
# ============================================================
rng = np.random.default_rng(42)
Win, Wres = create_real_reservoir(N_res, reservoir_density, desired_radius, input_scale, rng)
reservoir = RealReservoir(Win, Wres) if COMPUTE_RESERVOIR else None

fingerprint = np.array(f"{MODEL_FORMAT}|{DATA_SOURCE}|{N_total}|{N}|{Fs}|{polarity}|{EDGE_SMOOTH}|"
                       f"{EDGE_WINDOW}|{PW_RANGE_US}|{COMPUTE_RESERVOIR}|{N_res}|{reservoir_density}|"
                       f"{desired_radius}|{input_scale}|raw-samples")


def compute_features():
    groups = {"pw": [], "bw": [], "shape": []}
    if reservoir is not None:
        groups["res"] = []
        print(f"Reservoir backend: {reservoir.backend}")
    snr = np.zeros(N_total)
    started = time.time()
    for k in range(N_total):
        f, info = extract_features(np.asarray(X_all[k], dtype=np.float64), SAMPLES_PER_PRI, Fs,
                                   polarity, EDGE_SMOOTH, EDGE_WINDOW, PW_RANGE_US, reservoir)
        for g in groups:
            groups[g].append(f[g])
        snr[k] = info["snr_db"]
        if (k + 1) % 200 == 0:
            rate = (k + 1) / (time.time() - started)
            print(f"Feature extraction : {k + 1}/{N_total}  ({rate:.1f} windows/s, "
                  f"~{(N_total - k - 1) / rate / 60:.1f} min left)")
    return {g: np.array(v) for g, v in groups.items()}, snr


cached = False
if FEATURE_CACHE and os.path.exists(FEATURE_CACHE):
    with np.load(FEATURE_CACHE, allow_pickle=False) as cache:
        if str(cache["fingerprint"]) == str(fingerprint):
            features_all = {g: cache[g].copy() for g in ("pw", "bw", "shape", "res") if g in cache.files}
            snr_all = cache["snr_db"].copy()
            cached = True
            print(f"Loaded cached features from {FEATURE_CACHE}")
if not cached:
    features_all, snr_all = compute_features()
    if FEATURE_CACHE:
        np.savez(FEATURE_CACHE, fingerprint=fingerprint, snr_db=snr_all, **features_all)
print(f"Median in-pulse SNR: {np.median(snr_all):.1f} dB")

X_train = {g: v[train] for g, v in features_all.items()}
X_test = {g: v[test] for g, v in features_all.items()}
Y_train = np.column_stack([PW_all[train], BW_all[train]])
Y_test = np.column_stack([PW_all[test], BW_all[test]])


def grouped_cv_rmse(features, y, groups, setting_id, alphas, n_folds):
    """K-fold CV where all windows of one PW/BW setting stay in the same fold."""
    X = assemble(features, groups)
    ids = np.random.RandomState(0).permutation(np.unique(setting_id))
    fold_of = dict(zip(ids, np.arange(ids.size) % n_folds))
    fold = np.array([fold_of[s] for s in setting_id])
    sse = np.zeros(len(alphas))
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd < 1e-12] = 1.0
        y_mu, y_sd = y[tr].mean(), max(y[tr].std(), 1e-12)
        Xtr = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        Xva = np.column_stack([np.ones(va.sum()), (X[va] - mu) / sd])
        for a, alpha in enumerate(alphas):
            w = ridge_solve(Xtr, (y[tr] - y_mu) / y_sd, alpha)
            sse[a] += np.sum((Xva @ w * y_sd + y_mu - y[va])**2)
    return np.sqrt(sse / len(y))


print("\n============================================================")
print("TRAINING SEPARATE RIDGE READOUT PER PARAMETER")
print("============================================================")
readouts = {}
for t, name in enumerate(("PW", "BW")):
    unit = "us" if name == "PW" else "kHz"
    best = None
    for groups in CANDIDATE_FEATURES[name]:
        if any(g not in X_train for g in groups):
            continue
        rmse = grouped_cv_rmse(X_train, Y_train[:, t], groups, SETTING_ID[train], ALPHA_GRID, CV_FOLDS)
        i = int(np.argmin(rmse))
        print(f"{name}: features={'+'.join(groups):18s} best alpha={ALPHA_GRID[i]:.1e}  "
              f"CV RMSE={rmse[i]:.5f} {unit}")
        if best is None or rmse[i] < best[0]:
            best = (rmse[i], groups, ALPHA_GRID[i])
    _, groups, alpha = best
    readouts[name] = Readout.fit(name, groups, X_train, Y_train[:, t], alpha)
    print(f"--> {name} readout: features={'+'.join(groups)}  alpha={alpha:.1e}  "
          f"weights={readouts[name].w.size}")

uses_reservoir = any("res" in r.groups for r in readouts.values())
save_model(
    MODEL_PATH, [readouts["PW"], readouts["BW"]], fs=Fs, samples_per_pri=SAMPLES_PER_PRI,
    polarity=polarity, edge_smooth=EDGE_SMOOTH, edge_window=EDGE_WINDOW, pw_range_us=PW_RANGE_US,
    reservoir_arrays=dict(Win=Win, Wres=Wres) if uses_reservoir else None,
    PRI_us=PRI_us, pri_per_window=PRI_PER_WINDOW, N_res=N_res, reservoir_density=reservoir_density,
    desired_radius=desired_radius, input_scale=input_scale,
    pw_feature_names=np.array(PW_FEATURE_NAMES), bw_feature_names=np.array(BW_FEATURE_NAMES),
)
print(f"Saved model to {MODEL_PATH}")

# ============================================================
# TEST
# ============================================================
def metrics(y, p):
    e = p - y
    return (np.sqrt(np.mean(e**2)), np.mean(np.abs(e)), np.median(np.abs(e)), np.max(np.abs(e)),
            1 - np.sum(e**2) / np.sum((y - y.mean())**2))


PW_true, BW_true = Y_test[:, 0], Y_test[:, 1]
PW_pred, BW_pred = readouts["PW"].predict(X_test), readouts["BW"].predict(X_test)
M_true, M_pred = BW_true / PW_true * 1e9, BW_pred / PW_pred * 1e9       # Hz/s

for label, y, p in (("PW", PW_true, PW_pred), ("BW", BW_true, BW_pred), ("M", M_true, M_pred)):
    plt.figure()
    plt.plot(y, label="True")
    plt.plot(p, label="Pred")
    plt.title(f"{label} pred Vs True (held-out settings)")
    plt.legend()
plt.show()

print("\n============================================================")
print("RESULTS ON HELD-OUT PW/BW SETTINGS")
print("============================================================")
r = metrics(PW_true, PW_pred)
print(f"PW  RMSE {r[0]*1e3:9.2f} ns   MAE {r[1]*1e3:9.2f} ns   median {r[2]*1e3:9.2f} ns   "
      f"max {r[3]*1e3:9.2f} ns   R^2 {r[4]:.6f}")
r = metrics(BW_true, BW_pred)
print(f"BW  RMSE {r[0]:9.3f} kHz  MAE {r[1]:9.3f} kHz  median {r[2]:9.3f} kHz  "
      f"max {r[3]:9.3f} kHz  R^2 {r[4]:.6f}")
rel = np.abs(M_pred - M_true) / np.abs(M_true) * 100
print(f"M   mean rel {rel.mean():.3f}%   median rel {np.median(rel):.3f}%   within 5%: {np.mean(rel <= 5)*100:.1f}%")

print("\nSample | PW true us | PW pred us | BW true kHz | BW pred kHz")
for i in range(min(10, len(PW_true))):
    print(f"{i:6d} | {PW_true[i]:10.4f} | {PW_pred[i]:10.4f} | {BW_true[i]:11.3f} | {BW_pred[i]:11.3f}")

print("\n============================================================")
print("CONFIGURATION")
print("============================================================")
print(f"Fs {Fs/1e6:.6g} MSPS · window {N} samples (3 x PRI {PRI_us:g} us, fixed) · "
      f"polarity {'raises' if polarity > 0 else 'lowers'}")
print(f"Reservoir {N_res} neurons (real), density {reservoir_density}, radius {desired_radius}, "
      "driven by raw samples (no Phi compression)")
for name, r in readouts.items():
    print(f"  {name} features / alpha : {'+'.join(r.groups)} / {r.alpha:.1e}")
print(f"Shape points: {N_SHAPE}")
print("DONE")
