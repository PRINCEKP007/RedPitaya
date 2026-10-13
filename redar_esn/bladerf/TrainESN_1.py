import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from lib import create_input_matrix,create_local_measurement_matrix,create_reservoir_matrix,calculate_metrics
from esn_core import (MODEL_FORMAT, Reservoir, Readout, assemble, extract_features, ridge_solve,
                      save_model, PW_FEATURE_NAMES, BW_FEATURE_NAMES)
import random
import time
np.random.seed(42)
random.seed(42)
Fs = 20e6

# Fixed TX timing: PRI = 512 us, one window = 3 PRI = 30720 samples,
# 3 pulses of the same PW/BW per window. PRI is not estimated.
PRI_us = 512.0
SAMPLES_PER_PRI = int(round(Fs * PRI_us * 1e-6))     # 10240

# A = np.memmap(
#     "ESN_Train.bin",
#     dtype="<c16",
#     mode="r",
#     shape=(30720, 6000),
#     order="F"
# )

A = np.load("X_3PRI.npy", mmap_mode="r")
A = A.reshape(30720, 20000, order="F")

print(A.shape)
print(A.dtype)

PW_raw = np.genfromtxt("PW.csv", delimiter=",")      # us
BW_raw = np.genfromtxt("BW.csv",delimiter=",")       # kHz

N_SETTINGS = 2000
REPEATS = 10              # each PW/BW setting was captured 10 times (consecutive columns)

PW = PW_raw[0:N_SETTINGS]
BW = BW_raw[0:N_SETTINGS]

# Same layout as before (PW_Full[:, j] = PW, reshaped row-major): column k -> setting k // 10
PW_Full = np.repeat(PW, REPEATS)
BW_Full = np.repeat(BW, REPEATS)
SETTING_ID = np.repeat(np.arange(N_SETTINGS), REPEATS)   # used to keep repeats in one CV fold

Ntrain = 19000
Ntest = 1000

reservoir_density = 0.02
desired_radius = 0.9
input_scale = 0.5
N_res = 1000

BLOCK_SIZE = 10
STRIDE = 10
N = len(A)
M_local = 1
N_blocks = N // STRIDE

# ------------------------------------------------------------
# SEPARATE READOUT PER PARAMETER
# Feature groups:
#   "pw"  : physical PW features from the folded pulse envelope
#           (energy width, interpolated 50% edges, 25/50/75% widths, contrast)
#   "bw"  : physical BW features (chirp rate x PW from phase / inst. frequency,
#           occupied and RMS spectral bandwidth)
#   "res" : ESN reservoir energy-mean features (N_res values)
# For each parameter, every candidate feature set and every alpha is scored by
# grouped K-fold CV on the training set; the best combination is used.
# ------------------------------------------------------------
COMPUTE_RESERVOIR = True      # False: physical features only (faster training and RX)
CANDIDATE_FEATURES = {
    "PW": [["res"], ["pw"], ["pw", "res"]],
    "BW": [["res"], ["bw", "pw"], ["bw", "pw", "res"]],
}
ALPHA_GRID = np.logspace(-6, 3, 19)
CV_FOLDS = 5
FEATURE_CACHE = "esn_features_cache.npz"   # set to None to always recompute


print("\n============================================================")
print("TRAINING DATASET")
print("============================================================")



Phi_local = create_local_measurement_matrix(M_local,BLOCK_SIZE)
Wres = create_reservoir_matrix(N_res,reservoir_density,desired_radius)
Win = create_input_matrix(N_res,M_local,input_scale)
reservoir = Reservoir(Phi_local, Win, Wres, BLOCK_SIZE, STRIDE) if COMPUTE_RESERVOIR else None

N_total = Ntrain + Ntest
fingerprint = np.array(f"{MODEL_FORMAT}|{N_total}|{COMPUTE_RESERVOIR}|{N_res}|{reservoir_density}|"
                       f"{desired_radius}|{input_scale}|{BLOCK_SIZE}|{STRIDE}|{SAMPLES_PER_PRI}")


def compute_features():
    groups = {"pw": [], "bw": []}
    if reservoir is not None:
        groups["res"] = []
    detected = np.zeros(N_total, dtype=bool)
    started = time.time()
    for k in range(N_total):
        x = np.asarray(A[:, k])
        features, info = extract_features(x, SAMPLES_PER_PRI, Fs, reservoir)
        for g in groups:
            groups[g].append(features[g])
        detected[k] = info["detected"]
        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - started)
            print(f"Feature extraction : {k + 1}/{N_total}  "
                  f"({rate:.1f} windows/s, ~{(N_total - k - 1) / rate / 60:.1f} min left)")
    return {g: np.array(v) for g, v in groups.items()}, detected


if FEATURE_CACHE and os.path.exists(FEATURE_CACHE):
    with np.load(FEATURE_CACHE, allow_pickle=False) as cache:
        cached = str(cache["fingerprint"]) == str(fingerprint)
        if cached:
            features_all = {g: cache[g].copy() for g in ("pw", "bw", "res") if g in cache.files}
            detected_all = cache["detected"].copy()
            print(f"\nLoaded cached features from {FEATURE_CACHE}")
else:
    cached = False
if not cached:
    if reservoir is not None:
        print(f"\nReservoir backend: {reservoir.backend}")
    features_all, detected_all = compute_features()
    if FEATURE_CACHE:
        np.savez(FEATURE_CACHE, fingerprint=fingerprint, detected=detected_all, **features_all)

train = np.arange(0, Ntrain)
test = np.arange(Ntrain, N_total)

X_train = {g: v[train] for g, v in features_all.items()}
X_test = {g: v[test] for g, v in features_all.items()}

Y_train = np.column_stack([PW_Full[train], BW_Full[train]])     # PW us, BW kHz
Y_test = np.column_stack([PW_Full[test], BW_Full[test]])

print("\nDataset shapes:")
for g in X_train:
    print(f"X_train[{g}]:", X_train[g].shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)
print(f"Pulse detected: train {detected_all[train].mean()*100:.1f}%  test {detected_all[test].mean()*100:.1f}%")


def grouped_cv_rmse(features, y, groups, setting_id, alphas, n_folds):
    """K-fold CV where all repeats of one PW/BW setting stay in the same fold."""
    X = assemble(features, groups)
    settings = np.random.RandomState(0).permutation(np.unique(setting_id))
    fold_of = dict(zip(settings, np.arange(settings.size) % n_folds))
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
    y_tr = Y_train[:, t]
    best = None
    for groups in CANDIDATE_FEATURES[name]:
        if any(g not in X_train for g in groups):
            continue
        rmse = grouped_cv_rmse(X_train, y_tr, groups, SETTING_ID[train], ALPHA_GRID, CV_FOLDS)
        i = int(np.argmin(rmse))
        unit = "us" if name == "PW" else "kHz"
        print(f"{name}: features={'+'.join(groups):10s} best alpha={ALPHA_GRID[i]:.1e}  "
              f"CV RMSE={rmse[i]:.5f} {unit}")
        if best is None or rmse[i] < best[0]:
            best = (rmse[i], groups, ALPHA_GRID[i])
    _, groups, alpha = best
    readouts[name] = Readout.fit(name, groups, X_train, y_tr, alpha)
    print(f"--> {name} readout: features={'+'.join(groups)}  alpha={alpha:.1e}  "
          f"weights={readouts[name].w.size}")

# ============================================================
# SAVE FINAL WEIGHTS / MODEL
# ============================================================
uses_reservoir = any("res" in r.groups for r in readouts.values())
save_model(
    "esn_model.npz",
    [readouts["PW"], readouts["BW"]],      # separate PW and BW weights
    fs=Fs,
    samples_per_pri=SAMPLES_PER_PRI,
    # fixed random parts of the reservoir pipeline (only if a readout uses them)
    reservoir_arrays=dict(Win=Win, Wres=Wres, Phi_local=Phi_local,
                          BLOCK_SIZE=BLOCK_SIZE, STRIDE=STRIDE) if uses_reservoir else None,
    # hyperparameters
    PRI_us=PRI_us,
    N_res=N_res,
    reservoir_density=reservoir_density,
    desired_radius=desired_radius,
    input_scale=input_scale,
    M_local=M_local,
    pw_feature_names=np.array(PW_FEATURE_NAMES),
    bw_feature_names=np.array(BW_FEATURE_NAMES),
)
print("Saved model to esn_model.npz")


PW_true_us = Y_test[:, 0]
BW_true_kHz = Y_test[:, 1]

PW_pred_us = readouts["PW"].predict(X_test)
BW_pred_kHz = readouts["BW"].predict(X_test)

# Chirp rate magnitude in Hz/s
M_true = (BW_true_kHz /PW_true_us) * 1e9
M_pred = (BW_pred_kHz /PW_pred_us) * 1e9


PW_metrics = calculate_metrics(PW_true_us,PW_pred_us)
(PW_rmse,PW_mae,PW_median,PW_max_err,PW_r2) = PW_metrics

plt.plot(PW_true_us,label="True")
plt.plot(PW_pred_us,label="Pred")
plt.title("PW pred Vs True")
plt.legend()
plt.show()

BW_metrics = calculate_metrics(BW_true_kHz,BW_pred_kHz)
(BW_rmse,BW_mae,BW_median,BW_max_err,BW_r2) = BW_metrics

plt.plot(BW_true_kHz,label="True")
plt.plot(BW_pred_kHz,label="Pred")
plt.title("BW pred Vs True")
plt.legend()
plt.show()


M_metrics = calculate_metrics(M_true,M_pred)
(M_rmse,M_mae,M_median,M_max_err,M_r2) = M_metrics
plt.plot(M_true,label="True")
plt.plot(M_pred,label="Pred")
plt.title("M pred Vs True")
plt.legend()
plt.show()


M_relative_error = (np.abs(M_pred - M_true)/np.abs(M_true)) * 100
M_mean_relative_error = np.mean(M_relative_error)
M_median_relative_error = np.median(M_relative_error)
M_within_5 = np.mean(M_relative_error <= 5) * 100

print("\n")
print("============================================================")
print("EXPERIMENT 1 RESULTS")
print("SEPARATE PW / BW READOUTS (PHYSICAL + ESN FEATURES) + RIDGE")
print("============================================================")


print("\n---------------- PW ESTIMATION ----------------")

print(f"PW RMSE                  : "f"{PW_rmse * 1e3:.4f} ns")

print(f"PW MAE                   : "f"{PW_mae * 1e3:.4f} ns")

print(f"PW Median absolute error : "f"{PW_median * 1e3:.4f} ns")

print(f"PW Maximum absolute error: "f"{PW_max_err * 1e3:.4f} ns")

print(f"PW R^2                   : "f"{PW_r2:.6f}")


print("\n---------------- BW ESTIMATION ----------------")

print(f"BW RMSE                  : "f"{BW_rmse:.4f} KHz")

print(f"BW MAE                   : "f"{BW_mae:.4f} KHz")

print(f"BW Median absolute error : "f"{BW_median:.4f} KHz")

print(f"BW Maximum absolute error: "f"{BW_max_err:.4f} KHz")

print(f"BW R^2                   : "f"{BW_r2:.6f}")

print("\n---------------- CHIRP RATE M ----------------")

print(f"M RMSE                   : "f"{M_rmse:.6e} Hz/s")

print(f"M MAE                    : "f"{M_mae:.6e} Hz/s")

print(f"M Median absolute error  : "f"{M_median:.6e} Hz/s")

print(f"M Maximum absolute error : "f"{M_max_err:.6e} Hz/s")

print(f"M R^2                    : "f"{M_r2:.6f}")

print(f"Mean relative error      : "f"{M_mean_relative_error:.3f}%")

print(f"Median relative error    : "f"{M_median_relative_error:.3f}%")

print(f"Within 5% relative error : "f"{M_within_5:.2f}%")

# ============================================================
# 34. SHOW SOME EXAMPLES
# ============================================================

print("\n")
print("============================================================")
print("EXAMPLE PREDICTIONS")
print("============================================================")

print("\nSample | ""PW true us | PW pred us | ""BW true kHz | BW pred kHz | ""M true (1e9) | M pred (1e9)"
)

print("-" * 90)

for i in range(
    min(10, Ntest)
):

    print(
        f"{i:6d} | "
        f"{PW_true_us[i]:10.4f} | "
        f"{PW_pred_us[i]:10.4f} | "
        f"{BW_true_kHz[i]:11.3f} | "
        f"{BW_pred_kHz[i]:11.3f} | "
        f"{M_true[i]/1e9:12.4f} | "
        f"{M_pred[i]/1e9:12.4f}"
    )

print("\n")
print("============================================================")
print("EXPERIMENT CONFIGURATION")
print("============================================================")

print(
    f"Sampling frequency       : {Fs/1e6:.1f} MHz"
)

print(
    f"Observation samples      : {N}  (3 x PRI {PRI_us:g} us, fixed)"
)

print(
    f"Training signals         : {Ntrain}"
)

print(
    f"Testing signals          : {Ntest}"
)

print(
    f"Reservoir neurons        : {N_res}"
)
print(
    f"Reservoir density        : "
    f"{reservoir_density}"
)
print(
    f"Reservoir spectral radius: "
    f"{desired_radius}"
)
print(
    f"Compression measurements : "
    f"{M_local} per block"
)
print(
    f"Block size               : "
    f"{BLOCK_SIZE}"
)
print(
    f"Readout                  : RIDGE (separate weights per parameter)"
)
for name, r in readouts.items():
    print(
        f"  {name} features / alpha  : "
        f"{'+'.join(r.groups)} / {r.alpha:.1e}"
    )
print("\n============================================================")
print("DONE")
print("============================================================")
