import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from lib import create_input_matrix,compress_signal,add_complex_awgn,generate_chirp_pulse,create_local_measurement_matrix,create_reservoir_matrix,run_reservoir,energy_mean_pooling,calculate_metrics,pw_feature_extract,envelope_autocorr_features
import random
np.random.seed(42)
random.seed(42)


Fs = 100e6                  # Sampling frequency = 100 MHz
Ts = 1 / Fs                 # Sampling period

N = 6000                    # Number of time samples

N_train = 1000
N_test = 100

PW_min = 3e-6               # 3 us
PW_max = 6e-6               # 6 us

BW_min = 10e6               # 10 MHz
BW_max = 25e6               # 25 MHz



SNR_dB = 40
f_start_range = None
N_res = 600

reservoir_density = 0.03
desired_radius = 0.8
input_scale = 0.5

BLOCK_SIZE = 10
STRIDE = 3

M_local = 1
N_blocks = N // STRIDE
ridge_alpha = 1e-2          # fallback alpha (used when CV is disabled)

# ------------------------------------------------------------
# SEPARATE READOUT PER PARAMETER
# Each target gets its own feature set, its own ridge alpha
# and its own weight vector.
#   "res" : reservoir energy-mean features  (N_res values)
#   "pri" : envelope autocorrelation features (pri_est_us, peak_val)
# ------------------------------------------------------------
TARGETS = ["PW", "BW", "PRI"]
TARGET_FEATURES = {
    "PW":  ["res"],
    "BW":  ["res"],
    "PRI": ["pri"],
}
USE_CV_ALPHA = True
ALPHA_GRID = np.logspace(-6, 3, 19)
CV_FOLDS = 5


print("\n============================================================")
print("GENERATING DATASET")
print("============================================================")



Phi_local = create_local_measurement_matrix(M_local,BLOCK_SIZE)
Wres = create_reservoir_matrix(N_res,reservoir_density,desired_radius)
Win = create_input_matrix(N_res,M_local,input_scale)

X_train = []
Y_train = []

X_test = []
Y_test = []

print("\nGenerating training signals...")

pw_ = np.arange(PW_min,PW_max,25e-9)
bw_ = np.array([5e6,10e6,15e6,20e6])
#pri_ = np.array([15e-6,17e-6,19e-6,21e-6])
pri_ = np.arange(15e-6,20e-6,25e-9)



for k in range(N_train):
    num1 = random.randint(0,len(pw_)-1)
    num2 = random.randint(0,len(bw_)-1)
    num3 = random.randint(0,len(pri_)-1)
    PW = pw_[num1]
    BW = bw_[num2]
    PRI = pri_[num3]
    x = generate_chirp_pulse(Fs,N,PW,BW,PRI)
    x_noisy = add_complex_awgn(x,SNR_dB)
    Y_compressed = compress_signal(x_noisy,Phi_local,BLOCK_SIZE,STRIDE)
    states = run_reservoir(Y_compressed,Win,Wres)

    #masked_mean, masked_max, n_active = masked_pooling(states, Y_compressed)
    pw_bw_feat = pw_feature_extract(Y_compressed, Fs,states)
    pri_feat = envelope_autocorr_features(Y_compressed, Fs, STRIDE, lag_min_us=13, lag_max_us=22)

    #print(pw_bw_feat.shape)

    X_train.append({"res": pw_bw_feat, "pri": pri_feat})
    
    Y_train.append([PW * 1e6,BW / 1e6,PRI * 1e6])

    #np.array(X_train)
    if (k + 1) % 100 == 0:

        print(
            f"Training signals generated: "
            f"{k + 1}/{N_train}"
        )

print("\nGenerating test signals...")

X_train = {g: np.array([f[g] for f in X_train]) for g in ("res", "pri")}
Y_train = np.array(Y_train)

for k in range(N_test):
    num1 = random.randint(0,len(pw_)-1)
    num2 = random.randint(0,len(bw_)-1)
    num3 = random.randint(0,len(pri_)-1)
    PW = pw_[num1]
    BW = bw_[num2]
    PRI = pri_[num3]
    x = generate_chirp_pulse(Fs,N,PW,BW,PRI)
    x_noisy = add_complex_awgn(x,SNR_dB)
    Y_compressed = compress_signal(x_noisy,Phi_local,BLOCK_SIZE,STRIDE)
    states = run_reservoir(Y_compressed,Win,Wres)

    #masked_mean, masked_max, n_active = masked_pooling(states, Y_compressed)
    pw_bw_feat = pw_feature_extract(Y_compressed, Fs,states)
    pri_feat = envelope_autocorr_features(Y_compressed, Fs, STRIDE, lag_min_us=13, lag_max_us=22)
    X_test.append({"res": pw_bw_feat, "pri": pri_feat})
    Y_test.append([PW * 1e6,BW / 1e6, PRI*1e6])


X_test = {g: np.array([f[g] for f in X_test]) for g in ("res", "pri")}
Y_test = np.array(Y_test)


print("\nDataset shapes:")
for g in X_train:
    print(f"X_train[{g}]:", X_train[g].shape)
print("Y_train:", Y_train.shape)
for g in X_test:
    print(f"X_test[{g}] :", X_test[g].shape)
print("Y_test :", Y_test.shape)

def build_features(X, groups):
    return np.column_stack([X[g] for g in groups])


def normalize(X_tr, X_te):
    mu = np.mean(X_tr, axis=0)
    sd = np.std(X_tr, axis=0)
    sd[sd < 1e-12] = 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def ridge_fit(X_aug, y, alpha):
    D = X_aug.shape[1]
    I = np.eye(D)
    I[0, 0] = 0                       # do not penalize the bias
    A = X_aug.T @ X_aug + alpha * I
    b = X_aug.T @ y
    return np.linalg.solve(A, b)      # weight vector, shape (D,)


def cv_select_alpha(X, y, alphas, n_folds):
    """K-fold CV on the training set; returns the alpha with lowest MSE."""
    idx = np.random.permutation(len(X))
    folds = np.array_split(idx, n_folds)
    mse = np.zeros(len(alphas))
    for k in range(n_folds):
        val = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        X_tr, X_val = normalize(X[tr], X[val])
        y_mu, y_sd = y[tr].mean(), max(y[tr].std(), 1e-12)
        y_tr = (y[tr] - y_mu) / y_sd
        X_tr, X_val = add_bias(X_tr), add_bias(X_val)
        for a, alpha in enumerate(alphas):
            w = ridge_fit(X_tr, y_tr, alpha)
            y_hat = X_val @ w * y_sd + y_mu
            mse[a] += np.mean((y_hat - y[val]) ** 2)
    return alphas[np.argmin(mse)]


print("\n============================================================")
print("TRAINING SEPARATE RIDGE READOUT PER PARAMETER")
print("============================================================")

W_readouts = {}                       # one weight vector per parameter
readout_alpha = {}
Y_pred = np.zeros_like(Y_test, dtype=float)

for t, name in enumerate(TARGETS):
    groups = TARGET_FEATURES[name]
    Xtr = build_features(X_train, groups)
    Xte = build_features(X_test, groups)
    y_tr = Y_train[:, t]

    alpha = (cv_select_alpha(Xtr, y_tr, ALPHA_GRID, CV_FOLDS)
             if USE_CV_ALPHA else ridge_alpha)

    # feature + target normalization (per parameter)
    Xtr_n, Xte_n = normalize(Xtr, Xte)
    y_mu, y_sd = y_tr.mean(), max(y_tr.std(), 1e-12)
    y_tr_n = (y_tr - y_mu) / y_sd

    w = ridge_fit(add_bias(Xtr_n), y_tr_n, alpha)
    Y_pred[:, t] = add_bias(Xte_n) @ w * y_sd + y_mu

    W_readouts[name] = w
    readout_alpha[name] = alpha
    print(f"{name:4s}: features={'+'.join(groups):8s} "
          f"dim={Xtr.shape[1]:4d}  alpha={alpha:.3e}  |w|={np.linalg.norm(w):.3e}")

PW_true_us = Y_test[:, 0]
BW_true_MHz = Y_test[:, 1]
PRI_true_us = Y_test[:, 2]

PW_pred_us = Y_pred[:, 0]
BW_pred_MHz = Y_pred[:, 1]
PRI_pred_us = Y_pred[:, 2]


M_true = (BW_true_MHz /PW_true_us) * 1e12
#PW_pred_us_safe = np.maximum(PW_pred_us,1e-12)

M_pred = (BW_pred_MHz /PW_pred_us) * 1e12

PW_metrics = calculate_metrics(PW_true_us,PW_pred_us)
(PW_rmse,PW_mae,PW_median,PW_max,PW_r2) = PW_metrics
plt.plot(PW_true_us,label="True")
plt.plot(PW_pred_us,label="Perd")
plt.title("PW pred Vs True")
plt.legend()
plt.show()

BW_metrics = calculate_metrics(BW_true_MHz,BW_pred_MHz)
(BW_rmse,BW_mae,BW_median,BW_max,BW_r2) = BW_metrics
plt.plot(BW_true_MHz,label="True")
plt.plot(BW_pred_MHz,label="Perd")
plt.title("BW pred Vs True")
plt.show()


M_metrics = calculate_metrics(M_true,M_pred)
(M_rmse,M_mae,M_median,M_max,M_r2) = M_metrics
plt.plot(M_true,label="True")
plt.plot(M_pred,label="Perd")
plt.title("M pred Vs True")  
plt.legend()  
plt.show()


PRI_metrics = calculate_metrics(PRI_true_us,PRI_pred_us)
(PRI_rmse,PRI_mae,PRI_median,PRI_max,PRI_r2) = PRI_metrics
plt.plot(PRI_true_us,label="True")
plt.plot(PRI_pred_us,label="Perd")
plt.title("PRI pred Vs True")  
plt.legend()  
plt.show()

M_relative_error = (np.abs(M_pred - M_true)/np.abs(M_true)) * 100
M_mean_relative_error = np.mean(M_relative_error)
M_median_relative_error = np.median(M_relative_error)
M_within_5 = np.mean(M_relative_error <= 5) * 100

print("\n")
print("============================================================")
print("EXPERIMENT 1 RESULTS")
print("ENERGY MEAN POOLING + RIDGE REGRESSION")
print("============================================================")


print("\n---------------- PW ESTIMATION ----------------")

print(f"PW RMSE                  : "f"{PW_rmse * 1e3:.4f} ns")

print(f"PW MAE                   : "f"{PW_mae * 1e3:.4f} ns")

print(f"PW Median absolute error : "f"{PW_median * 1e3:.4f} ns")

print(f"PW Maximum absolute error: "f"{PW_max * 1e3:.4f} ns")

print(f"PW R^2                   : "f"{PW_r2:.6f}")


print("\n---------------- PRI ESTIMATION ----------------")

print(f"PRI RMSE                  : "f"{PRI_rmse * 1e3:.4f} ns")

print(f"PRI MAE                   : "f"{PRI_mae * 1e3:.4f} ns")

print(f"PRI Median absolute error : "f"{PRI_median * 1e3:.4f} ns")

print(f"PRI Maximum absolute error: "f"{PRI_max * 1e3:.4f} ns")

print(f"PRI R^2                   : "f"{PRI_r2:.6f}")

print("\n---------------- BW ESTIMATION ----------------")

print(f"BW RMSE                  : "f"{BW_rmse * 1e3:.4f} KHz")

print(f"BW MAE                   : "f"{BW_mae * 1e3:.4f} KHz")

print(f"BW Median absolute error : "f"{BW_median * 1e3:.4f} KHz")

print(f"BW Maximum absolute error: "f"{BW_max * 1e3:.4f} KHz")

print(f"BW R^2                   : "f"{BW_r2:.6f}")

print("\n---------------- CHIRP RATE M ----------------")

print(f"M RMSE                   : "f"{M_rmse:.6e} Hz/s")

print(f"M MAE                    : "f"{M_mae:.6e} Hz/s")

print(f"M Median absolute error  : "f"{M_median:.6e} Hz/s")

print(f"M Maximum absolute error : "f"{M_max:.6e} Hz/s")

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

print("\nSample | ""PW true | PW pred | ""BW true | BW pred | ""M true (1e12) | M pred (1e12)"
)

print("-" * 90)

for i in range(
    min(10, N_test)
):

    print(
        f"{i:6d} | "
        f"{PW_true_us[i]*1e3:7.4f} | "
        f"{PW_pred_us[i]*1e3:7.4f} | "
        f"{BW_true_MHz[i]*1e3:7.3f} | "
        f"{BW_pred_MHz[i]*1e3:7.3f} | "
        f"{M_true[i]/1e12:12.4f} | "
        f"{M_pred[i]/1e12:12.4f}"
    )

print("\n")
print("============================================================")
print("EXPERIMENT CONFIGURATION")
print("============================================================")

print(
    f"Sampling frequency       : {Fs/1e6:.1f} MHz"
)

print(
    f"Observation samples      : {N}"
)

print(
    f"Training signals         : {N_train}"
)

print(
    f"Testing signals          : {N_test}"
)
print(
    f"PW range                 : "
    f"{PW_min*1e6:.1f} - {PW_max*1e6:.1f} us"
)
print(
    f"BW range                 : "
    f"{BW_min/1e6:.1f} - {BW_max/1e6:.1f} MHz"
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
    f"Pooling                  : ENERGY MEAN"
)
print(
    f"Readout                  : RIDGE (separate weights per parameter)"
)
for name in TARGETS:
    print(
        f"  {name:4s} features / alpha : "
        f"{'+'.join(TARGET_FEATURES[name])} / {readout_alpha[name]:.3e}"
    )
print("\n============================================================")
print("DONE")
print("============================================================")