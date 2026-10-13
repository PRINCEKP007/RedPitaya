"""Physical reservoir computing on the atomic receiver: is the mapping learnable?

Input: the acquired data, one column per window of 3 PRI real samples
(3*samples_per_pri x N_dataset), plus the known RF parameters per window.
The measured photodetector signal is used directly as the reservoir state
(the atoms perform the nonlinear high-dimensional transform); only linear
readouts are trained. Steps:

 1. INFORMATION TESTS - before any regression, do the measured states for known
    parameter values separate? (repeatability vs separation, nearest-neighbour
    identification of settings, state distance vs parameter distance)
 2. LINEAR DECODABILITY - grouped cross-validated ridge readouts for PW, BW and
    the chirp rate K = BW/PW estimated DIRECTLY (own weights), plus a
    shuffled-label control that must fail.
 3. HELD-OUT TEST - unseen PW/BW settings: direct K vs derived K = BW_hat/PW_hat,
    and the same windows rotated to random start phases (live windows hold two
    full pulses plus parts of a third).
 4. Optional: digital ESN on the same data, to show post-processing does not add
    information (data-processing inequality).

Writes the model (for RedPitaya_ESN_Dashboard.py), rc_report/report.json and figures.
Run: python TrainPhysicalRC.py
"""
import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # figures are saved to REPORT_DIR
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from physical_rc import MODEL_FORMAT, UNITS, PhysicalStates, save_model
from rp_core import Readout, assemble, ridge_solve

# ============================================================
# DATA: 3PRI samples x N_dataset
# ============================================================
DATA_SOURCE = "matrix"                   # "matrix" or "recorded" (dashboard recorder folder)
MATRIX_FILE = "X_3PRI_real.npy"          # shape (3*samples_per_pri, N) - or (N, 3*spp), detected
PW_CSV, BW_CSV = "PW.csv", "BW.csv"      # PW in us, BW in kHz: one per window, or one per setting
REPEATS = 10                             # windows per setting when the CSVs list settings
RECORD_DIR = "rp_dataset"
DECIMATION = 32                          # Red Pitaya decimation used for the matrix data
BASE_RATE = 125e6
PRI_us = 512.0
PRI_PER_WINDOW = 3

# ============================================================
# PHYSICAL STATES + READOUTS
# ============================================================
NORMALISE = "zscore"         # per-window zero-mean/unit-std ("none" keeps absolute level)
BIN_US = 1.0                 # virtual-node duration (samples averaged per node)
N_SPECTRUM = 400             # PRI harmonics kept for the shift-invariant representation
CANDIDATES = {               # per target: representations tried by grouped CV
    "PW": [["aligned"], ["spectrum"], ["aligned", "spectrum"]],
    "BW": [["aligned"], ["spectrum"], ["aligned", "spectrum"]],
    "K":  [["aligned"], ["spectrum"], ["aligned", "spectrum"]],
}
ALPHA_GRID = np.logspace(-4, 4, 17)
CV_FOLDS = 5
TEST_FRACTION = 0.1          # held-out PW/BW SETTINGS
KNN_SUBSET = 4000            # windows used for the nearest-neighbour test
SIGNAL_DIMS = 10             # separability is also measured in the top-N principal
                             # directions of the setting means (where the parameters live);
                             # in the full node space, per-node noise dominates distances
COMPARE_DIGITAL_ESN = True   # extra: digital ESN on the same raw windows (subset)
ESN_SUBSET, ESN_NEURONS = 1500, 200
MODEL_PATH = "rp_physical_rc_model.npz"
REPORT_DIR = "rc_report"
SEED = 0


# ============================================================
# LOAD
# ============================================================
def load_matrix():
    fs = BASE_RATE / DECIMATION
    spp = int(round(fs * PRI_us * 1e-6))
    n_win = spp * PRI_PER_WINDOW
    A = np.load(MATRIX_FILE, mmap_mode="r")
    if A.ndim == 1:
        A = A.reshape(n_win, -1, order="F")
    if A.shape[0] != n_win and A.shape[1] == n_win:
        A = A.T                                        # given as N x samples
    if A.shape[0] != n_win:
        raise ValueError(f"Expected {n_win} samples per window (3 x {spp}); got shape {A.shape}.")
    n = A.shape[1]
    pw, bw = np.atleast_1d(np.genfromtxt(PW_CSV, delimiter=",")), np.atleast_1d(np.genfromtxt(BW_CSV, delimiter=","))
    if pw.size != n:
        pw, bw = np.repeat(pw, REPEATS), np.repeat(bw, REPEATS)
    n = min(n, pw.size)
    return A[:, :n].T, pw[:n], bw[:n], fs, spp


def load_recorded():
    with open(os.path.join(RECORD_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    X = np.memmap(os.path.join(RECORD_DIR, "windows_f32.bin"), dtype="<f4", mode="r")
    X = X.reshape(-1, int(meta["samples_per_window"]))
    with open(os.path.join(RECORD_DIR, "labels.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = min(len(rows), X.shape[0])
    pw = np.array([float(r["pw_us"]) for r in rows[:n]])
    bw = np.array([float(r["bw_khz"]) for r in rows[:n]])
    return X[:n], pw, bw, float(meta["fs_hz"]), int(meta["samples_per_pri"])


X_raw, PW, BW, Fs, SPP = load_matrix() if DATA_SOURCE == "matrix" else load_recorded()
K = BW / PW                                            # kHz/us (x1e9 = Hz/s), ground truth
TRUE = dict(PW=PW, BW=BW, K=K)
N = len(PW)
_, SETTING = np.unique(np.round(np.column_stack([PW, BW]), 4), axis=0, return_inverse=True)
SETTING = SETTING.reshape(-1)
os.makedirs(REPORT_DIR, exist_ok=True)
report = dict(data=dict(windows=N, settings=int(SETTING.max() + 1), fs_hz=Fs, samples_per_pri=SPP,
                        pw_range_us=[float(PW.min()), float(PW.max())],
                        bw_range_khz=[float(BW.min()), float(BW.max())]))
print(f"Windows {N} · settings {SETTING.max() + 1} · Fs {Fs/1e6:.6g} MSPS · {SPP} samples/PRI")
print(f"PW {PW.min():.3f}..{PW.max():.3f} us · BW {BW.min():.1f}..{BW.max():.1f} kHz · "
      f"K {K.min():.4f}..{K.max():.4f} kHz/us")

rng = np.random.RandomState(SEED)
ids = rng.permutation(SETTING.max() + 1)
test_ids = set(ids[:max(1, int(round(TEST_FRACTION * ids.size)))].tolist())
is_test = np.array([s in test_ids for s in SETTING])
train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)

# ============================================================
# PHYSICAL STATES (template learned on training windows only)
# ============================================================
bin_size = max(1, int(round(BIN_US * 1e-6 * Fs)))
template = PhysicalStates.build_template([X_raw[k] for k in rng.choice(train, min(400, train.size), replace=False)],
                                         SPP, NORMALISE)
states_fn = PhysicalStates(SPP, bin_size, NORMALISE, template, N_SPECTRUM)
t0 = time.time()
S = {"aligned": [], "spectrum": []}
match = np.zeros(N)
for k in range(N):
    st, info = states_fn.states(X_raw[k])
    for g in S:
        S[g].append(st[g])
    match[k] = info["match"]
    if (k + 1) % 2000 == 0:
        print(f"States: {k + 1}/{N}")
S = {g: np.array(v) for g, v in S.items()}
print(f"Physical states: {states_fn.n_nodes} virtual nodes ({bin_size} samples = {bin_size/Fs*1e6:.2f} us each), "
      f"{N_SPECTRUM} spectral lines · {time.time() - t0:.1f} s · median template match {np.median(match):.3f}")

# ============================================================
# 1. INFORMATION TESTS: do known parameter values give distinguishable states?
# ============================================================
print("\n==================== 1. INFORMATION / SEPARABILITY ====================")
Z = S["aligned"]
Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
settings_ok = np.array([s for s in np.unique(SETTING) if np.sum(SETTING == s) >= 2])
_, _, vt = np.linalg.svd(np.array([Zs[SETTING == s].mean(0) for s in settings_ok]) - Zs.mean(0),
                         full_matrices=False)
spaces = {"full": Zs, "signal": Zs @ vt[:SIGNAL_DIMS].T}
sub = rng.choice(N, min(KNN_SUBSET, N), replace=False)
has_partner = np.array([np.sum(SETTING[sub] == SETTING[i]) > 1 for i in sub])
chance = float(np.mean([np.sum(SETTING[sub] == SETTING[i]) - 1 for i in sub[has_partner]]) / (len(sub) - 1))
cs = settings_ok
pw_c = np.array([PW[SETTING == s][0] for s in cs])
bw_c = np.array([BW[SETTING == s][0] for s in cs])
iu = np.triu_indices(len(cs), 1)
dpar = np.sqrt(((pw_c[:, None] - pw_c[None]) / PW.std())**2 + ((bw_c[:, None] - bw_c[None]) / BW.std())**2)[iu]
report["information"] = dict(signal_dims=SIGNAL_DIMS, nn_chance_percent=chance * 100)
for space, V in spaces.items():
    label = "full node space" if space == "full" else f"signal subspace ({SIGNAL_DIMS} dims)"
    cent = np.array([V[SETTING == s].mean(0) for s in cs])
    within = np.array([np.sqrt(np.mean(np.sum((V[SETTING == s] - c)**2, axis=1))) for s, c in zip(cs, cent)])
    d_cent = np.sqrt(np.maximum(((cent[:, None, :] - cent[None, :, :])**2).sum(-1), 0))
    dstate = d_cent[iu]
    np.fill_diagonal(d_cent, np.inf)
    nearest = d_cent.min(1)
    sep_ratio = nearest / (within + 1e-12)
    sq = (V[sub]**2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * V[sub] @ V[sub].T
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    nn_acc = float(np.mean(SETTING[sub][nn][has_partner] == SETTING[sub][has_partner]))
    nn_err = dict(PW=float(np.median(np.abs(PW[sub][nn] - PW[sub]))),
                  BW=float(np.median(np.abs(BW[sub][nn] - BW[sub]))))
    rho = float(spearmanr(dpar, dstate).correlation) if dpar.size > 2 else float("nan")
    print(f"[{label}]")
    print(f"  repeat spread {np.median(within):.3f} · nearest other setting {np.median(nearest):.3f} · "
          f"ratio median {np.median(sep_ratio):.2f} · settings with ratio > 1: {np.mean(sep_ratio > 1)*100:.1f}%")
    print(f"  nearest neighbour = same setting: {nn_acc*100:.1f}% (chance {chance*100:.2f}%) · "
          f"NN parameter error median PW {nn_err['PW']:.3f} us, BW {nn_err['BW']:.1f} kHz")
    print(f"  state distance vs parameter distance (Spearman): {rho:.3f}  (1 = perfectly consistent mapping)")
    report["information"][space] = dict(repeat_spread_median=float(np.median(within)),
                                        nearest_setting_distance_median=float(np.median(nearest)),
                                        separation_ratio_median=float(np.median(sep_ratio)),
                                        settings_separable_percent=float(np.mean(sep_ratio > 1) * 100),
                                        nn_same_setting_percent=nn_acc * 100, nn_param_error_median=nn_err,
                                        distance_vs_param_spearman=rho)
    if space == "signal":
        sig_dstate, sig_ratio, sig_rho = dstate, sep_ratio, rho

# ============================================================
# 2. LINEAR DECODABILITY (grouped CV) + SHUFFLED-LABEL CONTROL
# ============================================================
def grouped_folds(setting, n_folds, seed=SEED):
    u = np.random.RandomState(seed).permutation(np.unique(setting))
    fold_of = dict(zip(u, np.arange(u.size) % n_folds))
    return np.array([fold_of[s] for s in setting])


def cv_predict(X, y, setting, alphas, n_folds=CV_FOLDS):
    """Out-of-fold predictions for every alpha (settings never split across folds)."""
    fold = grouped_folds(setting, n_folds)
    pred = np.zeros((len(alphas), len(y)))
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-12] = 1.0
        y_mu, y_sd = y[tr].mean(), max(y[tr].std(), 1e-12)
        Xtr = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        Xva = np.column_stack([np.ones(va.sum()), (X[va] - mu) / sd])
        for a, alpha in enumerate(alphas):
            pred[a, va] = Xva @ ridge_solve(Xtr, (y[tr] - y_mu) / y_sd, alpha) * y_sd + y_mu
    return pred


def metrics(y, p):
    e = p - y
    rel = np.abs(e) / np.abs(y) * 100
    return dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))),
                median_abs=float(np.median(np.abs(e))), max_abs=float(np.max(np.abs(e))),
                bias=float(np.mean(e)), r2=float(1 - np.sum(e**2) / np.sum((y - y.mean())**2)),
                mape_percent=float(rel.mean()), within_1_percent=float(np.mean(rel <= 1) * 100),
                within_5_percent=float(np.mean(rel <= 5) * 100))


print("\n==================== 2. LINEAR READOUT ON PHYSICAL STATES (grouped CV) ====================")
S_train = {g: v[train] for g, v in S.items()}
chosen, report["cv"] = {}, {}
for name in ("PW", "BW", "K"):
    y = TRUE[name][train]
    best = None
    for groups in CANDIDATES[name]:
        pred = cv_predict(assemble(S_train, groups), y, SETTING[train], ALPHA_GRID)
        rmse = np.sqrt(np.mean((pred - y)**2, axis=1))
        a = int(np.argmin(rmse))
        m = metrics(y, pred[a])
        report["cv"][f"{name}:{'+'.join(groups)}"] = dict(alpha=float(ALPHA_GRID[a]), **m)
        print(f"{name:2s} {'+'.join(groups):17s} alpha {ALPHA_GRID[a]:.0e}  CV RMSE {m['rmse']:.5g} {UNITS[name]}  "
              f"R² {m['r2']:.4f}  MAPE {m['mape_percent']:.3f}%")
        if best is None or m["rmse"] < best[0]:
            best = (m["rmse"], groups, float(ALPHA_GRID[a]))
    chosen[name] = best

# Shuffled-label control: permute parameter values ACROSS settings. A real mapping
# must vanish (R² <= ~0); a positive R² here would signal leakage / overfitting.
perm = np.random.RandomState(1).permutation(np.unique(SETTING[train]))
remap = dict(zip(np.unique(SETTING[train]), perm))
control = {}
for name in ("PW", "BW", "K"):
    y = TRUE[name][train]
    by_setting = {s: y[SETTING[train] == s][0] for s in np.unique(SETTING[train])}
    y_shuf = np.array([by_setting[remap[s]] for s in SETTING[train]])
    _, groups, alpha = chosen[name]
    p = cv_predict(assemble(S_train, groups), y_shuf, SETTING[train], [alpha])[0]
    control[name] = metrics(y_shuf, p)["r2"]
print("Shuffled-label control R² (should be <= ~0): " +
      ", ".join(f"{n} {v:.3f}" for n, v in control.items()))
report["shuffled_label_control_r2"] = control

# ============================================================
# 3. FINAL READOUTS + HELD-OUT SETTINGS (direct vs derived chirp rate)
# ============================================================
readouts = {n: Readout.fit(n, chosen[n][1], S_train, TRUE[n][train], chosen[n][2]) for n in ("PW", "BW", "K")}
save_model(MODEL_PATH, list(readouts.values()), states_fn, Fs, PRI_us, PRI_PER_WINDOW)
print(f"\nSaved {MODEL_PATH} ({MODEL_FORMAT}): " +
      "; ".join(f"{n} on {'+'.join(r.groups)} (alpha {r.alpha:.0e})" for n, r in readouts.items()))


def evaluate(windows_states):
    pred = {n: readouts[n].predict(windows_states) for n in readouts}
    pred["K_derived"] = pred["BW"] / pred["PW"]
    return pred


print("\n==================== 3. HELD-OUT PW/BW SETTINGS ====================")
pred = evaluate({g: v[test] for g, v in S.items()})
# Live-like windows: the same test windows starting at random phases (2 pulses + parts).
roll_rng = np.random.RandomState(2)
rolled = {"aligned": [], "spectrum": []}
for k in test:
    st, _ = states_fn.states(np.roll(np.asarray(X_raw[k]), roll_rng.randint(SPP * PRI_PER_WINDOW)))
    for g in rolled:
        rolled[g].append(st[g])
pred_rolled = evaluate({g: np.array(v) for g, v in rolled.items()})
report["test"] = {}
rows = [("PW", PW[test], "PW"), ("BW", BW[test], "BW"), ("K direct", K[test], "K"), ("K derived", K[test], "K_derived")]
print(f"{'':10s} {'RMSE':>12s} {'MAE':>12s} {'max':>12s} {'bias':>11s} {'R²':>8s} {'MAPE':>8s} {'≤1%':>6s} {'≤5%':>6s}   rolled-window RMSE")
for label, y, key in rows:
    m, mr = metrics(y, pred[key]), metrics(y, pred_rolled[key])
    report["test"][label] = dict(aligned_windows=m, random_phase_windows=mr)
    unit = UNITS["K" if key.startswith("K") else key]
    print(f"{label:10s} {m['rmse']:12.5g} {m['mae']:12.5g} {m['max_abs']:12.5g} {m['bias']:11.3g} {m['r2']:8.4f} "
          f"{m['mape_percent']:7.3f}% {m['within_1_percent']:5.1f}% {m['within_5_percent']:5.1f}%   {mr['rmse']:.5g} {unit}")
kd, kv = report["test"]["K direct"]["aligned_windows"], report["test"]["K derived"]["aligned_windows"]
print(f"Chirp rate: direct readout RMSE {kd['rmse']*1e9:.4g} Hz/s vs derived BW/PW {kv['rmse']*1e9:.4g} Hz/s "
      f"({'direct better' if kd['rmse'] < kv['rmse'] else 'derived better'}, ratio {kv['rmse']/max(kd['rmse'],1e-30):.2f}x)")

# Repeatability of the estimate itself (within-setting spread) vs total error.
for name in ("PW", "BW", "K"):
    spread = np.median([np.std(pred[name][SETTING[test] == s]) for s in np.unique(SETTING[test])])
    report["test"][name if name != "K" else "K direct"]["within_setting_std_median"] = float(spread)

# ============================================================
# 4. OPTIONAL: digital ESN on the same raw windows (does post-processing add information?)
# ============================================================
if COMPARE_DIGITAL_ESN:
    from rp_core import RealReservoir, create_real_reservoir
    print("\n==================== 4. DIGITAL ESN ON THE SAME DATA (subset) ====================")
    subset = np.sort(rng.choice(train, min(ESN_SUBSET, train.size), replace=False))
    phi, win, wres = create_real_reservoir(ESN_NEURONS, 0.03, 0.9, 1, 4, 0.5, np.random.default_rng(SEED))
    esn = RealReservoir(phi, win, wres, 4, 4)
    R = np.array([esn.features(np.asarray(X_raw[k])) for k in subset])
    report["digital_esn_vs_physical"] = {}
    for name in ("PW", "BW", "K"):
        y = TRUE[name][subset]
        r_phys = np.sqrt(np.mean((cv_predict(assemble({g: v[subset] for g, v in S.items()}, chosen[name][1]),
                                             y, SETTING[subset], ALPHA_GRID) - y)**2, axis=1)).min()
        r_esn = np.sqrt(np.mean((cv_predict(R, y, SETTING[subset], ALPHA_GRID) - y)**2, axis=1)).min()
        report["digital_esn_vs_physical"][name] = dict(physical_rmse=float(r_phys), digital_esn_rmse=float(r_esn))
        print(f"{name:2s} CV RMSE  physical states + linear readout {r_phys:.5g}  ·  digital ESN ({ESN_NEURONS} neurons) "
              f"{r_esn:.5g} {UNITS[name]}")
    print("A digital ESN cannot add information to the measurement; where it scores better, the information\n"
          "was already in the physical states but not linearly accessible (a readout limitation, not new physics).")

# ============================================================
# FIGURES + REPORT
# ============================================================
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for a, (label, y, key) in zip(ax.flat, rows):
    a.scatter(y, pred[key], s=8, label="aligned windows")
    a.scatter(y, pred_rolled[key], s=8, alpha=0.5, label="random-phase windows")
    lo, hi = y.min(), y.max()
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    unit = UNITS["K" if key.startswith("K") else key]
    a.set_title(f"{label}: RMSE {report['test'][label]['aligned_windows']['rmse']:.4g} {unit}")
    a.set_xlabel(f"true ({unit})")
    a.set_ylabel(f"estimated ({unit})")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "pred_vs_true.png"), dpi=120)

pcs = spaces["signal"][sub][:, :2]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (label, values) in zip(ax, (("PW (us)", PW[sub]), ("BW (kHz)", BW[sub]), ("K (kHz/us)", K[sub]))):
    sc = a.scatter(pcs[:, 0], pcs[:, 1], c=values, s=6, cmap="viridis")
    fig.colorbar(sc, ax=a, label=label)
    a.set_title(f"Signal subspace of physical states · colour = {label}")
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "state_pca.png"), dpi=120)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
pick = np.random.RandomState(3).choice(dpar.size, min(20000, dpar.size), replace=False)
ax[0].scatter(dpar[pick], sig_dstate[pick], s=3, alpha=0.4)
ax[0].set_xlabel("parameter distance (normalised PW, BW)")
ax[0].set_ylabel("state distance (setting centroids, signal subspace)")
ax[0].set_title(f"Consistency of the mapping · Spearman {sig_rho:.3f}")
ax[1].hist(np.log10(sig_ratio), bins=40)
ax[1].axvline(0, color="k", ls="--")
ax[1].set_xlabel("log10(nearest-setting distance / repeat spread)")
ax[1].set_title("Separability per setting, signal subspace (> 0: distinguishable)")
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "separability.png"), dpi=120)

with open(os.path.join(REPORT_DIR, "report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"\nReport and figures written to {REPORT_DIR}/ (report.json, pred_vs_true.png, state_pca.png, separability.png)")
