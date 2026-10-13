"""Shared feature extraction and readouts for TrainESN_1.py and BladeRF_ESN_Dashboard.py.

numpy/scipy only (numba optional) so the real-time dashboard can import it
without matplotlib. Training and inference both call EsnModel.features() /
extract_features(), so the readouts always see exactly the features they
were fitted on.

NO-PHI VERSION: the reservoir is driven by every raw complex I/Q sample
directly (one input per time step). There is no local Phi measurement matrix,
block or stride compression in front of it.

Each estimated parameter (PW, BW) has its OWN readout: its own feature set,
normalisation, ridge alpha and weight vector. PRI is fixed by the TX setup
and is not estimated; it is used to fold the 3-PRI window into one PRI.
"""
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.sparse import csr_matrix

try:
    from numba import njit
except ImportError:
    njit = None

MODEL_FORMAT = "esn-separate-readout-v1-nophi"
TARGETS = ("PW", "BW")                    # PW in us, BW in kHz
PHYSICAL_GROUPS = ("pw", "bw")            # expanded with 2nd-order terms
MIN_CONTRAST_DB = 3.0                     # pulse/noise level ratio for "detected"
SPECTRUM_SMOOTH = 33                      # periodogram bins averaged for spectral width
# The TX chirp is 3-4 MHz wide (+/-2 MHz at baseband). A 3-tap average keeps
# 87% amplitude at 2 MHz, and lag-2 instantaneous frequency is unambiguous to
# +/-5 MHz, so neither aliases the sweep.
IQ_SMOOTH = 3
IF_LAG = 2
PW_FEATURE_NAMES = ("pw_energy_us", "pw_edge_us", "pw_count25_us",
                    "pw_count50_us", "pw_count75_us", "contrast_db")
BW_FEATURE_NAMES = ("kq1_x_pw_khz", "kq3_x_pw_khz", "kif1_x_pw_khz", "kif2_x_pw_khz",
                    "chirp_rate_khz_per_us", "obw90_khz", "obw98_khz", "rms_bw_khz",
                    "pw_edge_us")


# ============================================================
# RESERVOIR (same split-I/Q tanh + energy-mean pooling as lib.py)
# ============================================================
def _energy_kernel(measurements, win, data, indices, indptr):
    neurons = win.shape[0]
    state = np.zeros(neurons, dtype=np.complex128)
    energy = np.zeros(neurons, dtype=np.float64)
    for t in range(measurements.shape[0]):
        next_state = np.empty(neurons, dtype=np.complex128)
        for row in range(neurons):
            z = 0.0j
            for k in range(indptr[row], indptr[row + 1]):
                z += data[k] * state[indices[k]]
            for channel in range(win.shape[1]):
                z += win[row, channel] * measurements[t, channel]
            re, im = np.tanh(z.real), np.tanh(z.imag)
            next_state[row] = re + 1j * im
            energy[row] += re * re + im * im
        state = next_state
    return energy / measurements.shape[0]


_compiled_energy = njit(cache=True, nogil=True)(_energy_kernel) if njit is not None else None


class Reservoir:
    """Complex ESN driven sample-by-sample by the raw I/Q (no Phi compression).

    Win has shape (neurons, 1): every sample x[t] enters as z = Wres r + Win x[t].
    A 3-PRI window at 20 MSPS is 30,720 reservoir steps (10x the compressed version).
    """
    def __init__(self, win, wres):
        self.win = np.ascontiguousarray(np.asarray(win, dtype=np.complex128).reshape(-1, 1))
        self.wres = csr_matrix(np.asarray(wres, dtype=np.complex128))
        self.neurons = self.wres.shape[0]
        if self.win.shape[0] != self.neurons:
            raise ValueError("Win must have one row per reservoir neuron.")
        self.backend = "Numba" if _compiled_energy is not None else "SciPy sparse"

    def energy(self, x):
        y = np.ascontiguousarray(np.asarray(x, dtype=np.complex128).reshape(-1, 1))   # raw samples
        if _compiled_energy is not None:
            return _compiled_energy(y, self.win, self.wres.data, self.wres.indices, self.wres.indptr)
        state = np.zeros(self.neurons, dtype=np.complex128)
        energy = np.zeros(self.neurons, dtype=np.float64)
        for measurement in y:
            z = self.wres @ state + self.win @ measurement
            state = np.tanh(z.real) + 1j * np.tanh(z.imag)
            energy += state.real**2 + state.imag**2
        return energy / len(y)


# ============================================================
# PHYSICAL PULSE FEATURES
# ============================================================
def _circular_smooth(v, n):
    if n <= 1:
        return v.copy()
    padded = np.r_[v[v.size - n // 2:], v, v[:n - 1 - n // 2]]
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def _longest_circular_run(mask):
    """(first index, length) of the longest circular run of True, or None."""
    if mask.all() or not mask.any():
        return None
    start = int(np.argmin(mask))               # an off sample
    rolled = np.roll(mask, -start).astype(np.int8)
    d = np.diff(np.r_[0, rolled, 0])
    rises, falls = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = int(np.argmax(falls - rises))
    return (int(rises[i]) + start) % mask.size, int(falls[i] - rises[i])


def _runs(mask, min_length):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return [(s, e) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
            if e - s >= min_length]


def _pooled_if_slope(z, segments, lag, fs):
    """Chirp rate (kHz/us) from instantaneous frequency, slope pooled over pulses."""
    num = den = 0.0
    for s, e in segments:
        if e - s <= lag + 2:
            continue
        f = np.angle(z[s + lag:e] * np.conj(z[s:e - lag])) * fs / (2 * np.pi * lag)
        t = np.arange(e - s - lag) / fs * 1e6                       # us
        t -= t.mean()
        num += t @ (f - f.mean())
        den += t @ t
    return num / den * 1e-3 if den > 0 else 0.0                     # Hz/us -> kHz/us


def _pooled_quadratic_phase(z, segments, fs):
    """Chirp rate (kHz/us) from a common t^2 term of the unwrapped phase."""
    num = den = 0.0
    for s, e in segments:
        phase = np.unwrap(np.angle(z[s:e]))
        t = np.arange(e - s) / fs * 1e6
        t -= t.mean()
        q = t * t
        q -= q.mean()                     # orthogonal to [1, t] on a symmetric grid
        num += q @ phase
        den += q @ q
    # phase = pi*k*t^2: with t in us, c [rad/us^2] = pi*k[MHz/us], so k[kHz/us] = 1e3*c/pi
    return num / den / np.pi * 1e3 if den > 0 else 0.0


def pulse_features(x, samples_per_pri, fs, smooth=9, guard=8):
    """PW and BW features from one window of whole PRIs (here 3 x 512 us).

    Returns (pw_vector, bw_vector, info). Vectors are zeros when no pulse is
    found, and info["detected"] is False.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = int(samples_per_pri)
    n_pri = x.size // n
    x = x[:n_pri * n]
    pw = np.zeros(len(PW_FEATURE_NAMES))
    bw = np.zeros(len(BW_FEATURE_NAMES))
    info = dict(detected=False, contrast_db=0.0, duty=0.0, f_offset_khz=0.0)
    if n_pri < 1:
        return pw, bw, info
    ts_us = 1e6 / fs

    # ---- fold the PRIs: pulses line up because PRI is fixed, averaging 3 pulses
    power = x.real**2 + x.imag**2
    folded = power.reshape(n_pri, n).mean(axis=0)
    sm = _circular_smooth(folded, smooth)
    lo0, hi0 = np.percentile(sm, [1.0, 99.5])
    if not (hi0 > 0 and hi0 > 1.5 * lo0):
        return pw, bw, info
    on = sm > 0.5 * (lo0 + hi0)
    core_on = minimum_filter1d(on, 2 * smooth + 1, mode="wrap")
    core_off = minimum_filter1d(~on, 2 * smooth + 1, mode="wrap")
    if not core_on.any() or not core_off.any():
        return pw, bw, info
    hi, lo = folded[core_on].mean(), folded[core_off].mean()
    if not hi > lo:
        return pw, bw, info
    span = hi - lo

    # ---- PW: energy, interpolated 50% edges and multi-level counts
    pw_energy = np.sum(folded - lo) / span * ts_us
    counts = [np.count_nonzero(sm > lo + q * span) * ts_us for q in (0.25, 0.5, 0.75)]
    # Edges from a heavier (still symmetric) smoothing: noise must not split the pulse,
    # and a symmetric moving average does not move the 50% crossing of a step.
    thr = lo + 0.5 * span
    se = _circular_smooth(folded, 4 * smooth)
    run = _longest_circular_run(se > thr)
    if run is None:
        return pw, bw, info
    r, length = run
    a, b = se[(r - 1) % n], se[r]
    c, d = se[(r + length - 1) % n], se[(r + length) % n]
    frac_r = (thr - a) / (b - a) if b > a else 0.5
    frac_f = (c - thr) / (c - d) if c > d else 0.5
    pw_edge = (length + frac_f - frac_r) * ts_us
    contrast_db = 10 * np.log10(hi / max(lo, 1e-30))
    pw[:] = (pw_energy, pw_edge, *counts, contrast_db)
    info.update(detected=bool(contrast_db >= MIN_CONTRAST_DB), contrast_db=float(contrast_db),
                duty=float(pw_energy / (n * ts_us)))

    # ---- BW: chirp rate x PW, plus spectral width, over the pulse interiors
    if length - 2 * guard < 32:
        return pw, bw, info
    keep = np.zeros(n, dtype=bool)
    keep[(r + np.arange(guard, length - guard)) % n] = True
    mask = np.tile(keep, n_pri)
    segments = _runs(mask, 32)
    if not segments:
        return pw, bw, info
    pair = mask[1:] & mask[:-1]
    f_c = np.angle(np.sum((x[1:] * np.conj(x[:-1]))[pair])) * fs / (2 * np.pi)
    y = x * np.exp(-2j * np.pi * f_c / fs * np.arange(x.size))
    ys = np.convolve(y, np.ones(IQ_SMOOTH) / IQ_SMOOTH, mode="same")
    inner = [(s + IQ_SMOOTH, e - IQ_SMOOTH) for s, e in segments if e - s > 32 + 2 * IQ_SMOOTH]
    kq1 = _pooled_quadratic_phase(y, segments, fs)
    kqs = _pooled_quadratic_phase(ys, inner, fs) if inner else kq1
    kif1 = _pooled_if_slope(y, segments, 1, fs)
    kifl = _pooled_if_slope(ys, inner, IF_LAG, fs) if inner else kif1

    gated = np.where(mask, y, 0)
    nfft = 1 << int(np.ceil(np.log2(x.size)))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(gated, nfft)))**2
    # Smooth the periodogram, subtract the known noise floor (off-pulse power x gated
    # samples) and keep only bins clearly above it, so noise does not widen the estimate.
    spectrum = np.convolve(spectrum, np.ones(SPECTRUM_SMOOTH) / SPECTRUM_SMOOTH, mode="same")
    floor = np.count_nonzero(mask) * lo
    spectrum -= floor
    spectrum[spectrum < 3 * floor / np.sqrt(SPECTRUM_SMOOTH)] = 0
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)) * 1e-3          # kHz
    total = spectrum.sum()
    obw90 = obw98 = rms_bw = centroid = 0.0
    if total > 0:
        centroid = float(freqs @ spectrum / total)
        rms_bw = np.sqrt(12 * ((freqs - centroid)**2 @ spectrum) / total)
        cdf = np.cumsum(spectrum) / total
        q = np.interp([0.01, 0.05, 0.95, 0.99], cdf, freqs)
        obw98, obw90 = q[3] - q[0], q[2] - q[1]
    # BW = |chirp rate| x PW; the edge-based PW stays unbiased at low SNR.
    bw[:] = (abs(kq1) * pw_edge, abs(kqs) * pw_edge, abs(kif1) * pw_edge,
             abs(kifl) * pw_edge, abs(kqs), obw90, obw98, rms_bw, pw_edge)
    info["f_offset_khz"] = float(f_c * 1e-3 + centroid)
    return pw, bw, info


# ============================================================
# FEATURE ASSEMBLY + SEPARATE RIDGE READOUTS
# ============================================================
def _poly2(v):
    """[v, v_i*v_j (i<=j)] row-wise; v is (n, d)."""
    i, j = np.triu_indices(v.shape[1])
    return np.hstack([v, v[:, i] * v[:, j]])


def assemble(features, groups):
    """Concatenate feature groups for one readout. Accepts 1-D or (n, d) groups."""
    parts = []
    for g in groups:
        v = np.atleast_2d(np.asarray(features[g], dtype=np.float64))
        parts.append(_poly2(v) if g in PHYSICAL_GROUPS else v)
    return np.hstack(parts)


def ridge_solve(X_aug, y, alpha):
    """Closed-form ridge; bias (column 0) is not penalised."""
    reg = alpha * np.eye(X_aug.shape[1])
    reg[0, 0] = 0
    return np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)


class Readout:
    """One parameter's own feature set, normalisation and weight vector."""
    def __init__(self, name, groups, w, x_mean, x_std, y_mean, y_std, alpha):
        self.name, self.groups = name, tuple(groups)
        self.w = np.asarray(w, dtype=np.float64).reshape(-1)
        self.x_mean = np.asarray(x_mean, dtype=np.float64).reshape(-1)
        self.x_std = np.asarray(x_std, dtype=np.float64).reshape(-1)
        self.y_mean, self.y_std, self.alpha = float(y_mean), float(y_std), float(alpha)

    @classmethod
    def fit(cls, name, groups, features, y, alpha):
        X = assemble(features, groups)
        x_mean, x_std = X.mean(axis=0), X.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        y_mean, y_std = float(np.mean(y)), max(float(np.std(y)), 1e-12)
        X_aug = np.column_stack([np.ones(len(X)), (X - x_mean) / x_std])
        w = ridge_solve(X_aug, (np.asarray(y) - y_mean) / y_std, alpha)
        return cls(name, groups, w, x_mean, x_std, y_mean, y_std, alpha)

    def predict(self, features):
        X = (assemble(features, self.groups) - self.x_mean) / self.x_std
        out = (X @ self.w[1:] + self.w[0]) * self.y_std + self.y_mean
        return out if out.size > 1 else float(out[0])


def extract_features(x, samples_per_pri, fs, reservoir=None):
    pw, bw, info = pulse_features(x, samples_per_pri, fs)
    features = dict(pw=pw, bw=bw)
    if reservoir is not None:
        features["res"] = reservoir.energy(x)
    return features, info


def save_model(path, readouts, fs, samples_per_pri, reservoir_arrays=None, **extra):
    arrays = dict(model_format=np.array(MODEL_FORMAT), Fs=fs, samples_per_pri=samples_per_pri,
                  targets=np.array([r.name for r in readouts]))
    for r in readouts:
        arrays.update({f"{r.name}_W": r.w, f"{r.name}_X_mean": r.x_mean, f"{r.name}_X_std": r.x_std,
                       f"{r.name}_Y_mean": r.y_mean, f"{r.name}_Y_std": r.y_std,
                       f"{r.name}_alpha": r.alpha, f"{r.name}_groups": np.array(r.groups)})
    arrays.update(reservoir_arrays or {})
    arrays.update(extra)
    np.savez(path, **arrays)


class EsnModel:
    """Loads a model saved by the no-Phi TrainESN_1.py and predicts PW (us) and BW (kHz) from raw I/Q."""
    def __init__(self, readouts, fs, samples_per_pri, reservoir=None):
        self.readouts, self.fs, self.samples_per_pri = readouts, fs, samples_per_pri
        self.reservoir = reservoir
        self.needs_reservoir = any("res" in r.groups for r in readouts.values())
        if self.needs_reservoir and reservoir is None:
            raise ValueError("A readout uses reservoir features but the model has no reservoir.")
        self._check_dimensions()

    @classmethod
    def load(cls, path, samples_per_pri=None):
        with np.load(path, allow_pickle=False) as archive:
            saved = {name: archive[name].copy() for name in archive.files}
        fs = float(saved["Fs"].item())
        reservoir = None
        if "Phi_local" in saved or str(saved.get("model_format", "")) != MODEL_FORMAT:
            raise ValueError("This model was trained with Phi compression (or is not from this version). "
                             "Retrain with redar_esn_no_phi/bladerf/TrainESN_1.py, or use the compressed "
                             "version's dashboard.")
        if all(k in saved for k in ("Win", "Wres")):
            reservoir = Reservoir(saved["Win"], saved["Wres"])
        readouts = {}
        for name in saved["targets"].tolist():
            readouts[name] = Readout(name, saved[f"{name}_groups"].tolist(), saved[f"{name}_W"],
                                     saved[f"{name}_X_mean"], saved[f"{name}_X_std"],
                                     saved[f"{name}_Y_mean"], saved[f"{name}_Y_std"],
                                     saved[f"{name}_alpha"])
        spp = int(saved["samples_per_pri"].item())
        if samples_per_pri is not None and spp != samples_per_pri:
            raise ValueError(f"Model was trained for {spp} samples/PRI; RX uses {samples_per_pri}.")
        if sorted(readouts) != sorted(TARGETS):
            raise ValueError("Model must contain separate PW and BW readouts.")
        return cls(readouts, fs, spp, reservoir)

    def _check_dimensions(self):
        dummy = dict(pw=np.zeros(len(PW_FEATURE_NAMES)), bw=np.zeros(len(BW_FEATURE_NAMES)))
        if self.reservoir is not None:
            dummy["res"] = np.zeros(self.reservoir.neurons)
        for r in self.readouts.values():
            d = assemble(dummy, r.groups).shape[1]
            if r.x_mean.size != d or r.x_std.size != d or r.w.size != d + 1:
                raise ValueError(f"{r.name} readout expects {r.x_mean.size} features; model gives {d}.")
            for array in (r.w, r.x_mean, r.x_std):
                if not np.isfinite(array).all():
                    raise ValueError(f"{r.name} readout contains non-finite values.")
            if np.any(r.x_std <= 0) or r.y_std <= 0:
                raise ValueError(f"{r.name} readout standard deviations must be positive.")

    def describe(self):
        parts = [f"{r.name}: {'+'.join(r.groups)}" + ("" if np.isnan(r.alpha) else f" (alpha {r.alpha:.1e})")
                 for r in self.readouts.values()]
        res = f"{self.reservoir.neurons} neurons, {self.reservoir.backend}" if self.needs_reservoir else "no reservoir"
        return "Separate readouts (no Phi, raw-sample reservoir); " + "; ".join(parts) + f"; {res}"

    def predict(self, x):
        features, info = extract_features(x, self.samples_per_pri, self.fs,
                                          self.reservoir if self.needs_reservoir else None)
        return {name: r.predict(features) for name, r in self.readouts.items()}, info
