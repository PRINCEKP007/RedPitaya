"""Real-signal ESN core shared by TrainESN_Real.py and RedPitaya_ESN_Dashboard.py.

The Red Pitaya records the photodetector output of the atomic vapour cell: a
REAL voltage, not I/Q. There is no carrier phase, so the chirp is only visible
through how the atoms respond while it sweeps across the resonance:

* PW  - from the pulse edges of the PRI-folded detector trace.
* BW  - the response peaks while the instantaneous frequency is within the
        atomic linewidth G of resonance, i.e. for ~G/BW of the pulse. Width,
        position and shape of the in-pulse response therefore encode BW.
* res - a real-valued ESN (tanh reservoir) driven by every sample of the
        standardised trace (no-Phi version: no compression before the reservoir).

Each parameter has its own readout (feature set, normalisation, ridge alpha,
weights). PRI is fixed (512 us); a window holds whole PRIs, so folding the
window into one PRI averages the pulses and removes the unknown capture phase.

numpy/scipy only (numba optional) so the real-time UI can import it.
"""
import numpy as np
from scipy.sparse import csr_matrix

try:
    from numba import njit
except ImportError:
    njit = None

MODEL_FORMAT = "rp-real-esn-v1-nophi"     # reservoir driven by raw samples, no Phi
TARGETS = ("PW", "BW")                       # PW in us, BW in kHz
PHYSICAL_GROUPS = ("pw", "bw")               # expanded with 2nd-order terms
N_SHAPE = 32                                 # resampled in-pulse response points
MIN_SNR_DB = 6.0                             # in-pulse response / off-pulse noise
PW_FEATURE_NAMES = ("pw_deriv_us", "pw_edge50_us", "pw_thr10_us", "pw_thr50_us",
                    "snr_db", "rise_level_rel", "fall_level_rel")
BW_FEATURE_NAMES = ("fwhm_u", "inv_fwhm_u", "fwhm_minrel_u", "inv_fwhm_minrel_u",
                    "rms_width_u", "inv_rms_width_u", "peak_u", "centroid_u",
                    "fill_factor", "pw_edge50_us")


# ============================================================
# REAL RESERVOIR
# ============================================================
def _real_kernel(u, win, data, indices, indptr):
    neurons = win.shape[0]
    state = np.zeros(neurons)
    s1 = np.zeros(neurons)
    s2 = np.zeros(neurons)
    for t in range(u.shape[0]):
        nxt = np.empty(neurons)
        for row in range(neurons):
            z = 0.0
            for k in range(indptr[row], indptr[row + 1]):
                z += data[k] * state[indices[k]]
            for c in range(win.shape[1]):
                z += win[row, c] * u[t, c]
            v = np.tanh(z)
            nxt[row] = v
            s1[row] += v
            s2[row] += v * v
        state = nxt
    return np.concatenate((s1 / u.shape[0], s2 / u.shape[0]))


_compiled_kernel = njit(cache=True, nogil=True)(_real_kernel) if njit is not None else None


def create_real_reservoir(n_res, density, radius, input_scale, rng):
    """Random real Win (one input: the raw sample) and sparse Wres at the given spectral radius."""
    wres = np.zeros((n_res, n_res))
    mask = rng.random((n_res, n_res)) < density
    wres[mask] = rng.standard_normal(mask.sum())
    current = np.max(np.abs(np.linalg.eigvals(wres)))
    if current > 0:
        wres *= radius / current
    win = input_scale * rng.standard_normal((n_res, 1))
    return win, wres


class RealReservoir:
    """Mean and mean-square of each neuron's state (2 x neurons features).

    NO-PHI VERSION: driven sample-by-sample by the standardised raw trace
    (no Phi measurement matrix, no block/stride compression)."""
    def __init__(self, win, wres):
        self.win = np.ascontiguousarray(np.asarray(win, dtype=np.float64).reshape(-1, 1))
        self.wres = csr_matrix(np.asarray(wres, dtype=np.float64))
        self.neurons = self.wres.shape[0]
        if self.win.shape[0] != self.neurons:
            raise ValueError("Win must have one row per reservoir neuron.")
        self.backend = "Numba" if _compiled_kernel is not None else "SciPy sparse"

    def features(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = (x - x.mean()) / (x.std() + 1e-12)           # gain/offset independent input
        u = np.ascontiguousarray(x.reshape(-1, 1))      # every raw sample is one input step
        if _compiled_kernel is not None:
            return _compiled_kernel(u, self.win, self.wres.data, self.wres.indices, self.wres.indptr)
        state = np.zeros(self.neurons)
        s1 = np.zeros(self.neurons)
        s2 = np.zeros(self.neurons)
        for row in u:
            state = np.tanh(self.wres @ state + self.win @ row)
            s1 += state
            s2 += state * state
        return np.r_[s1, s2] / len(u)


# ============================================================
# PHYSICAL FEATURES OF THE REAL DETECTOR TRACE
# ============================================================
def _circular_smooth(v, n):
    if n <= 1:
        return v.copy()
    padded = np.r_[v[v.size - n // 2:], v, v[:n - 1 - n // 2]]
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def _parabolic(v, i):
    """Sub-sample offset of the peak of v at index i (circular)."""
    a, b, c = v[(i - 1) % v.size], v[i], v[(i + 1) % v.size]
    den = a - 2 * b + c
    return 0.5 * (a - c) / den if den < 0 else 0.0


def _crossing(sm, thr, centre, half, rising):
    """Fractional circular position nearest `centre` where sm crosses thr."""
    n = sm.size
    best = None
    c = int(round(centre))
    for i in range(c - half, c + half):
        a, b = sm[i % n], sm[(i + 1) % n]
        hit = (a < thr <= b) if rising else (a >= thr > b)
        if hit:
            pos = i + (thr - a) / (b - a)
            if best is None or abs(pos - centre) < abs(best - centre):
                best = pos
    return centre if best is None else best


def _width_at(p, u, level, i):
    """Width (in u) of the contiguous region around index i where p >= level."""
    left = i
    while left > 0 and p[left - 1] >= level:
        left -= 1
    right = i
    while right < p.size - 1 and p[right + 1] >= level:
        right += 1
    ul = u[left] if left == 0 else u[left - 1] + (level - p[left - 1]) / (p[left] - p[left - 1]) * (u[left] - u[left - 1])
    ur = u[right] if right == p.size - 1 else u[right] + (p[right] - level) / (p[right] - p[right + 1]) * (u[right + 1] - u[right])
    return max(ur - ul, 1e-3)


def _step_detector(s, half):
    """Circular mean(next `half` samples) - mean(previous `half`): a matched filter for
    a level step, far less noise-sensitive than a sample derivative."""
    c = np.cumsum(np.r_[s, s, s])
    n = s.size
    i = np.arange(n) + n
    return ((c[i + half - 1] - c[i - 1]) - (c[i - 1] - c[i - half - 1])) / half


def _best_edge_pair(dd, lmin, lmax):
    """Rise r and fall f maximising dd[r] - dd[f] with (f - r) mod n in [lmin, lmax]."""
    n = dd.size
    ext = np.r_[dd, dd]
    width = lmax - lmin + 1
    windows = np.lib.stride_tricks.sliding_window_view(ext[lmin:lmin + n + width - 1], width)
    k = np.argmin(windows, axis=1)                       # best fall for each rise
    score = dd - windows[np.arange(n), k]
    r = int(np.argmax(score))
    return r, (r + lmin + int(k[r])) % n


def real_pulse_features(x, samples_per_pri, fs, polarity=1, edge_smooth=4, edge_window=20,
                        pw_range_us=None):
    """PW, BW and shape features from one window of whole PRIs of real samples.

    polarity: +1 if the pulse raises the detector level, -1 if it lowers it.
    edge_smooth: samples averaged for level crossings and the response shape.
    edge_window: samples averaged on each side of a candidate edge to find it.
    pw_range_us: (min, max) pulse width to search; the rise and fall are then picked
                 as a pair, so a weak pulse edge cannot lose to a stronger in-pulse slope.
    Returns (pw, bw, shape, info); vectors are zero when no pulse is found.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(samples_per_pri)
    n_pri = x.size // n
    pw = np.zeros(len(PW_FEATURE_NAMES))
    bw = np.zeros(len(BW_FEATURE_NAMES))
    shape = np.zeros(N_SHAPE)
    info = dict(detected=False, snr_db=0.0, rise=np.nan, fall=np.nan, base=0.0,
                polarity=int(polarity), folded=None)
    if n_pri < 1:
        return pw, bw, shape, info
    ts_us = 1e6 / fs
    folded = x[:n_pri * n].reshape(n_pri, n).mean(axis=0)     # average the pulses
    info["folded"] = folded
    s = polarity * folded                                      # pulse is positive-going
    k = max(1, int(edge_smooth))
    sm = _circular_smooth(s, k)

    # ---- edges: strongest rising and falling level steps of the folded trace
    L = max(2, int(edge_window))
    dd = _step_detector(s, L)
    if pw_range_us is not None:
        lmin = max(1, int(np.floor(pw_range_us[0] * 1e-6 * fs)))
        lmax = min(n - 1, int(np.ceil(pw_range_us[1] * 1e-6 * fs)))
        r0, f0 = _best_edge_pair(dd, lmin, max(lmin, lmax))
    else:
        r0, f0 = int(np.argmax(dd)), int(np.argmin(dd))
    rise_d = r0 + _parabolic(dd, r0)
    fall_d = f0 + _parabolic(-dd, f0)
    length = (fall_d - rise_d) % n
    g = max(3 * k + 2, L // 2)
    if length < 4 * g or n - length < 4 * g:
        return pw, bw, shape, info
    off_idx = (int(round(fall_d)) + g + np.arange(int(n - length) - 2 * g)) % n
    on_idx = (int(round(rise_d)) + g + np.arange(int(length) - 2 * g)) % n
    base = float(np.median(s[off_idx]))
    noise = 1.4826 * float(np.median(np.abs(s[off_idx] - base))) + 1e-15
    on_dev = sm[on_idx] - base
    peak = float(on_dev.max())
    if not peak > 0:
        return pw, bw, shape, info
    snr_db = 20 * np.log10(peak / noise)

    # ---- PW: derivative edges, 50%-of-edge-step crossings, threshold widths
    lvl_r = float(np.mean(s[(int(round(rise_d)) + g + np.arange(L)) % n])) - base
    lvl_f = float(np.mean(s[(int(round(fall_d)) - g - np.arange(1, L + 1)) % n])) - base
    rise50 = _crossing(sm, base + 0.5 * lvl_r, rise_d, g, rising=True) if lvl_r > 0 else rise_d
    fall50 = _crossing(sm, base + 0.5 * lvl_f, fall_d, g, rising=False) if lvl_f > 0 else fall_d
    pw_edge50 = ((fall50 - rise50) % n) * ts_us
    dev = sm - base
    pw[:] = (length * ts_us, pw_edge50,
             np.count_nonzero(dev > 0.1 * peak) * ts_us, np.count_nonzero(dev > 0.5 * peak) * ts_us,
             snr_db, lvl_r / peak, lvl_f / peak)
    info.update(detected=bool(snr_db >= MIN_SNR_DB), snr_db=float(snr_db),
                rise=float(rise50 % n), fall=float(fall50 % n), base=float(polarity * base))

    # ---- BW: shape of the in-pulse response versus normalised time u in [0, 1]
    span = (fall50 - rise50) % n
    m = max(int(span), 8)
    pos = (rise50 + np.linspace(0, span, m)) % n
    profile = (np.interp(pos, np.arange(n + 1), np.r_[sm, sm[0]]) - base) / peak
    u = np.linspace(0.0, 1.0, m)
    shape[:] = np.interp(np.linspace(0, 1, N_SHAPE), u, profile)
    i = int(np.argmax(profile))
    fwhm = _width_at(profile, u, 0.5, i)
    pmin = float(profile.min())
    rel = (profile - pmin) / max(1.0 - pmin, 1e-9)
    fwhm_rel = _width_at(rel, u, 0.5, i)
    w = np.clip(rel, 0, None)
    w_sum = max(float(w.sum()), 1e-12)
    centroid = float(u @ w / w_sum)
    rms = max(float(np.sqrt(((u - centroid)**2) @ w / w_sum)), 1e-3)
    bw[:] = (fwhm, 1 / fwhm, fwhm_rel, 1 / fwhm_rel, rms, 1 / rms, u[i], centroid,
             float(profile.mean()), pw_edge50)
    return pw, bw, shape, info


def estimate_polarity(windows, pw_true_us, samples_per_pri, fs, edge_smooth, edge_window, pw_range_us=None):
    """+1/-1: whichever polarity makes the measured PW agree with the labels more often."""
    votes = 0
    for x, truth in zip(windows, pw_true_us):
        err = [abs(real_pulse_features(x, samples_per_pri, fs, p, edge_smooth, edge_window,
                                       pw_range_us)[0][1] - truth)
               for p in (1, -1)]
        votes += 1 if err[0] <= err[1] else -1
    return 1 if votes >= 0 else -1


# ============================================================
# FEATURE ASSEMBLY + SEPARATE RIDGE READOUTS
# ============================================================
def _poly2(v):
    i, j = np.triu_indices(v.shape[1])
    return np.hstack([v, v[:, i] * v[:, j]])


def assemble(features, groups):
    parts = []
    for g in groups:
        v = np.atleast_2d(np.asarray(features[g], dtype=np.float64))
        parts.append(_poly2(v) if g in PHYSICAL_GROUPS else v)
    return np.hstack(parts)


def ridge_solve(X_aug, y, alpha):
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
        return cls(name, groups, ridge_solve(X_aug, (np.asarray(y) - y_mean) / y_std, alpha),
                   x_mean, x_std, y_mean, y_std, alpha)

    def predict(self, features):
        X = (assemble(features, self.groups) - self.x_mean) / self.x_std
        out = (X @ self.w[1:] + self.w[0]) * self.y_std + self.y_mean
        return out if out.size > 1 else float(out[0])


def extract_features(x, samples_per_pri, fs, polarity, edge_smooth, edge_window, pw_range_us,
                     reservoir=None):
    pw, bw, shape, info = real_pulse_features(x, samples_per_pri, fs, polarity, edge_smooth,
                                              edge_window, pw_range_us)
    features = dict(pw=pw, bw=bw, shape=shape)
    if reservoir is not None:
        features["res"] = reservoir.features(x)
    return features, info


def save_model(path, readouts, fs, samples_per_pri, polarity, edge_smooth, edge_window,
               pw_range_us, reservoir_arrays=None, **extra):
    arrays = dict(model_format=np.array(MODEL_FORMAT), Fs=fs, samples_per_pri=samples_per_pri,
                  polarity=polarity, edge_smooth=edge_smooth, edge_window=edge_window,
                  pw_range_us=np.asarray(pw_range_us, dtype=np.float64),
                  targets=np.array([r.name for r in readouts]))
    for r in readouts:
        arrays.update({f"{r.name}_W": r.w, f"{r.name}_X_mean": r.x_mean, f"{r.name}_X_std": r.x_std,
                       f"{r.name}_Y_mean": r.y_mean, f"{r.name}_Y_std": r.y_std,
                       f"{r.name}_alpha": r.alpha, f"{r.name}_groups": np.array(r.groups)})
    arrays.update(reservoir_arrays or {})
    arrays.update(extra)
    np.savez(path, **arrays)


class RealEsnModel:
    """Model saved by TrainESN_Real.py: PW (us) and BW (kHz) from a real trace."""
    def __init__(self, readouts, fs, samples_per_pri, polarity, edge_smooth, edge_window, pw_range_us,
                 reservoir=None):
        self.readouts, self.fs, self.samples_per_pri = readouts, float(fs), int(samples_per_pri)
        self.polarity, self.edge_smooth, self.edge_window = int(polarity), int(edge_smooth), int(edge_window)
        self.pw_range_us = tuple(float(v) for v in pw_range_us)
        self.reservoir = reservoir
        self.needs_reservoir = any("res" in r.groups for r in readouts.values())
        if self.needs_reservoir and reservoir is None:
            raise ValueError("A readout uses reservoir features but the model has no reservoir.")
        if sorted(readouts) != sorted(TARGETS):
            raise ValueError("Model must contain separate PW and BW readouts.")
        self._check_dimensions()

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            saved = {name: archive[name].copy() for name in archive.files}
        if str(saved.get("model_format", "")) != MODEL_FORMAT:
            raise ValueError("Not a TrainESN_Real.py model (expected format " + MODEL_FORMAT + ").")
        reservoir = None
        if all(k in saved for k in ("Win", "Wres")):
            reservoir = RealReservoir(saved["Win"], saved["Wres"])
        readouts = {name: Readout(name, saved[f"{name}_groups"].tolist(), saved[f"{name}_W"],
                                  saved[f"{name}_X_mean"], saved[f"{name}_X_std"],
                                  saved[f"{name}_Y_mean"], saved[f"{name}_Y_std"], saved[f"{name}_alpha"])
                    for name in saved["targets"].tolist()}
        return cls(readouts, saved["Fs"].item(), saved["samples_per_pri"].item(),
                   saved["polarity"].item(), saved["edge_smooth"].item(), saved["edge_window"].item(),
                   saved["pw_range_us"], reservoir)

    def _check_dimensions(self):
        dummy = dict(pw=np.zeros(len(PW_FEATURE_NAMES)), bw=np.zeros(len(BW_FEATURE_NAMES)),
                     shape=np.zeros(N_SHAPE))
        if self.reservoir is not None:
            dummy["res"] = np.zeros(2 * self.reservoir.neurons)
        for r in self.readouts.values():
            d = assemble(dummy, r.groups).shape[1]
            if r.x_mean.size != d or r.x_std.size != d or r.w.size != d + 1:
                raise ValueError(f"{r.name} readout expects {r.x_mean.size} features; model gives {d}.")
            if not all(np.isfinite(a).all() for a in (r.w, r.x_mean, r.x_std)):
                raise ValueError(f"{r.name} readout contains non-finite values.")

    def describe(self):
        parts = [f"{r.name}: {'+'.join(r.groups)} (alpha {r.alpha:.1e})" for r in self.readouts.values()]
        res = (f"{self.reservoir.neurons} neurons, {self.reservoir.backend}" if self.needs_reservoir
               else "no reservoir")
        pol = "pulse raises level" if self.polarity > 0 else "pulse lowers level"
        return "Separate readouts; " + "; ".join(parts) + f"; {res}; {pol}"

    def predict(self, x):
        features, info = extract_features(x, self.samples_per_pri, self.fs, self.polarity, self.edge_smooth,
                                          self.edge_window, self.pw_range_us,
                                          self.reservoir if self.needs_reservoir else None)
        return {name: r.predict(features) for name, r in self.readouts.items()}, info
