"""Physical reservoir computing: the atomic receiver IS the reservoir.

The RF chirp drives the atoms; their (nonlinear, history-dependent) optical
response, sampled by the Red Pitaya, is taken as the reservoir state. Only a
LINEAR readout is trained on it. No digital reservoir is inserted: a digital
transform after acquisition cannot add information that the measurement lacks,
so the question is whether the measured states themselves separate the RF
parameters.

Window phase. The live window (3 PRI) starts anywhere in the pulse train - it
may hold two complete pulses and parts of a third. Because the PRI is fixed,
folding the window into one PRI and aligning it (circular cross-correlation with
a template learned from the training data) gives the same representation for any
start phase. Training and live data therefore see identical states.

Representations (each target picks its own by grouped CV):
  "aligned"  : folded + aligned PRI, averaged into `bin` sample "virtual nodes"
  "spectrum" : |FFT| of the folded PRI (shift invariant without alignment)

Targets, each with its OWN readout: PW (us), BW (kHz) and the chirp rate
K = BW/PW (kHz/us = 1e9 Hz/s) estimated DIRECTLY, not derived from PW and BW.
"""
import numpy as np

from rp_core import Readout, assemble

MODEL_FORMAT = "rp-physical-rc-v1"
TARGETS = ("PW", "BW", "K")
UNITS = {"PW": "us", "BW": "kHz", "K": "kHz/us"}
MIN_MATCH = 0.5          # correlation with the template below this: "no pulse / unfamiliar"


def fold(x, samples_per_pri):
    x = np.asarray(x, dtype=np.float64)
    n_pri = x.size // samples_per_pri
    if n_pri < 1:
        raise ValueError("Window is shorter than one PRI.")
    return x[:n_pri * samples_per_pri].reshape(n_pri, samples_per_pri).mean(axis=0)


def _normalise(f, mode):
    f = f - f.mean()
    return f / (f.std() + 1e-15) if mode == "zscore" else f


def _circular_xcorr_shift(f, template):
    """Fractional shift s such that roll(f, -s) best matches template."""
    xc = np.fft.irfft(np.fft.rfft(f) * np.conj(np.fft.rfft(template)), n=f.size)
    i = int(np.argmax(xc))
    a, b, c = xc[i - 1], xc[i], xc[(i + 1) % f.size]
    den = a - 2 * b + c
    offset = 0.5 * (a - c) / den if den < 0 else 0.0
    return i + offset


def _fractional_roll(f, shift):
    """roll(f, -shift) for non-integer shift (band-limited, circular)."""
    k = np.fft.rfftfreq(f.size) * f.size
    return np.fft.irfft(np.fft.rfft(f) * np.exp(2j * np.pi * k * shift / f.size), n=f.size)


class PhysicalStates:
    """Turns a raw window of real samples into physical-reservoir state vectors."""
    def __init__(self, samples_per_pri, bin_size, normalise, template, n_spectrum):
        self.samples_per_pri, self.bin_size = int(samples_per_pri), int(bin_size)
        self.normalise, self.n_spectrum = str(normalise), int(n_spectrum)
        self.template = np.asarray(template, dtype=np.float64)
        self.n_nodes = self.samples_per_pri // self.bin_size

    @staticmethod
    def build_template(windows, samples_per_pri, normalise, iterations=4):
        """Iteratively aligned mean of folded training windows."""
        folded = [_normalise(fold(x, samples_per_pri), normalise) for x in windows]
        template = folded[0]
        for _ in range(iterations):
            template = np.mean([_fractional_roll(f, _circular_xcorr_shift(f, template)) for f in folded], axis=0)
        return _normalise(template, "zscore")

    def states(self, x):
        folded = fold(x, self.samples_per_pri)
        f = _normalise(folded, self.normalise)
        shift = _circular_xcorr_shift(f, self.template)
        aligned_full = _fractional_roll(f, shift)
        aligned = aligned_full[:self.n_nodes * self.bin_size].reshape(self.n_nodes, self.bin_size).mean(axis=1)
        spectrum = np.abs(np.fft.rfft(f))[1:1 + self.n_spectrum] / f.size
        z = aligned_full - aligned_full.mean()
        match = float(z @ self.template / (np.linalg.norm(z) * np.linalg.norm(self.template) + 1e-15))
        info = dict(folded=folded, shift=float(shift), match=match, detected=match >= MIN_MATCH,
                    aligned=aligned_full)
        return dict(aligned=aligned, spectrum=spectrum), info


def save_model(path, readouts, states, fs, pri_us, pri_per_window, **extra):
    arrays = dict(model_format=np.array(MODEL_FORMAT), Fs=fs, samples_per_pri=states.samples_per_pri,
                  pri_us=pri_us, pri_per_window=pri_per_window, bin_size=states.bin_size,
                  normalise=np.array(states.normalise), n_spectrum=states.n_spectrum,
                  template=states.template, targets=np.array([r.name for r in readouts]))
    for r in readouts:
        arrays.update({f"{r.name}_W": r.w, f"{r.name}_X_mean": r.x_mean, f"{r.name}_X_std": r.x_std,
                       f"{r.name}_Y_mean": r.y_mean, f"{r.name}_Y_std": r.y_std,
                       f"{r.name}_alpha": r.alpha, f"{r.name}_groups": np.array(r.groups)})
    arrays.update(extra)
    np.savez(path, **arrays)


class PhysicalRCModel:
    """Linear readouts on the measured atomic-receiver states: PW, BW and K directly."""
    def __init__(self, readouts, states, fs):
        self.readouts, self.states_fn, self.fs = readouts, states, float(fs)
        self.samples_per_pri = states.samples_per_pri
        dummy = dict(aligned=np.zeros(states.n_nodes), spectrum=np.zeros(states.n_spectrum))
        for r in readouts.values():
            d = assemble(dummy, r.groups).shape[1]
            if r.x_mean.size != d or r.w.size != d + 1:
                raise ValueError(f"{r.name} readout expects {r.x_mean.size} states; model gives {d}.")
        if not {"PW", "BW"} <= set(readouts):
            raise ValueError("Model must contain PW and BW readouts.")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            saved = {name: archive[name].copy() for name in archive.files}
        if str(saved.get("model_format", "")) != MODEL_FORMAT:
            raise ValueError("Not a TrainPhysicalRC.py model.")
        states = PhysicalStates(saved["samples_per_pri"].item(), saved["bin_size"].item(),
                                str(saved["normalise"]), saved["template"], saved["n_spectrum"].item())
        readouts = {n: Readout(n, saved[f"{n}_groups"].tolist(), saved[f"{n}_W"], saved[f"{n}_X_mean"],
                               saved[f"{n}_X_std"], saved[f"{n}_Y_mean"], saved[f"{n}_Y_std"], saved[f"{n}_alpha"])
                    for n in saved["targets"].tolist()}
        return cls(readouts, states, saved["Fs"].item())

    def describe(self):
        parts = [f"{r.name}: {'+'.join(r.groups)} (alpha {r.alpha:.1e})" for r in self.readouts.values()]
        return (f"Physical RC (atomic receiver as reservoir, {self.states_fn.n_nodes} virtual nodes, "
                f"linear readouts); " + "; ".join(parts))

    def predict(self, x):
        features, info = self.states_fn.states(x)
        return {name: r.predict(features) for name, r in self.readouts.items()}, info
