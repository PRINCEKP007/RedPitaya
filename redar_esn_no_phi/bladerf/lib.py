import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
def generate_chirp_pulse(Fs, N, PW, BW, PRI):
    t = np.arange(N) / Fs
    N_pw = max(1, int(round(PW * Fs)))
    N_pw = min(N_pw, N)
    N_pri = max(1, int(round(PRI * Fs)))
    tp = np.arange(N_pw) / Fs
    M = BW / PW
    f0 = -BW / 2
    phase = 2 * np.pi * (f0 * tp +0.5 * M * tp**2)
    pulse = np.exp(1j * phase)
    x = np.zeros(N, dtype=np.complex128)
    for start in range(15, N, N_pri):
        end = min(start + N_pw, N)
        pulse_length = end - start
        x[start:end] = pulse[:pulse_length]

    return x

def add_complex_awgn(x, SNR_dB):
    signal_power = np.mean(np.abs(x)**2)
    noise_power = signal_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(x))+1j * np.random.randn(len(x)))
    return x + noise

def create_local_measurement_matrix(M_local, BLOCK_SIZE):
    Phi = (np.random.randn(M_local, BLOCK_SIZE)+1j * np.random.randn(M_local, BLOCK_SIZE))
    row_norm = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi = Phi / (row_norm + 1e-12)
    return Phi

def compress_signal(x, Phi_local, BLOCK_SIZE, STRIDE):

    measurements = []
    N = len(x)
    for start in range(0, N - BLOCK_SIZE + 1, STRIDE):
        block = x[start:start + BLOCK_SIZE]
        y_block = Phi_local @ block
        measurements.append(y_block)
    measurements = np.array(measurements)
    return measurements

def create_reservoir_matrix(N_res, density, desired_radius):
    W = np.zeros((N_res, N_res), dtype=np.complex128)
    mask = np.random.rand(N_res, N_res) < density
    W[mask] = (np.random.randn(np.sum(mask))+1j * np.random.randn(np.sum(mask)))
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius > 0:
        W = W * (desired_radius / current_radius)

    return W

def create_input_matrix(N_res, input_dimension, input_scale):    
    Win = input_scale * (np.random.randn(N_res, input_dimension)+1j * np.random.randn(N_res, input_dimension))
    return Win

def run_reservoir(Y, Win, Wres):
    N_time = Y.shape[0]
    N_res = Wres.shape[0]
    states = np.zeros((N_time, N_res),dtype=np.complex128)
    r = np.zeros(N_res, dtype=np.complex128)
    for n in range(N_time):
        u = Y[n]
        z = Wres @ r + Win @ u
        r = (np.tanh(np.real(z))+1j * np.tanh(np.imag(z)))
        states[n] = r

    return states


def energy_mean_pooling(states):
    energy = np.abs(states)**2
    features = np.mean(energy, axis=0)
    return features


def calculate_metrics(y_true, y_pred):
    error = y_pred - y_true
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    median_abs_error = np.median(np.abs(error))
    max_abs_error = np.max(np.abs(error))
    # R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return (rmse,mae,median_abs_error,max_abs_error,r2)

def envelope_autocorr_features(states, Fs, STRIDE, lag_min_us=13, lag_max_us=22):
    # energy envelope across time blocks (no averaging away the axis)
    env = np.sum(np.abs(states)**2, axis=1)          # shape (N_blocks,)
    env = env - env.mean()

    acf = np.correlate(env, env, mode='full')
    acf = acf[len(acf)//2:]                            # keep non-negative lags
    acf /= (acf[0] + 1e-12)

    block_dt = STRIDE / Fs                              # seconds per block index
    lag_lo = int(lag_min_us*1e-6 / block_dt)
    lag_hi = int(lag_max_us*1e-6 / block_dt)

    search = acf[lag_lo:lag_hi]
    peak_idx = np.argmax(search) + lag_lo
    peak_val = acf[peak_idx]

    pri_est_us = peak_idx * block_dt * 1e6              # direct PRI estimate, in us

    return np.array([pri_est_us, peak_val])             # small, physically meaningful

def pw_feature_extract(states, Fs,sts):
    # energy envelope across time blocks (no averaging away the axis)
    env = np.sum(np.abs(states)**2, axis=1)          # shape (N_blocks,)
    env = env - env.mean()
    y = np.zeros_like(env)
    y[1:] = (env[1:] - env[:-1]) * Fs

    y = abs(y)

    th = np.mean(y)
    peaks, properties = find_peaks(y, height=1e8, distance=2, prominence=0.2)

    values = properties['peak_heights']

    if np.mod(len(peaks),2) != 0:
        peak = np.zeros(len(peaks) + 1)
        peak[1::] = peaks

        value = np.zeros(len(values) + 1)
        value[1::] =  values

        value[0] = np.mean(values)
    else:
        peak = peaks
    r = int(len(peak)/2)

    peak = peak.reshape(r,2)

    tp_est = np.mean(np.abs(peak[:,0] - peak[:,1])) * (10/Fs)

    energy = np.abs(sts)**2
    Em = np.mean(energy, axis=0)

    feature = np.zeros(len(Em) + 1)
    feature[-1] = tp_est

    out = np.zeros(2)
    out[0] = tp_est
    out[1] = tp_est

    return Em


def envelope_pw_features(states, Fs, STRIDE, threshold_frac=0.5):
    # Energy pooling across ESN states
    env = np.sum(np.abs(states)**2, axis=1)

    # Normalize
    env = env / (env.max() + 1e-12)

    # Threshold
    threshold = threshold_frac

    # Find pulse samples
    pulse = env > threshold

    # Find contiguous pulse regions
    changes = np.diff(pulse.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends   = np.where(changes == -1)[0] + 1

    # Handle pulse starting/ending at boundary
    if pulse[0]:
        starts = np.r_[0, starts]
    if pulse[-1]:
        ends = np.r_[ends, len(pulse)]

    # Width of each detected pulse
    widths = ends - starts

    # Convert block count → time
    block_dt = STRIDE / Fs
    pw_us = widths * block_dt * 1e6

    # Robust representative PW
    if len(pw_us) == 0:
        return np.array([0.0, 0.0])

    pw_est_us = np.median(pw_us)

    return np.array([
        pw_est_us,
        np.mean(pw_us)
    ])


def masked_pooling(states, Y_compressed, k_thresh=3.0):
    """
    Pool reservoir states only over blocks where a pulse is actually present,
    instead of diluting with blocks that contain only noise.
    Uses the compressed measurement magnitude (closer to raw signal, less
    distorted by reservoir nonlinearity) to decide which blocks are 'on'.
    """
    env = np.abs(Y_compressed[:, 0])**2          # per-block energy of raw compressed signal
    noise_floor = np.median(env)
    mad = np.median(np.abs(env - noise_floor)) + 1e-12
    threshold = noise_floor + k_thresh * mad
    active = env > threshold
 
    if np.sum(active) == 0:                      # fallback if detector finds nothing
        active = np.ones(len(env), dtype=bool)
 
    active_energy = np.abs(states[active])**2
    masked_mean = np.mean(active_energy, axis=0)
    masked_max  = np.max(active_energy, axis=0)
 
    n_active = np.sum(active)                    # direct duty-cycle / PW-proxy feature
 
    return masked_mean, masked_max, np.array([float(n_active)])


def fm_demod(iq):
    """Polar discriminator FM demod. Returns instantaneous freq, ~[-1, 1]."""
    prod = iq[1:] * np.conj(iq[:-1])
    demod = np.angle(prod)
    return (demod / np.pi).astype(np.float32)


def deemphasis(x, fs, tau=75e-6):
    """Single-pole de-emphasis filter (75us US, 50us EU)."""
    dt = 1.0 / fs
    alpha = dt / (tau + dt)
    return signal.lfilter([alpha], [1, -(1 - alpha)], x).astype(np.float32)


def fm_to_audio(iq, fs, audio_fs=48000, tau=75e-6, channel_bw=200e3):
    """
    iq       : complex IQ samples
    fs       : IQ sample rate (Hz)
    audio_fs : desired output audio rate (Hz)
    Returns  : (audio, actual_audio_fs)
    """
    nyq = fs / 2
    sos = signal.butter(6, min(channel_bw / 2, nyq * 0.95) / nyq, btype="low", output="sos")
    iq_filt = signal.sosfilt(sos, iq)

    demod = fm_demod(iq_filt)

    decim = max(int(round(fs / audio_fs)), 1)
    audio = signal.decimate(demod, decim, ftype="fir", zero_phase=True)
    actual_fs = fs / decim

    audio = deemphasis(audio, actual_fs, tau=tau)
    audio = 0.9 * audio / (np.max(np.abs(audio)) + 1e-9)
    return audio.astype(np.float32), actual_fs