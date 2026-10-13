"""
Signal generation library for pulsed LFM chirp transmissions.
"""
import numpy as np


def Generate_Signal(Fs, PW, BW):
    """
    Generate complex baseband linear frequency modulated (LFM) up-chirp.

    Parameters:
        Fs (float): Sample rate in Hz.
        PW (float): Pulse width duration in seconds.
        BW (float): Chirp sweep bandwidth in Hz.

    Returns:
        np.ndarray: Complex baseband chirp samples of length round(PW * Fs).
    """
    n = int(round(PW * Fs))
    t = np.arange(n, dtype=np.float64) / Fs
    k = BW / PW
    phase = 2.0 * np.pi * (-0.5 * BW * t + 0.5 * k * t * t)
    return np.exp(1j * phase)
