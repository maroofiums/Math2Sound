"""Frequency-domain helpers.

Hann window w[n] = 0.5 - 0.5 cos(2*pi*n/(M-1)) reduces spectral leakage;
X[k] = sum_n x[n] w[n] e^{-2 pi i k n / M},  bin k <-> frequency k*Fs/M Hz.
"""
import numpy as np


def spectrum(signal: np.ndarray, fs: int, max_len: int = 65536):
    """Return (freqs_hz, magnitudes) of the first max_len samples."""
    m = min(len(signal), max_len)
    if m < 2:
        return np.array([0.0]), np.array([0.0])
    seg = np.asarray(signal[:m], dtype=float) * np.hanning(m)
    return np.fft.rfftfreq(m, 1 / fs), np.abs(np.fft.rfft(seg))


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Frequency of the strongest non-DC bin."""
    if len(mag) < 2:
        return 0.0
    return float(freqs[1 + int(np.argmax(mag[1:]))])
