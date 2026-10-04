"""One function that runs the whole pipeline; shared by the API and the UI.

expression -> f(x) -> x(t) -> y -> sanitize/normalize -> PCM16 -> WAV (+ stats, plots data)
"""
import numpy as np

from .audio import normalize, to_pcm16, wav_bytes
from .dsp import dominant_frequency, spectrum
from .latex import to_latex
from .parser import parse_expression
from .signal_gen import generate_signal

ALLOWED_FS = (8000, 22050, 44100, 48000)


def evaluate_number(text: str) -> float:
    """Evaluate a constant such as '2*pi*440*2' with the same safe parser."""
    v = float(parse_expression(text)(np.array([0.0]))[0])
    if not np.isfinite(v):
        raise ValueError(f"{text!r} is not a finite number")
    return v


def _jsonable(a) -> list:
    return [float(v) if np.isfinite(v) else None for v in a]


def synthesize(expression: str, x_min: str = "0", x_max: str = "2*pi*440*2",
               duration: float = 2.0, fs: int = 44100, remove_dc: bool = True,
               window_ms: float = 20.0, max_curve_points: int = 1500) -> dict:
    if not 0 < duration <= 10:
        raise ValueError("duration must be in (0, 10] seconds")
    if fs not in ALLOWED_FS:
        raise ValueError(f"fs must be one of {ALLOWED_FS}")
    f = parse_expression(expression)
    a, b = evaluate_number(x_min), evaluate_number(x_max)
    if a == b:
        raise ValueError("x_min and x_max must differ")

    t, x, y = generate_signal(f, a, b, duration, fs)
    n_total = len(y)
    finite = np.isfinite(y)
    raw_peak = float(np.max(np.abs(y[finite]))) if finite.any() else 0.0

    y_norm = normalize(y, remove_dc)
    pcm = to_pcm16(y_norm)
    freqs, mag = spectrum(pcm / 32767.0, fs)
    dom = dominant_frequency(freqs, mag)

    idx = np.linspace(0, n_total - 1, min(max_curve_points, n_total)).astype(int)
    n_win = n_total if window_ms <= 0 else max(2, min(n_total, int(fs * window_ms / 1000)))
    step = max(1, n_win // 3000)
    nyq = fs / 2
    fmax = float(np.clip(dom * 8, 1000, nyq))
    keep = freqs <= fmax
    keep[0] = False                                   # hide the DC bin
    sin_hz = abs(b - a) / (2 * np.pi * duration)      # frequency of sin(x) under this mapping

    return {
        "latex": to_latex(expression),
        "n_samples": n_total, "fs": fs, "duration": duration,
        "nyquist": nyq, "nonfinite": int((~finite).sum()), "raw_peak": raw_peak,
        "pcm_min": int(pcm.min()), "pcm_max": int(pcm.max()),
        "dominant_hz": dom, "sin_hz": float(sin_hz), "aliased": bool(sin_hz > nyq),
        "curve_x": _jsonable(x[idx]), "curve_y": _jsonable(y[idx]),
        "wave_t_ms": _jsonable(t[:n_win:step] * 1000), "wave_y": _jsonable(y_norm[:n_win:step]),
        "spec_f": _jsonable(freqs[keep]), "spec_mag": _jsonable(mag[keep] / max(mag[keep].max(initial=0), 1e-12)),
        "wav": wav_bytes(pcm, fs),
    }
