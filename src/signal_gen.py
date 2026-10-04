"""Signal generation: sampling f(x(t)).

Named signal_gen.py (not signal.py) so it never shadows the stdlib `signal`.

Math:  N = round(Fs*T),  t_n = n/Fs,
       x(t) = x_min + (x_max - x_min) * t / T,   y[n] = f(x(t_n))
"""
from typing import Callable

import numpy as np


def time_axis(duration: float, fs: int) -> np.ndarray:
    n = int(round(fs * duration))
    return np.arange(n) / fs                      # t_n = n / Fs


def map_x(t: np.ndarray, x_min: float, x_max: float, duration: float) -> np.ndarray:
    return x_min + (x_max - x_min) * t / duration


def generate_signal(f: Callable[[np.ndarray], np.ndarray], x_min: float,
                    x_max: float, duration: float, fs: int = 44100):
    """Return (t, x, y) as float arrays of equal length."""
    if duration <= 0 or fs <= 0:
        raise ValueError("duration and fs must be positive")
    if x_min == x_max:
        raise ValueError("x_min and x_max must differ")
    t = time_axis(duration, fs)
    x = map_x(t, x_min, x_max, duration)
    return t, x, f(x)
