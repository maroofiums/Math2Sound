"""Math2Sound CLI.  Example:  python app.py "sin(x)" --xmin 0 --xmax 880*pi"""
import argparse
import ast
import math

import numpy as np

from src.audio import normalize, play, to_pcm16, write_wav
from src.parser import parse_expression
from src.signal_gen import generate_signal


def number(s: str) -> float:
    """Accept '3.14', '2*pi', '-1' etc. (reuses the safe parser)."""
    return float(parse_expression(s)(np.array([0.0]))[0])


def plot(t, x, y, y_norm, fs) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(x, y); ax[0].set_title("f(x)")
    n = min(len(t), 2000)
    ax[1].plot(t[:n], y_norm[:n]); ax[1].set_title("waveform (first samples)")
    spec = np.abs(np.fft.rfft(y_norm))
    ax[2].plot(np.fft.rfftfreq(len(y_norm), 1 / fs), spec)
    ax[2].set_xlim(0, 5000); ax[2].set_title("spectrum (Hz)")
    plt.tight_layout(); plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a math expression to audio")
    p.add_argument("expression")
    p.add_argument("--xmin", default="0")
    p.add_argument("--xmax", default="2*pi*440*2",
                   help="default gives 440 cycles/s for sin(x) over 2 s")
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--fs", type=int, default=44100)
    p.add_argument("--out", default="output.wav")
    p.add_argument("--play", action="store_true")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    f = parse_expression(a.expression)
    t, x, y = generate_signal(f, number(a.xmin), number(a.xmax), a.duration, a.fs)
    y_norm = normalize(y)
    write_wav(a.out, to_pcm16(y_norm), a.fs)
    print(f"Wrote {a.out}: {len(y)} samples, {a.fs} Hz, "
          f"{a.duration}s, non-finite values: {int((~np.isfinite(y)).sum())}")
    if a.plot:
        plot(t, x, y, y_norm, a.fs)
    if a.play:
        play(a.out)


if __name__ == "__main__":
    main()
