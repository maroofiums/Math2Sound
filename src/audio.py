"""Audio: sanitize -> normalize -> PCM16 -> WAV -> playback."""
import io
import os
import subprocess
import sys
import wave

import numpy as np

INT16_MAX = 32767


def sanitize(y: np.ndarray) -> np.ndarray:
    """NaN -> 0; +/-inf -> +/- largest finite magnitude (or 0 if none)."""
    y = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(y)
    peak = np.max(np.abs(y[finite])) if finite.any() else 0.0
    y[np.isnan(y)] = 0.0
    y[np.isposinf(y)] = peak
    y[np.isneginf(y)] = -peak
    return y


def normalize(y: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    """y / max|y| so that -1 <= y <= 1.  All-zero input stays zero.

    remove_dc subtracts the mean: a constant offset is inaudible (and wastes
    headroom), e.g. x^2 on [-1, 1] is mostly offset.
    """
    y = sanitize(y)
    if remove_dc and y.size:
        y = y - y.mean()
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak == 0.0 or not np.isfinite(peak):
        return np.zeros_like(y)
    return y / peak


def to_pcm16(y_norm: np.ndarray) -> np.ndarray:
    """Quantize [-1, 1] floats to int16: round(y * 32767), clipped."""
    return np.round(np.clip(y_norm, -1.0, 1.0) * INT16_MAX).astype(np.int16)


def write_wav(path, pcm: np.ndarray, fs: int = 44100) -> None:
    """Write mono 16-bit WAV using the standard library."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # bytes per sample = 16 bits
        w.setframerate(fs)
        w.writeframes(pcm.astype("<i2").tobytes())


def play(path: str) -> None:
    """Best-effort playback with the OS default player (no extra deps)."""
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
    else:
        for cmd in (["aplay", "-q"], ["paplay"], ["ffplay", "-nodisp", "-autoexit"]):
            try:
                subprocess.run(cmd + [path], check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        print(f"No player found. Open {os.path.abspath(path)} manually.")


def wav_bytes(pcm: np.ndarray, fs: int = 44100) -> bytes:
    """Same WAV as write_wav, but returned in memory (for APIs / web UIs)."""
    buf = io.BytesIO()
    write_wav(buf, pcm, fs)
    return buf.getvalue()
