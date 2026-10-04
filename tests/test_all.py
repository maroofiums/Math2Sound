import os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from src.parser import parse_expression
from src.signal_gen import generate_signal, time_axis, map_x
from src.audio import normalize, to_pcm16, write_wav


def test_parser_basic():
    x = np.array([0.0, 1.0, 2.0])
    assert np.allclose(parse_expression("x^2")(x), [0, 1, 4])
    assert np.allclose(parse_expression("-x + 2*pi")(x), -x + 2 * np.pi)
    assert np.allclose(parse_expression("sin(x)+cos(2*x)")(x), np.sin(x) + np.cos(2 * x))


def test_parser_rejects_unsafe():
    for bad in ["__import__('os')", "x.real", "foo(x)", "y+1", ""]:
        with pytest.raises((ValueError, SyntaxError)):
            parse_expression(bad)


def test_one_over_x_gives_inf_not_crash():
    y = parse_expression("1/x")(np.array([0.0, 1.0]))
    assert np.isinf(y[0]) and y[1] == 1


def test_sampling_and_mapping():
    assert len(time_axis(2, 44100)) == 88200
    t = time_axis(1, 4)                       # [0, .25, .5, .75]
    assert np.allclose(map_x(t, -1, 1, 1), [-1, -0.5, 0, 0.5])


def test_sine_frequency():
    f = parse_expression("sin(x)")
    _, _, y = generate_signal(f, 0, 2 * np.pi * 440, 1.0, 44100)
    spec = np.abs(np.fft.rfft(y))
    assert np.argmax(spec) == 440             # 1 s -> bin k = k Hz


def test_normalize_edge_cases():
    assert np.all(normalize(np.zeros(5)) == 0)
    y = normalize(np.array([0.2, 2.5, -3.1, np.nan, np.inf, -np.inf]))
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0
    assert np.all(normalize(np.array([np.nan, np.nan])) == 0)


def test_pcm_and_wav(tmp_path):
    pcm = to_pcm16(np.array([-1.0, 0.0, 1.0]))
    assert pcm.dtype == np.int16 and list(pcm) == [-32767, 0, 32767]
    p = str(tmp_path / "a.wav")
    write_wav(p, pcm, 8000)
    with wave.open(p) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (8000, 1, 2, 3)
