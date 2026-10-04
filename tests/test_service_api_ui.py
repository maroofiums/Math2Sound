import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from src.latex import to_latex
from src.service import synthesize


def test_latex():
    assert to_latex("1/x") == r"\frac{1}{x}"
    assert to_latex("x^2") == r"x^{2}"
    assert to_latex("sqrt(x^2+1)") == r"\sqrt{x^{2}+1}"
    assert to_latex("2*pi*x") == r"2 \pi x"
    assert to_latex("sin(x)^2") == r"\sin\left(x\right)^{2}"
    assert to_latex("-(x+1)") == r"-\left(x+1\right)"


def test_service_440hz():
    d = synthesize("sin(x)", "0", "2*pi*440", 1.0, 44100)
    assert abs(d["dominant_hz"] - 440) < 2 and not d["aliased"]
    assert d["wav"][:4] == b"RIFF" and d["pcm_max"] <= 32767


def test_service_edge_cases_and_errors():
    d = synthesize("1/x", "-1", "1", 1.0, 8000)
    assert d["nonfinite"] >= 1 and None not in d["wave_y"]
    assert synthesize("sin(x)", "0", "2*pi*5000", 1.0, 8000)["aliased"]
    for bad in [dict(expression="foo(x)"), dict(expression="x", x_max="0"), dict(expression="x", duration=99)]:
        with pytest.raises(ValueError):
            synthesize(**bad)


def test_streamlit_app_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    at.text_input(key="expr").set_value("1/x").run()
    assert not at.exception and "frac" in at.latex[0].value
    at.text_input(key="expr").set_value("foo(").run()
    assert at.error
