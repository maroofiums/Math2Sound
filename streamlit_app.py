"""Streamlit app.   Run:  streamlit run streamlit_app.py
Calls src.service.synthesize() directly (no separate backend)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.service import synthesize

PRESETS = {
    "sin(x)  (440 Hz)": ("sin(x)", "0", "2*pi*440*2", 2.0),
    "sin(x)+cos(2*x)": ("sin(x)+cos(2*x)", "0", "2*pi*220*2", 2.0),
    "harmonics": ("sin(x)+0.5*sin(3*x)+0.25*sin(5*x)", "0", "2*pi*220*2", 2.0),
    "x^2": ("x^2", "-1", "1", 1.0),
    "1/x": ("1/x", "-1", "1", 1.0),
    "sqrt(abs(x))": ("sqrt(abs(x))", "-1", "1", 1.0),
    "chirp sin(x^2)": ("sin(x^2)", "0", "150", 2.0),
    "tan(x)": ("tan(x)", "0", "2*pi*110*2", 2.0),
}
KEYS = [("x", "x"), ("x\u00b2", "^2"), ("x\u207f", "^"), ("\u00f7", "/"), ("\u00d7", "*"), ("+", "+"), ("\u2212", "-"), ("(", "("), (")", ")"),
        ("\u221a", "sqrt("), ("|a|", "abs("), ("\u03c0", "pi"), ("e", "e"), ("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("exp", "exp("), ("ln", "log("),
        ("7", "7"), ("8", "8"), ("9", "9"), ("4", "4"), ("5", "5"), ("6", "6"), ("1", "1"), ("2", "2"), ("3", "3"),
        ("0", "0"), (".", "."), ("\u232b", "@bs"), ("clear", "@clr")]

st.set_page_config(page_title="Math2Sound", layout="wide")
for k, v in dict(expr="sin(x)+0.5*sin(3*x)", xmin="0", xmax="2*pi*220*2", T=2.0).items():
    st.session_state.setdefault(k, v)


def set_preset():
    e, a, b, t = PRESETS[st.session_state.preset]
    st.session_state.update(expr=e, xmin=a, xmax=b, T=t)


def press(tok: str):
    s = st.session_state
    s.expr = "" if tok == "@clr" else s.expr[:-1] if tok == "@bs" else s.expr + tok


@st.cache_data(show_spinner=False)
def run(expression, x_min, x_max, duration, fs, remove_dc, window_ms):
    try:
        return synthesize(expression, x_min, x_max, duration, fs, remove_dc, window_ms)
    except (ValueError, SyntaxError, RecursionError) as e:
        return {"error": str(e) or "Invalid expression"}


def nan(a):
    return np.array([np.nan if v is None else v for v in a], dtype=float)


st.title("Math2Sound")
st.caption("Expression, f(x), sampled x(t), normalization, 16-bit PCM, WAV. No machine learning.")

st.selectbox("Preset", list(PRESETS), index=None, placeholder="Choose a preset",
             key="preset", on_change=set_preset)
preview = st.container()
st.text_input("f(x)  (use ^ for powers, / for fractions, sqrt( ) for roots)", key="expr")
with st.expander("Keypad", expanded=True):
    for r in range(0, len(KEYS), 9):
        for i, (c, (label, tok)) in enumerate(zip(st.columns(9), KEYS[r:r + 9])):
            c.button(label, key=f"k{r + i}", on_click=press, args=(tok,), use_container_width=True)

c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
c1.text_input("x min", key="xmin")
c2.text_input("x max", key="xmax")
c3.number_input("Duration T (s)", 0.1, 10.0, step=0.5, key="T")
fs = c4.selectbox("Sample rate", [8000, 22050, 44100, 48000], index=2)
dc = c5.checkbox("Remove DC offset", True)
win = st.select_slider("Waveform window (ms, 0 = full)", [0, 5, 20, 100, 500], value=20)

d = run(st.session_state.expr, st.session_state.xmin, st.session_state.xmax,
        float(st.session_state.T), fs, dc, float(win))
if "error" in d:
    preview.error(d["error"])
    st.stop()

preview.latex("y = " + d["latex"])
st.audio(d["wav"], format="audio/wav")
st.download_button("Download WAV", d["wav"], "math2sound.wav", "audio/wav")

m = st.columns(6)
m[0].metric("Samples N = Fs*T", f"{d['n_samples']:,}")
m[1].metric("Nyquist", f"{d['nyquist']:.0f} Hz")
m[2].metric("Non-finite values", d["nonfinite"])
m[3].metric("Peak |y| (raw)", f"{d['raw_peak']:.4g}")
m[4].metric("PCM range", f"{d['pcm_min']} to {d['pcm_max']}")
m[5].metric("Dominant freq", f"{d['dominant_hz']:.1f} Hz")
st.write(f"Under this mapping sin(x) has frequency (x_max - x_min) / (2*pi*T) = **{d['sin_hz']:.1f} Hz**.")
if d["aliased"]:
    st.warning("That exceeds the Nyquist frequency, so it aliases: frequencies above Fs/2 fold back down.")

p = st.columns(3)
x, y = nan(d["curve_x"]), nan(d["curve_y"])
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, y)
if np.isfinite(y).any():
    lo, hi = np.percentile(y[np.isfinite(y)], [1, 99])
    pad = (hi - lo) * .08 or 1.0
    ax.set_ylim(lo - pad, hi + pad)
ax.set_title("f(x)"); ax.set_xlabel("x"); ax.grid(alpha=.3)
p[0].pyplot(fig); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(d["wave_t_ms"], d["wave_y"], color="tab:orange")
ax.set_title("waveform y(t)"); ax.set_xlabel("ms"); ax.set_ylim(-1.05, 1.05); ax.grid(alpha=.3)
p[1].pyplot(fig); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3))
ax.vlines(d["spec_f"], 0, d["spec_mag"], color="tab:green", linewidth=1)
ax.set_title("spectrum"); ax.set_xlabel("Hz"); ax.grid(alpha=.3)
p[2].pyplot(fig); plt.close(fig)
