# Math2Sound
Expression -> f(x) -> x(t) -> samples -> normalize -> PCM16 -> WAV.

    pip install -r requirements.txt
    streamlit run streamlit_app.py    # UI
    python app.py "sin(x)" --play     # command line
    python -m pytest
