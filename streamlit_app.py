import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="H2S Corrosion Prediction", layout="wide")
st.title("H2S Corrosion Prediction")
st.markdown("Predict H2S/CO2 corrosion rate and remaining pipe life.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("corrosion", "corrosion_rate_model.pkl"), ("life", "remaining_life_model.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
h2s_ppm = st.sidebar.slider("H2S Ppm", 0, 500, 250)
co2_pct = st.sidebar.slider("Co2 Pct", 0, 20, 10)
temp_c = st.sidebar.slider("Temp C", 20, 150, 85)
pressure_mpa = st.sidebar.slider("Pressure Mpa", 0, 50, 25)
ph = st.sidebar.slider("Ph", 3, 10, 6)
flow_velocity_ms = st.sidebar.slider("Flow Velocity Ms", 0, 10, 5)
pipe_material = st.sidebar.selectbox("Pipe Material", ['carbon_steel', 'stainless_steel', 'duplex', 'inconel'])

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[h2s_ppm, co2_pct, temp_c, pressure_mpa, ph, flow_velocity_ms, pipe_material]])
        m = models["corrosion"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Corrosion", result if isinstance(result, str) else f"{result:.4f}")
        m = models["life"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Life", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

