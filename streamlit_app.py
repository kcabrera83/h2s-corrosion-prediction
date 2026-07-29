import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="H2S Corrosion Prediction", layout="wide")
st.title("H2S Corrosion Prediction")
st.markdown("Predict H2S/CO2 corrosion rate and remaining pipe life.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'corrosion': joblib.load(d / 'corrosion_rate_model.pkl'), 'life': joblib.load(d / 'remaining_life_model.pkl')}

st.sidebar.header("Input Parameters")
h2s_ppm = st.sidebar.slider('H2S Ppm', 0, 500, 250)
co2_pct = st.sidebar.slider('Co2 Pct', 0, 20, 10)
temp_c = st.sidebar.slider('Temp C', 20, 150, 85)
pressure_mpa = st.sidebar.slider('Pressure Mpa', 0, 50, 25)
ph = st.sidebar.slider('Ph', 3, 10, 6)
flow_velocity = st.sidebar.slider('Flow Velocity', 0, 10, 5)
pipe_material = st.sidebar.selectbox('Pipe Material', ['carbon_steel','stainless_steel','duplex','inconel'])

if st.sidebar.button("Run"):
    try:
        x = np.array([[h2s_ppm, co2_pct, temp_c, pressure_mpa, ph, flow_velocity, pipe_material]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))