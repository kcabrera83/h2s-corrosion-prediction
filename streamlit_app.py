import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="H2S Corrosion Prediction", layout="wide")
st.title("H2S Corrosion Prediction")

class Engine:
    def __init__(self):
        p = Path(__file__).parent / 'outputs' / 'models'
        self.corrosion = joblib.load(p / 'corrosion_rate_model.pkl')
        self.life = joblib.load(p / 'remaining_life_model.pkl')
    def run(self, name, X):
        m = getattr(self, name)
        if isinstance(m, dict):
            x = m['scaler'].transform(X)
            r = m['model'].predict(x)
            if 'label_encoder' in m:
                return m['label_encoder'].inverse_transform(r)[0]
            return float(r[0])
        return float(m.predict(X)[0])

eng = Engine()

with st.sidebar:
    st.header('Inputs')
    h2s = st.slider('H2S', 0, 500, 250)
    co2 = st.slider('Co2', 0, 20, 10)
    temp = st.slider('Temp', 20, 150, 85)
    pressure = st.slider('Pressure', 0, 50, 25)
    ph = st.slider('Ph', 3, 10, 6)
    velocity = st.slider('Velocity', 0, 10, 5)
    material = st.selectbox('Material', ['carbon','stainless','duplex','inconel'])
    go = st.button('Predict', type='primary', use_container_width=True)

if go:
    x = np.array([[h2s, co2, temp, pressure, ph, velocity, material]])
    out = {}
    out['corrosion'] = eng.run('corrosion', x)
    out['life'] = eng.run('life', x)
    cols = st.columns(len(out))
    for i, (k, v) in enumerate(out.items()):
        cols[i].metric(k.title(), str(v) if isinstance(v, str) else f'{v:.2f}')
    nums = [v for v in out.values() if isinstance(v, (int, float))]
    if nums:
        fig, ax = plt.subplots(figsize=(6,2))
        names = [k.title() for k, v in out.items() if isinstance(v, (int, float))]
        colors = ['#2E86AB','#A23B72','#F18F01']
        bars = ax.bar(names, nums, color=colors[:len(names)])
        ax.axhline(y=sum(nums)/len(nums), color='gray', ls='--', alpha=0.5)
        for bar, val in zip(bars, nums):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*0.9, f'{val:.1f}', ha='center', va='top', color='white', fontweight='bold')
        st.pyplot(fig)