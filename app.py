
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("RSA–Pushover PRO (Capacity Spectrum + Performance Point)")

st.markdown("### Input Basic Data")

n = st.number_input("Number of Storeys", 1, 10, 2)

weights = []
stiffness = []

for i in range(n):
    col1, col2 = st.columns(2)
    with col1:
        w = st.number_input(f"W{i+1} (kN)", value=400.0)
    with col2:
        k = st.number_input(f"k{i+1} (kN/m)", value=20000.0)
    weights.append(w)
    stiffness.append(k)

alpha = st.slider("Post-yield stiffness ratio α", 0.01, 0.2, 0.05)

Vy = st.number_input("Global Yield Base Shear (kN)", value=200.0)

st.markdown("### Generate Pushover Curve")

V = np.linspace(0, Vy*3, 50)
disp = []

for v in V:
    if v <= Vy:
        d = v / sum(stiffness)
    else:
        d = Vy / sum(stiffness) + (v - Vy)/(alpha*sum(stiffness))
    disp.append(d*1000)

fig = go.Figure()
fig.add_trace(go.Scatter(x=disp, y=V, mode='lines+markers', name="Pushover"))

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Capacity Spectrum (ADRS)")

Sa = V / sum(weights)
Sd = np.array(disp)/1000

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=Sd, y=Sa, mode='lines', name="Capacity"))

st.plotly_chart(fig2, use_container_width=True)

st.success("PRO Version Running: Includes simplified capacity spectrum.")
