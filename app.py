
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="MDOF RSA–Pushover Reconciliation PRO", layout="wide")
st.title("MDOF RSA–Pushover Reconciliation PRO")
st.caption("Online Streamlit app: STAAD mass/stiffness → modal RSA → plastic moment capacity → nonlinear MDOF pushover → ADRS reconciliation")

def assemble_K(k):
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] += k[i]
        else:
            K[i, i] += k[i]
            K[i-1, i-1] += k[i]
            K[i, i-1] -= k[i]
            K[i-1, i] -= k[i]
    return K

def modal_analysis(W, k):
    g = 9.80665
    m = W / g
    M = np.diag(m)
    K = assemble_K(k)
    A = np.linalg.inv(M) @ K
    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    omega = np.sqrt(np.maximum(vals, 1e-12))
    T = 2*np.pi/omega

    phi = vecs.copy()
    for j in range(phi.shape[1]):
        if abs(phi[-1, j]) > 1e-12:
            phi[:, j] = phi[:, j] / phi[-1, j]
        if np.sum(phi[:, j]) < 0:
            phi[:, j] *= -1

    one = np.ones(len(W))
    total_m = np.sum(m)
    rows = []
    for j in range(len(W)):
        ph = phi[:, j]
        modal_mass = ph @ M @ ph
        gamma = (ph @ M @ one) / modal_mass
        eff_mass = gamma**2 * modal_mass
        rows.append({
            "Mode": j+1,
            "Omega_rad/s": omega[j],
            "Period_s": T[j],
            "Gamma": gamma,
            "Modal_Mass": modal_mass,
            "Effective_Modal_Mass": eff_mass,
            "Eff_Mass_Ratio_%": eff_mass/total_m*100
        })
    df = pd.DataFrame(rows)
    df["Cumulative_%"] = df["Eff_Mass_Ratio_%"].cumsum()
    return df, phi, K, M

def ubc97_sa(T, Ca, Cv):
    T = np.asarray(T, dtype=float)
    plateau = 2.5 * Ca
    T0 = 0.2 * Cv / plateau if plateau > 0 else 0
    Ts = Cv / plateau if plateau > 0 else 0
    Sa = np.zeros_like(T)
    for i, t in enumerate(T):
        if T0 > 0 and t <= T0:
            Sa[i] = Ca + (plateau - Ca) * t / T0
        elif t <= Ts:
            Sa[i] = plateau
        else:
            Sa[i] = Cv / max(t, 1e-9)
    return Sa

def cqc(values, periods, zeta=0.05):
    values = np.asarray(values, float)
    w = 2*np.pi/np.asarray(periods, float)
    total = 0.0
    for i in range(len(values)):
        for j in range(len(values)):
            beta = w[j] / w[i]
            rho = (8*zeta**2*(1+beta)*beta**1.5)/((1-beta**2)**2 + 4*zeta**2*beta*(1+beta)**2)
            total += rho * values[i] * values[j]
    return np.sqrt(max(total, 0))

def drift_backbone(V, Vy, Vu, k, alpha, residual):
    if k <= 0:
        return 0, "Invalid"
    dy = Vy/k
    if V <= Vy:
        return V/k, "Elastic"
    du = dy + (Vu - Vy)/max(alpha*k, 1e-12)
    if V <= Vu:
        return dy + (V - Vy)/max(alpha*k, 1e-12), "Yielded"
    kres = max(alpha*k*residual, k*0.002)
    return du + (V - Vu)/kres, "Beyond ultimate"

def find_intersection(Sd_cap, Sa_cap, Sd_dem, Sa_dem):
    cap = np.interp(Sd_dem, Sd_cap, Sa_cap, left=np.nan, right=np.nan)
    diff = cap - Sa_dem
    valid = ~np.isnan(diff)
    if valid.sum() < 2:
        return None
    x = Sd_dem[valid]; y = diff[valid]; sa = Sa_dem[valid]
    sign = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
    if len(sign) == 0:
        i = int(np.nanargmin(np.abs(y)))
        return {"Sd_m": x[i], "Sa_g": sa[i], "note": "closest"}
    i = sign[0]
    xx = x[i] - y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])
    return {"Sd_m": xx, "Sa_g": np.interp(xx, Sd_dem, Sa_dem), "note": "intersection"}

st.sidebar.header("Model Basis")
n = st.sidebar.number_input("Number of storeys", 1, 10, 3, 1)
frames = st.sidebar.number_input("Number of similar frames in analyzed axis", 1, 30, 3, 1)
basis = st.sidebar.radio("Plastic moment input basis", ["Per frame (multiply by frames)", "Whole axis total (do not multiply)"])
mult = frames if basis.startswith("Per frame") else 1

alpha = st.sidebar.slider("Post-yield stiffness ratio α", 0.01, 0.25, 0.05, 0.01)
overstrength = st.sidebar.slider("Ultimate / yield shear ratio", 1.0, 2.0, 1.25, 0.05)
residual = st.sidebar.slider("Residual branch stiffness factor", 0.05, 1.0, 0.25, 0.05)

st.sidebar.header("UBC 97 Spectrum")
Ca = st.sidebar.number_input("Ca", value=0.44, step=0.01)
Cv = st.sidebar.number_input("Cv", value=0.768, step=0.01)
R = st.sidebar.number_input("R", value=8.5, step=0.5)
Ie = st.sidebar.number_input("Importance factor I", value=1.0, step=0.1)
zeta = st.sidebar.number_input("CQC damping ratio", value=0.05, min_value=0.01, max_value=0.20, step=0.01)
plot_factor = st.sidebar.slider("Pushover max base shear = factor × first yield", 1.0, 8.0, 3.0, 0.5)
steps = st.sidebar.slider("Pushover steps", 30, 250, 100, 10)

default_W = [1509.12, 1283.42, 579.05] + [500.0]*(max(n-3,0))
default_k = [40090.0, 38510.0, 32400.0] + [25000.0]*(max(n-3,0))
rows = []
for i in range(n):
    rows.append({
        "Storey": i+1,
        "Floor weight W_i (kN)": default_W[i],
        "Storey stiffness k_i (kN/m)": default_k[i],
        "Height h_i (m)": 3.0,
        "Columns per frame": 3,
        "Bays per frame": 2,
        "Column Mp each (kN-m)": 300.0,
        "Beam Mp per end (kN-m)": 150.0,
        "Column participation factor": 1.0,
        "Beam participation factor": 1.0
    })

tabs = st.tabs(["1 Input", "2 Modal RSA", "3 Yield Capacity", "4 Nonlinear Pushover", "5 ADRS", "6 Manual Calc", "7 Downloads"])

with tabs[0]:
    st.subheader("Main input table")
    st.write("Paste STAAD floor weights and storey stiffness. Then input beam/column plastic moments and frame count.")
    df = st.data_editor(pd.DataFrame(rows), use_container_width=True, num_rows="fixed")
    st.warning("Basis rule: mass basis = stiffness basis = plastic moment/yield capacity basis.")

W = df["Floor weight W_i (kN)"].to_numpy(float)
k = df["Storey stiffness k_i (kN/m)"].to_numpy(float)
h = df["Height h_i (m)"].to_numpy(float)
cols = df["Columns per frame"].to_numpy(float)
bays = df["Bays per frame"].to_numpy(float)
Mp_col = df["Column Mp each (kN-m)"].to_numpy(float)
Mp_beam = df["Beam Mp per end (kN-m)"].to_numpy(float)
col_part = df["Column participation factor"].to_numpy(float)
beam_part = df["Beam participation factor"].to_numpy(float)

modal_df, phi, K, M = modal_analysis(W, k)
T = modal_df["Period_s"].to_numpy(float)
Gamma = modal_df["Gamma"].to_numpy(float)

Sa_elastic = ubc97_sa(T, Ca, Cv) * Ie
Sa_design = Sa_elastic / max(R, 1e-9)
W_eff = modal_df["Effective_Modal_Mass"].to_numpy(float) * 9.80665
V_modal_design = Sa_design * W_eff
modal_df["Sa_elastic_g"] = Sa_elastic
modal_df["Sa_design_g"] = Sa_design
modal_df["Mode_Base_Shear_Design_kN"] = V_modal_design
V_srss = float(np.sqrt(np.sum(V_modal_design**2)))
V_cqc = float(cqc(V_modal_design, T, zeta))

phi1 = phi[:, 0]
raw = np.abs(W * phi1)
floor_ratio = raw/raw.sum()
storey_ratio = np.array([floor_ratio[i:].sum() for i in range(n)])

Vy_col_pf = 2 * cols * Mp_col * col_part / h
Vy_beam_pf = bays * 2 * Mp_beam * beam_part / h
Vy_col = Vy_col_pf * mult
Vy_beam = Vy_beam_pf * mult
Vy = np.minimum(Vy_col, Vy_beam)
Vu = overstrength * Vy
dy_mm = Vy/k*1000

first_yield = float(np.min(Vy/np.maximum(storey_ratio,1e-12)))
Vb = np.linspace(0, first_yield*plot_factor, int(steps))
push = []
for v in Vb:
    shears = storey_ratio*v
    drifts=[]; states=[]
    for i in range(n):
        d,s = drift_backbone(shears[i], Vy[i], Vu[i], k[i], alpha, residual)
        drifts.append(d); states.append(s)
    drifts=np.array(drifts)
    disp=np.cumsum(drifts)
    row={"Base shear kN":v, "Roof displacement mm":disp[-1]*1000, "Global drift ratio %":disp[-1]/h.sum()*100,
         "Yielded storeys":", ".join(str(i+1) for i,s in enumerate(states) if s!="Elastic") or "None"}
    for i in range(n):
        row[f"S{i+1} shear kN"] = shears[i]
        row[f"S{i+1} drift mm"] = drifts[i]*1000
        row[f"F{i+1} displacement mm"] = disp[i]*1000
        row[f"S{i+1} state"] = states[i]
    push.append(row)
pushover_df = pd.DataFrame(push)

Gamma1 = Gamma[0]
phi_roof = phi1[-1]
Sd = (pushover_df["Roof displacement mm"].to_numpy()/1000)/max(abs(Gamma1*phi_roof),1e-12)
Sa_cap = pushover_df["Base shear kN"].to_numpy()/W_eff[0]
pushover_df["Sd_m_ADRS"] = Sd
pushover_df["Sa_g_ADRS"] = Sa_cap

T_range = np.linspace(0.02, max(5.0, T[0]*4), 300)
Sa_dem_el = ubc97_sa(T_range, Ca, Cv)*Ie
Sa_dem_des = Sa_dem_el/max(R,1e-9)
Sd_dem_el = Sa_dem_el*9.80665*T_range**2/(4*np.pi**2)
Sd_dem_des = Sa_dem_des*9.80665*T_range**2/(4*np.pi**2)
perf_el = find_intersection(Sd, Sa_cap, Sd_dem_el, Sa_dem_el)
perf_des = find_intersection(Sd, Sa_cap, Sd_dem_des, Sa_dem_des)

with tabs[1]:
    st.subheader("Modal RSA")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("T1", f"{T[0]:.4f} s")
    c2.metric("Mode 1 mass", f"{modal_df.loc[0,'Eff_Mass_Ratio_%']:.2f}%")
    c3.metric("Design RSA SRSS", f"{V_srss:.2f} kN")
    c4.metric("Design RSA CQC", f"{V_cqc:.2f} kN")
    st.dataframe(modal_df.round(4), use_container_width=True)
    fdf = pd.DataFrame({"Floor":np.arange(1,n+1), "W_i kN":W, "phi1_i":phi1, "W_i phi_i":raw, "Floor force ratio":floor_ratio, "Storey shear ratio":storey_ratio})
    st.subheader("First-mode pushover force pattern")
    st.dataframe(fdf.round(5), use_container_width=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=phi1, y=np.arange(1,n+1), mode="lines+markers"))
    fig.update_layout(title="Mode 1 shape normalized to roof = 1", xaxis_title="Mode shape", yaxis_title="Floor")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Yield capacity from plastic moments")
    ydf = pd.DataFrame({"Storey":np.arange(1,n+1), "Column Vy per frame kN":Vy_col_pf, "Beam Vy per frame kN":Vy_beam_pf, "Frame multiplier":mult,
                        "Column Vy total kN":Vy_col, "Beam Vy total kN":Vy_beam, "Governing Vy kN":Vy, "Ultimate Vu kN":Vu,
                        "Yield drift mm":dy_mm, "Yield drift ratio %":(Vy/k)/h*100})
    st.dataframe(ydf.round(4), use_container_width=True)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=ydf["Storey"], y=ydf["Column Vy total kN"], name="Column-controlled"))
    fig.add_trace(go.Bar(x=ydf["Storey"], y=ydf["Beam Vy total kN"], name="Beam-controlled"))
    fig.add_trace(go.Scatter(x=ydf["Storey"], y=ydf["Governing Vy kN"], mode="lines+markers", name="Governing"))
    fig.update_layout(barmode="group", title="Storey yield capacity")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Nonlinear MDOF pushover curve")
    c1,c2,c3=st.columns(3)
    c1.metric("First yield base shear", f"{first_yield:.2f} kN")
    c2.metric("RSA CQC / first yield", f"{V_cqc/first_yield:.3f}")
    c3.metric("Max roof displacement", f"{pushover_df['Roof displacement mm'].max():.1f} mm")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pushover_df["Roof displacement mm"], y=pushover_df["Base shear kN"], mode="lines+markers", name="Pushover"))
    fig.add_hline(y=V_cqc, line_dash="dash", annotation_text="RSA design CQC")
    for i in range(n):
        for label, cap in [("Yield", Vy[i]), ("Ultimate", Vu[i])]:
            vb = cap/max(storey_ratio[i],1e-12)
            if vb <= Vb[-1]:
                shears=storey_ratio*vb
                roof=sum(drift_backbone(shears[j], Vy[j], Vu[j], k[j], alpha, residual)[0] for j in range(n))*1000
                fig.add_trace(go.Scatter(x=[roof], y=[vb], mode="markers+text", text=[f"S{i+1} {label}"], textposition="top center", name=f"S{i+1} {label}"))
    fig.update_layout(title="Base shear vs roof displacement", xaxis_title="Roof displacement (mm)", yaxis_title="Base shear (kN)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pushover_df.round(4), use_container_width=True)

with tabs[4]:
    st.subheader("ADRS reconciliation")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=Sd, y=Sa_cap, mode="lines+markers", name="Capacity spectrum"))
    fig.add_trace(go.Scatter(x=Sd_dem_el, y=Sa_dem_el, mode="lines", name="Elastic demand"))
    fig.add_trace(go.Scatter(x=Sd_dem_des, y=Sa_dem_des, mode="lines", name=f"Design demand /R={R}"))
    if perf_el:
        fig.add_trace(go.Scatter(x=[perf_el["Sd_m"]], y=[perf_el["Sa_g"]], mode="markers+text", text=["Elastic PP"], textposition="top center", name="Elastic PP"))
    if perf_des:
        fig.add_trace(go.Scatter(x=[perf_des["Sd_m"]], y=[perf_des["Sa_g"]], mode="markers+text", text=["Design PP"], textposition="bottom center", name="Design PP"))
    fig.update_layout(title="Capacity spectrum vs demand spectrum", xaxis_title="Sd (m)", yaxis_title="Sa (g)")
    st.plotly_chart(fig, use_container_width=True)
    pr=[]
    for name,p in [("Elastic demand", perf_el), (f"Design demand R={R}", perf_des)]:
        if p:
            pr.append({"Case":name, "Sd m":p["Sd_m"], "Sa g":p["Sa_g"], "Roof displacement mm":p["Sd_m"]*Gamma1*phi_roof*1000, "Result":p["note"]})
    if pr:
        st.dataframe(pd.DataFrame(pr).round(5), use_container_width=True)

with tabs[5]:
    st.subheader("Manual calculation equations")
    st.markdown(r"""
\[
F_i=V_b\frac{W_i\phi_i}{\sum W_i\phi_i}
\]

\[
V_{y,col}=\frac{2\sum M_{p,col}}{h}
\]

\[
V_{y,beam}=\frac{\sum M_{p,beam}}{h}
\]

\[
\delta=\frac{V}{k}
\]

\[
\delta=\delta_y+\frac{V-V_y}{\alpha k}
\]

\[
S_d=\frac{\Delta_{roof}}{\Gamma_1\phi_{roof}}
\]

\[
S_a=\frac{V_b}{W_{eff,1}}
\]
""")
    ex=st.number_input("Base shear for row check (kN)", value=float(first_yield), step=10.0)
    ex_rows=[]
    for i in range(n):
        V=storey_ratio[i]*ex
        d,s=drift_backbone(V,Vy[i],Vu[i],k[i],alpha,residual)
        ex_rows.append({"Storey":i+1, "Storey shear ratio":storey_ratio[i], "Storey shear kN":V, "Vy kN":Vy[i], "Vu kN":Vu[i], "drift mm":d*1000, "state":s})
    st.dataframe(pd.DataFrame(ex_rows).round(5), use_container_width=True)

with tabs[6]:
    st.subheader("Downloads")
    ydf = pd.DataFrame({"Storey":np.arange(1,n+1), "Vy_col_total":Vy_col, "Vy_beam_total":Vy_beam, "Vy_governing":Vy, "Vu":Vu, "Yield_drift_mm":dy_mm})
    st.download_button("Download modal RSA CSV", modal_df.to_csv(index=False).encode(), "modal_rsa.csv", "text/csv")
    st.download_button("Download yield capacity CSV", ydf.to_csv(index=False).encode(), "yield_capacity.csv", "text/csv")
    st.download_button("Download pushover curve CSV", pushover_df.to_csv(index=False).encode(), "pushover_curve.csv", "text/csv")
